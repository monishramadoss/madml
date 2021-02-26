from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import bisect
import itertools
import multiprocessing as mp
import queue
from typing import List, Optional, Union, Iterator, Iterable, Sized, Any, Callable

import numpy as np

from .tensor import tensor
from .worker import worker_loop, IterableDatasetStopIteration, python_exit_status, ResumeIteration, _DatasetKind

MP_STATUS_CHECK_INTERVAL = 5.0

class Dataset(object):
    def __getitem__(self, index) -> tensor:
        pass

    def __add__(self, other):
        return ConcatDataset([self, other])

    def __len__(self):
        pass

class ConcatDataset(Dataset):
    datasets : List[Dataset]
    cumulative_sizes : List[int]

    @staticmethod
    def cumsum(sequence) -> List:
        r, s = [], 0
        for e in sequence:
            r.append(len(e) + s)
            s += len(e)
        return r

    def __init__(self, datasets: List[Dataset]):
        super(ConcatDataset, self).__init__()
        self.datasets = datasets
        self.cumulative_sizes = self.cumsum(self.datasets)

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        if idx < 0 and -idx > len(self):
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx += len(self)
            dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
            if dataset_idx == 0:
                sample_idx = idx
            else:
                sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
            return self.datasets[dataset_idx][sample_idx]

class IterableDataset(Dataset):
    def __iter__(self) -> Iterator:
        raise NotImplementedError

    def __add__(self, other: Dataset):
        return ChainDataset([self, other])

class TensorDataset(Dataset):
    tensors : List[tensor, ...]

    def __init__(self, tensors: List[tensor]) -> None:
        assert all(tensors[0].size(0) == t.shape[0] for t in tensors)
        self.tensors = tensors

    def __getitem__(self, index: int):
        return list(t[index] for t in self.tensors)

    def __len__(self):
        return self.tensors[0].shape[0]

class ChainDataset(IterableDataset):
    def __init__(self, datasets: Iterable[Dataset]) -> None:
        super(ChainDataset, self).__init__()
        self.datasets = datasets

    def __iter__(self) -> tensor:
        for d in self.datasets:
            assert isinstance(d, IterableDataset), "ChainDataset only supports IterableDataset"
            for x in d:
                yield x

    def __len__(self):
        total = 0
        for d in self.datasets:
            assert isinstance(d, IterableDataset), "ChainDataset only supports IterableDataset"
            # Cannot verify that all self.datasets are Sized
            total += len(d)  # type: ignore
        return total

class Sampler(object):
    def __init__(self):
        pass

    def __iter__(self):
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

class SequentialSampler(Sampler):
    data_source : Sized

    def __init__(self, data_source):
        super(SequentialSampler, self).__init__()
        self.data_source = data_source

    def __iter__(self) -> Iterator[int]:
        return iter(range(len(self.data_source)))

    def __len__(self) -> int:
        return len(self.data_source)

class RandomSampler(Sampler):
    data_source : Sized
    replacement : bool

    def __init__(self, data_source: Sized, replacement: bool=False, num_samples: Optional[int]=None,
                 generator=None) -> None:
        super(RandomSampler, self).__init__()
        self.data_source = data_source
        self.replacement = replacement
        self._num_samples = num_samples
        self.generator = generator
        if not isinstance(self.replacement, bool):
            raise TypeError("replacement should be a boolean value, but got "
                            "replacement={}".format(self.replacement))

        if self._num_samples is not None and not replacement:
            raise ValueError("With replacement=False, num_samples should not be specified, "
                             "since a random permute will be performed.")

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError("num_samples should be a positive integer "
                             "value, but got num_samples={}".format(self.num_samples))

    @property
    def num_samples(self) -> int:
        # dataset size might change at runtime
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def __iter__(self) -> List[int]:
        n = len(self.data_source)
        if self.generator is None:
            generator = -1
        else:
            generator = self.generator
        if self.replacement:
            for _ in range(self.num_samples // 32):
                yield from np.random.random_sample(size=32).astype(dtype=np.int64).tolist()
            yield from np.random.random_sample(size=(self.num_samples % 32)).astype(dtype=np.int64).tolist()
        else:
            yield from np.random.random_sample(n).astype(np.int64).tolist()

    def __len__(self) -> int:
        return self.num_samples

class BatchSampler(Sampler):
    def __init__(self, sampler: Sampler, batch_size: int, drop_last: bool) -> None:
        super(BatchSampler, self).__init__()
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError("batch_size should be a positive integer value, "
                             "but got batch_size={}".format(batch_size))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self) -> List[int]:
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size  # type: ignore
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size

class _InfiniteConstantSampler(Sampler):
    def __len__(self) -> int:
        pass

    def __init__(self):
        super(_InfiniteConstantSampler, self).__init__()

    def __iter__(self):
        while True:
            yield None

class DataLoader(object):
    dataset : Dataset
    batch_size : Optional[int]
    num_workers : int
    persistent_workers : bool
    prefetch_factor : int
    timeout : int
    drop_last : bool
    shuffle : bool

    def __init__(self, dataset, batch_size: int=1, shuffle: bool=False, drop_last: bool=False,
                 num_workers: int=-1, persistent_workers: bool=False, prefetch_factor: int=-1,
                 timeout: int=2000, sampler: Optional[Sampler]=None, generator=None,
                 worker_init_fn: Callable[[int], None]=None, multiprocessing_context=None) -> None:

        if sampler is None:  # give default samplers
            if isinstance(dataset, Iterable):
                # See NOTE [ Custom Samplers and IterableDataset ]
                sampler = _InfiniteConstantSampler()
            else:  # map-style
                if shuffle:
                    # Cannot statically verify that dataset is Sized
                    # Somewhat related: see NOTE [ Lack of Default `__len__` in
                    # Python Abstract Base Classes ]
                    sampler = RandomSampler(dataset, generator=generator)  # type: ignore
                else:
                    sampler = SequentialSampler(dataset)

        self.sampler = sampler
        self.__multiprocessing_context = None
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.num_workers = num_workers if num_workers > 0 else mp.cpu_count()
        self.batch_sampler = BatchSampler(sampler, batch_size, drop_last)
        self.IterableDataset_len_called = None
        self.timout = timeout
        self.persistent_workers = persistent_workers
        self.prefetch_factor = prefetch_factor
        self.worker_init_fn = worker_init_fn
        self.__multiprocessing_context = multiprocessing_context

        if isinstance(dataset, IterableDataset):
            self.dataset_kind = _DatasetKind.Iterable
            if shuffle is not False:
                raise ValueError("DataLoader with IterableDataset: expected unspecified "
                    "shuffle option, but got shuffle={}".format(shuffle))
            elif sampler is not None:
                raise ValueError("DataLoader with IterableDataset: expected unspecified "
                    "sampler option, but got sampler={}".format(sampler))
            elif self.batch_sampler is not None:
                raise ValueError("DataLoader with IterableDataset: expected unspecified "
                    "batch_sampler option, but got batch_sampler={}".format(self.batch_sampler))
        else:
            self.dataset_kind = _DatasetKind.Map

        if self.sampler is not None and shuffle:
            raise ValueError('sampler option is mutually exclusive with '
                             'shuffle')

        if self.batch_sampler is not None:
            # auto_collation with custom batch_sampler
            if batch_size != 1 or shuffle or sampler is not None or drop_last:
                raise ValueError('batch_sampler option is mutually exclusive '
                                 'with batch_size, shuffle, sampler, and '
                                 'drop_last')
            self.batch_size = None
            self.drop_last = False
        elif batch_size is None:
            # no auto_collation
            if drop_last:
                raise ValueError('batch_size=None option disables auto-batching '
                                 'and is mutually exclusive with drop_last')

        if self.sampler is None:  # give default samplers
            if self.dataset_kind == _DatasetKind.Iterable:
                # See NOTE [ Custom Samplers and IterableDataset ]
                self.sampler = _InfiniteConstantSampler()
            else:  # map-style
                if shuffle:
                    # Cannot statically verify that dataset is Sized
                    # Somewhat related: see NOTE [ Lack of Default `__len__` in
                    # Python Abstract Base Classes ]
                    self.sampler = RandomSampler(dataset, generator=generator)  # type: ignore
                else:
                    self.sampler = SequentialSampler(dataset)

    def _get_iterator(self):
        if self.num_workers == 0:
            return _SingleProcessDataLoaderIter(self)
        else:
            return _MultiProcessingDataLoaderIter(self)

    @property
    def multiprocessing_context(self):
        return self.__multiprocessing_context

    @multiprocessing_context.setter
    def multiprocessing_context(self, multiprocessing_context):
        if multiprocessing_context is not None:
            if self.num_workers > 0:
                if isinstance(multiprocessing_context, str) or isinstance(multiprocessing_context, bytes):
                    valid_start_methods = mp.get_all_start_methods()
                    if multiprocessing_context not in valid_start_methods:
                        raise ValueError(('multiprocessing_context option '
                             'should specify a valid start method in {!r}, but got '
                             'multiprocessing_context={!r}').format(valid_start_methods, multiprocessing_context))

            multiprocessing_context = mp.get_context(multiprocessing_context)
            if not isinstance(multiprocessing_context, mp.context.BaseContext):
                raise TypeError(('multiprocessing_context option should be a valid context '
                                 'object or a string specifying the start method, but got '
                                 'multiprocessing_context={}').format(multiprocessing_context))

            else:
                raise ValueError(('multiprocessing_context can only be used with '
                                  'multi-process loading (num_workers > 0), but got '
                                  'num_workers={}').format(self.num_workers))
        self.__multiprocessing_context = multiprocessing_context

    def __iter__(self):
        if self.num_workers > 0:
            if self._iterator is None:
                self._iterator = self._get_iterator()
            else:
                self._iterator.reset(self)
            return self._iterator
        else:
            return self._get_iterator()

    @property
    def auto_collation(self):
        return self.batch_sampler is not None

    @property
    def index_sampler(self):
        if self.auto_collation:
            return self.batch_sampler
        else:
            return self.sampler

    def __len__(self) -> int:
        if self.dataset_kind == _DatasetKind.Iterable:
            length = self._IterableDataset_len_called = len(self.dataset)  # type: ignore
            if self.batch_size is not None:  # IterableDataset doesn't allow custom sampler or batch_sampler
                from math import ceil
                if self.drop_last:
                    length = length // self.batch_size
                else:
                    length = ceil(length / self.batch_size)
            return length
        else:
            return len(self.index_sampler)

class _BaseDataLoaderIter(object):
    def __init__(self, loader: DataLoader) -> None:
        self._dataset = loader.dataset
        self.IterableDataset_len_called = loader.IterableDataset_len_called
        self._dataset_kind = loader.dataset_kind
        self._auto_collation = loader.auto_collation
        self._drop_last = loader.drop_last
        self._index_sampler = loader.index_sampler
        self._num_workers = loader.num_workers
        self._prefetch_factor = loader.prefetch_factor
        self._sampler_iter = iter(self._index_sampler)
        self._base_seed = np.seed(124)
        self._num_yielded = 0
        self._pin_memory = False
        self._timeout = loader.timeout
        self._persistent_workers = loader.persistent_workers

    def __iter__(self):
        return self

    def reset(self, loader, first_iter=False):
        self._sampler_iter = iter(self._index_sampler)
        self._num_yielded = 0
        self.IterableDataset_len_called = loader.IterableDataset_len_called

    def _next_index(self):
        return next(self._sampler_iter)

    def _next_data(self):
        raise NotImplementedError

    def __next__(self) -> Any:
        if self._sampler_iter is None:
            self.reset(self)
        data = self._next_data()
        self._num_yielded += 1
        return data

    def __len__(self) -> int:
        return len(self._index_sampler)

class _SingleProcessDataLoaderIter(_BaseDataLoaderIter):
    def __init__(self, loader) -> None:
        super(_SingleProcessDataLoaderIter, self).__init__(loader)
        assert self._num_workers <= 0
        self._dataset_fetcher = _DatasetKind.create_fetcher(self._dataset_kind, self._dataset, self._auto_collation, self._collate_fn, self._drop_last)

    def _next_data(self):
        index = self._next_index()  # may raise StopIteration
        data = self._dataset_fetcher.fetch(index)
        return data

class _MultiProcessingDataLoaderIter(_BaseDataLoaderIter):
    _send_idx : int

    def __init__(self, loader):
        super(_MultiProcessingDataLoaderIter, self).__init__(loader)

        assert self._num_workers > 0
        assert self._prefetch_factor > 0

        if loader.multiprocessing_context is None:
            multiprocessing_context = mp
        else:
            multiprocessing_context = loader.multiprocessing_context

        self._worker_init_fn = loader.worker_init_fn
        self._worker_queue_idx_cycle = itertools.cycle(range(self._num_workers))
        self._worker_result_queue = multiprocessing_context.Queue()
        self._worker_pids_set = False
        self._shutdown = False
        self._workers_done_event = multiprocessing_context.Event()
        self._colate_fn = None
        self._index_queues = []
        self._workers = []
        for i in range(self._num_workers):
            index_queue = mp.Queue()
            w = multiprocessing_context.Process(target=worker_loop,
                args=(self._dataset_kind, self._dataset, index_queue,
                      self._worker_result_queue, self._workers_done_event,
                      self._auto_collation, self._colate_fn, self._drop_last,
                      self._base_seed + i, self._worker_init_fn, i, self._num_workers, self._persistent_workers))

            w.daemon = True
            w.start()
            self._index_queues.append(index_queue)
            self._workers.append(w)
            self._send_idx = 0
            self._rcvd_idx = 0
            self._task_info = {}
            self._tasks_outstanding = 0
            self._workers_status = [True for i in range(self._num_workers)]

        if self._pin_memory:
            self._pin_memory_thread_done_event = mp.Event()

            # Queue is not type-annotated
            self._data_queue = mp.Queue()  # type: ignore
            # pin_memory_thread = mp.Process(
            #     target=_utils.pin_memory._pin_memory_loop,
                                                     #     args=(self._worker_result_queue,
                                                                                              #     self._data_queue,
                                                                                                                                       #     self._pin_memory_thread_done_event)
            # )
                                                     # pin_memory_thread.daemon
                                                                                              # =
                                                                                                                                       # True
                                                                                                                                       # pin_memory_thread.start()
                                                                                                                                       # Similar
                                                                                                                                       # to
                                                                                                                                       # workers
                                                                                                                                       # (see
                                                                                                                                       # comment
                                                                                                                                       # above),
                                                                                                                                       # we
                                                                                                                                       # only
                                                                                                                                       # register
                                                                                                                                       # pin_memory_thread
                                                                                                                                       # once
                                                                                                                                       # it
                                                                                                                                       # is
                                                                                                                                       # started.
                                                                                                                                       # self._pin_memory_thread
                                                                                                                                       # =
                                                                                                                                       # pin_memory_thread
        else:
            self._data_queue = self._worker_result_queue

        # _utils.signal_handling._set_worker_pids(id(self), tuple(w.pid for w
        # in self._workers)) # type: ignore
        # _utils.signal_handling._set_SIGCHLD_handler()
        self._worker_pids_set = True
        self.reset(loader, first_iter=True)

    def reset(self, loader, first_iter=False):
        super().reset(loader, first_iter)
        self._send_idx = 0
        self._rcvd_idx = 0
        self._task_info = {}
        self._tasks_outstanding = 0
        self._workers_status = [True for i in range(self._num_workers)]
        if not first_iter:
            for idx in range(self._num_workers):
                self._index_queues[idx].put(ResumeIteration())
            resume_iteration_cnt = self._num_workers
            while resume_iteration_cnt > 0:
                data = self._get_data()
                if isinstance(data, ResumeIteration):
                    resume_iteration_cnt -= 1
            # prime the prefetch loop
        for _ in range(self._prefetch_factor * self._num_workers):
            self._try_put_index()

    def _try_get_data(self, timeout=MP_STATUS_CHECK_INTERVAL):
        try:
            data = self._data_queue.get(timeout=timeout)
            return True, data
        except Exception as e:
            # At timeout and error, we manually check whether any worker has
            # failed.  Note that this is the only mechanism for Windows to
            # detect
            # worker failures.
            failed_workers = []
            for worker_id, w in enumerate(self._workers):
                if self._workers_status[worker_id] and not w.is_alive():
                    failed_workers.append(w)
                    self._mark_worker_as_unavailable(worker_id)
            if len(failed_workers) > 0:
                pids_str = ', '.join(str(w.pid) for w in failed_workers)
                raise RuntimeError('DataLoader worker (pid(s) {}) exited unexpectedly'.format(pids_str)) from e
            if isinstance(e, queue.Empty):
                return False, None
            import tempfile
            import errno
            try:
                # Raise an exception if we are this close to the FDs limit.
                # Apparently, trying to open only one file is not a sufficient
                # test.
                # See NOTE [ DataLoader on Linux and open files limit ]
                fds_limit_margin = 10
                fs = [tempfile.NamedTemporaryFile() for i in range(fds_limit_margin)]
            except OSError as e:
                if e.errno == errno.EMFILE:
                    raise RuntimeError("Too many open files. Communication with the"
                        " workers is no longer possible. Please increase the"
                        " limit using `ulimit -n` in the shell or change the"
                        " sharing strategy by calling"
                        " `torch.multiprocessing.set_sharing_strategy('file_system')`"
                        " at the beginning of your code") from None
            raise

    def _get_data(self):
        if self._timeout > 0:
            success, data = self._try_get_data(self._timeout)
            if success:
                return data
            else:
                raise RuntimeError('DataLoader timed out after {} seconds'.format(self._timeout))
        # elif self._pin_memory:
        #     while self._pin_memory_thread.is_alive():
        #         success, data = self._try_get_data()
        #         if success:
        #             return data
        #     else:
        #         raise RuntimeError('Pin memory thread exited unexpectedly')
        else:
            while True:
                success, data = self._try_get_data()
                if success:
                    return data

    def _next_data(self):
        while True:
            while self._rcvd_idx < self._send_idx:
                info = self._task_info[self._rcvd_idx]
                worker_id = info[0]
                if len(info) == 2 or self._workers_status[worker_id]:
                    break
                del self._task_info[self._rcvd_idx]
                self._rcvd_idx += 1
            else:
                if not self._persistent_workers:
                    self._shutdown_workers()
                raise StopIteration
            if len(self._task_info[self._rcvd_idx]) == 2:
                data = self._task_info.pop(self._rcvd_idx)[1]
                return self._process_data(data)

            assert not self._shutdown and self._tasks_outstanding > 0
            idx, data = self._get_data()
            self._tasks_outstanding -= 1
            if isinstance(data, IterableDatasetStopIteration):
                if self._persistent_workers:
                    self._workers_status[data.worker_id] = False
                else:
                    self._mark_worker_as_unavailable(data.worker_id)
                self._try_put_index()
                continue

            if idx != self._rcvd_idx:
                # store out-of-order samples
                self._task_info[idx] += (data,)
            else:
                del self._task_info[idx]
                return self._process_data(data)

    def _try_put_index(self):
        assert self._tasks_outstanding < self._prefetch_factor * self._num_workers
        try:
            index = self._next_index()
        except StopIteration:
            return
        for _ in range(self._num_workers):
            worker_queue_idx = next(self._worker_queue_idx_cycle)
            if self._workers_status[worker_queue_idx]:
                break
        else:
            # not found (i.e., didn't break)
            return

        self._index_queues[worker_queue_idx].put([self._send_idx, index])
        self._task_info[self._send_idx] = (worker_queue_idx,)
        self._tasks_outstanding += 1
        self._send_idx += 1

    def _process_data(self, data):
        self._rcvd_idx += 1
        self._try_put_index()
        if isinstance(data, Exception):
            raise data
        return data

    def _mark_worker_as_unavailable(self, worker_id, shutdown=False):
        assert self._workers_status[worker_id] or (self._persistent_workers and shutdown)
        q = self._index_queues[worker_id]
        q.put(None)
        self._workers_status[worker_id] = False
        assert self._workers_done_event.is_set() == shutdown

    def _shutdown_workers(self):
        if python_exit_status is True or python_exit_status is None:
            return
        if not self._shutdown:
            self._shutdown = True
            try:
                if hasattr(self, '_pin_memory_thread'):
                    self._pin_memory_thread_done_event.set()
                    self._worker_result_queue.put((None, None))
                    self._pin_memory_thread.join()
                    self._worker_result_queue.cancel_join_thread()
                    self._worker_result_queue.close()
                self._workers_done_event.set()
                for worker_id in range(len(self._workers)):
                    if self._persistent_workers or self._workers_status[worker_id]:
                        self._mark_worker_as_unavailable(worker_id, shutdown=True)
                    for w in self._workers:
                        w.join(timeout=MP_STATUS_CHECK_INTERVAL)
                        if w.is_alive():
                            w.terminate()
                for q in self._index_queues:
                    q.cancel_join_thread()
                    q.close()

            finally:
                if self._worker_pids_set:
                    # _utils.signal_handling._remove_worker_pids(id(self))
                    self._worker_pids_set = False

    def __del__(self):
        self._shutdown_workers()