from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import sys
import atexit
import queue
from dataclasses import dataclass
from typing import Union
from .data import MP_STATUS_CHECK_INTERVAL

python_exit_status = False
IS_WINDOWS = sys.platform == "win32"

def _set_python_exit_flag():
    global python_exit_status
    python_exit_status = True

atexit.register(_set_python_exit_flag)

class _BaseDatasetFetcher(object):
    def __init__(self, dataset, auto_collation, collate_fn, drop_last):
        self.dataset = dataset
        self.auto_collation = auto_collation
        self.collate_fn = collate_fn
        self.drop_last = drop_last

    def fetch(self, possibly_batched_index):
        raise NotImplementedError()

class _IterableDatasetFetcher(_BaseDatasetFetcher):
    def __init__(self, dataset, auto_collation, collate_fn, drop_last):
        super(_IterableDatasetFetcher, self).__init__(dataset, auto_collation, collate_fn, drop_last)
        self.dataset_iter = iter(dataset)

    def fetch(self, possibly_batched_index):
        if self.auto_collation:
            data = []
            for _ in possibly_batched_index:
                try:
                    data.append(next(self.dataset_iter))
                except StopIteration:
                    break
            if len(data) == 0 or (self.drop_last and len(data) < len(possibly_batched_index)):
                raise StopIteration
        else:
            data = next(self.dataset_iter)
        return self.collate_fn(data)

class _MapDatasetFetcher(_BaseDatasetFetcher):
    def __init__(self, dataset, auto_collation, collate_fn, drop_last):
        super(_MapDatasetFetcher, self).__init__(dataset, auto_collation, collate_fn, drop_last)

    def fetch(self, possibly_batched_index):
        if self.auto_collation:
            data = [self.dataset[idx] for idx in possibly_batched_index]
        else:
            data = self.dataset[possibly_batched_index]
        return self.collate_fn(data)

class _DatasetKind(object):
    Map = 0
    Iterable = 1

    @staticmethod
    def create_fetcher(kind, dataset, auto_collation, collate_fn, drop_last):
        if kind == _DatasetKind.Map:
            return _MapDatasetFetcher(dataset, auto_collation, collate_fn, drop_last)
        else:
            return _IterableDatasetFetcher(dataset, auto_collation, collate_fn, drop_last)

if IS_WINDOWS:
    import ctypes
    from ctypes.wintypes import DWORD, BOOL, HANDLE

    class ManagerWatchdog(object):
        def __init__(self):
            self.manager_pid = os.getppid()
            self.kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)  # type: ignore
            self.kernel32.OpenProcess.argtypes = (DWORD, BOOL, DWORD)
            self.kernel32.OpenProcess.restype = HANDLE
            self.kernel32.WaitForSingleObject.argtypes = (HANDLE, DWORD)
            self.kernel32.WaitForSingleObject.restype = DWORD

            SYNCHRONIZE = 0x00100000
            self.manager_handle = self.kernel32.OpenProcess(SYNCHRONIZE, 0, self.manager_pid)

            if not self.manager_handle:
                raise ctypes.WinError(ctypes.get_last_error())  # type: ignore

            self.manager_dead = False

        def is_alive(self):
            if not self.manager_dead:
                # Value obtained from
                # https://msdn.microsoft.com/en-us/library/windows/desktop/ms687032.aspx
                self.manager_dead = self.kernel32.WaitForSingleObject(self.manager_handle, 0) == 0
            return not self.manager_dead

else:
    class ManagerWatchdog(object):
        def __init__(self):
            self.manager_pid = os.getppid()
            self.manager_dead = False

        def is_alive(self):
            if not self.manager_dead:
                self.manager_dead = os.getppid() != self.manager_pid
            return not self.manager_dead

_worker_info = None

class WorkerInfo(object):
    __initialized = False

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.__keys = tuple(kwargs.keys())
        self.__initialized = True

    def __setattr__(self, key, val):
        if self.__initialized:
            raise RuntimeError("Cannot assign attributes to {} objects".format(self.__class__.__name__))
        return super(WorkerInfo, self).__setattr__(key, val)

    def __repr__(self):
        items = []
        for k in self.__keys:
            items.append('{}={}'.format(k, getattr(self, k)))
        return '{}({})'.format(self.__class__.__name__, ', '.join(items))

def get_worker_info():
    return _worker_info

@dataclass(frozen=True)
class IterableDatasetStopIteration(object):
    worker_id : int

@dataclass(frozen=True)
class ResumeIteration(object):
    pass

def worker_loop(dataset_kind, dataset, index_queue, data_queue, done_event,
                auto_collation, collate_fn, drop_last, seed, init_fn, worker_id,
                num_workers, persistent_workers):
    try:
        # _set_worker_signal_handlers()

        # seed = 123

        global _worker_info
        _worker_info = WorkerInfo(id=worker_id, num_workers=num_workers,
                                  seed=seed, dataset=dataset)

        init_exception = None

        try:
            if init_fn is not None:
                init_fn(worker_id)
        except Exception as e:
            init_exception = Exception("Exception {} in DataLoader worker process {}".format(e, worker_id))

        iteration_end = False

        watchdog = ManagerWatchdog()

        while watchdog.is_alive():
            try:
                r = index_queue.get(timeout=MP_STATUS_CHECK_INTERVAL)
            except queue.Empty:
                continue
            if isinstance(r, ResumeIteration):
                data_queue.put((r, None))
                iteration_end = False

                continue
            elif r is None:
                assert done_event.is_set() or iteration_end
                break
            elif done_event.is_set() or iteration_end:
                continue
            idx, index = r
            data : Union[IterableDatasetStopIteration, Exception]
            if init_exception is not None:
                data = init_exception
                init_exception = None
            else:
                try:
                    fetcher = _DatasetKind.create_fetcher(dataset_kind, dataset, auto_collation, collate_fn, drop_last)
                    data = fetcher.fetch(index)
                except Exception as e:
                    if isinstance(e, StopIteration) and dataset_kind == _DatasetKind.Iterable:
                        data = IterableDatasetStopIteration(worker_id)
                        iteration_end = True
                    else:
                        data = Exception("in DataLoader worker process {}".format(worker_id))
            data_queue.put((idx, data))
            del data, idx, index, r

    except KeyboardInterrupt:
        pass
    if done_event.is_set():
        data_queue.cancel_join_thread()
        data_queue.close()