from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import struct
from typing import List, Union, Optional, Type

import numpy as np
import backend

from concurrent.futures import ThreadPoolExecutor

global EXECUTOR
EXECUTOR = ThreadPoolExecutor(2)


# from .nn.module import module_cache, execution_order
def _convert_to_float(size: int, arr: List[bytes]) -> List[float]:
    ret_data = []
    ret_data.extend([bytearray(arr[i:i + 4]) for i in range(0, size, 4)])
    for i in range(len(ret_data)):
        ret_data[i] = struct.unpack("f", ret_data[i])
    return ret_data


def async_download(executor: ThreadPoolExecutor, host: np.ndarray, device: backend.tensor):
    if host.dtype == np.float32:
        return executor.submit(backend.tensor_to_np_float, (device, host))
        return None
    else:
        raise TypeError(" dtype: {0} is not implement.".format(host.dtype))

def async_upload(executor: ThreadPoolExecutor, host: np.ndarray, device: backend.tensor):
    if host.dtype == np.float32:
        return executor.submit(backend.np_to_tensor_float, (device, host))
        
    else:
        raise TypeError(" dtype: {0} is not Implemented".format(host.dtype))
    

class gpu_tensor(object):
    def __init__(self, data: np.ndarray):
        if data.dtype != np.float32:
            raise TypeError(" dtype: {1} is not implement.".format(data.dtype))
        lst_data = data.astype(float)
        self.data = backend.init_float(lst_data)
    

class tensor(object):
    shape : List[int]
    _init_shape : List[int]
    _host_memory : np.ndarray
    _device_memory : gpu_tensor
    on_device : bool
    id : int

    def __init__(self, data: Union[List[Union[float, int, bytes, bool]], np.ndarray], shape=None,
                 requires_grad: bool=True, dtype=float) -> None:
        if shape is None:
            shape = []
        if isinstance(data, np.ndarray):
            self._host_memory = data.astype(np.float32)
            self.shape = list(data.shape)
        else:
            self._host_memory = np.array(data).reshape(shape).astype(np.float32)
            self.shape = shape

        self._device_memory = gpu_tensor(self._host_memory)
        self._init_shape = self.shape
        self.size = 1

        for s in self.shape:
            self.size *= s

        self.on_device = True
        self.parent = []
        self.children = []
        self.id = id(self)
        if requires_grad:
            self._grad = tensor([0 for i in range(self.size)], self.shape, requires_grad=False)
        else:
            self._grad = None
        assert (len(self.shape) > 0)
        assert (self._host_memory.size == self.size)
        self._future_obj = None
        self.executor =  EXECUTOR  

    def __copy__(self):
        new = tensor(self._host_memory, self._init_shape, requires_grad=False)
        new._grad = self._grad
        return new

    def __len__(self):
        return self.shape[0]

    def T(self):
        assert len(self.shape) == 2
        self._host_memory = self._host_memory.T
        self.shape = list(self._host_memory.shape)
        return self

    def numpy(self) -> np.ndarray:
        return self._host_memory

    def __getitem__(self, idx: int):
        assert (self.shape[0] > idx)
        new_data = self._host_memory[idx]
        new_shape = self.shape[1:]
        return tensor(new_data, new_shape)

    def __setitem__(self, key: int, value) -> None:
        assert (self.size > key)
        assert (type(value) == type(self))
        self._host_memory[key] = value.host_data

    def reshape(self, shape: List[int]) -> None:
        self._host_memory = self._host_memory.reshape(shape)
        assert (self._host_memory.size == self.size)
        self.shape = list(self._host_memory.shape)
        if self._grad is not None:
            self._grad.reshape(self.shape)

    @property
    def gradient(self):
        return self._grad

    @gradient.setter
    def gradient(self, value) -> None:
        assert (type(value) == type(self))
        assert (self._grad.size == value.size)
        self._grad = value

    @property
    def grad_data(self) -> np.ndarray:
        _grad = self.gradient.host_data
        return _grad.ravel()

    @property
    def host_data(self) -> np.ndarray:
        if self._future_obj is not None and  self._future_obj.done():
            self._future_obj.result()
        self._future_obj = async_upload(self.executor, self._host_memory, self._device_memory.data)
        return self._host_memory

    @host_data.setter
    def host_data(self, value: np.ndarray) -> None:
        assert (value.size == self._host_memory.size)
        self.shape = list(value.shape)
        self._host_memory = value.astype(self._host_memory.dtype)
        if self._future_obj is not None and  self._future_obj.done():
            self._future_obj.result()
        self._future_obj = async_upload(self.executor, self._host_memory, self._device_memory.data)
       

    @property
    def device_data(self) -> backend.tensor:
        if self._future_obj is not None and  self._future_obj.is_alive():
            self._future_obj.result()
        self._future_obj = async_download(self.executor, self._host_memory, self._device_memory.data)
        
        return self._device_memory.data

    @device_data.setter
    def device_data(self, value: backend.tensor) -> None:
        self._device_memory.data = value
        if self._future_obj is not None and  self._future_obj.done():
            self._future_obj.result()
        self._future_obj = async_download(self.executor, self._host_memory, self._device_memory.data)
        
        

    def backward(self) -> None:
        for x in reversed(self.parent):
            if not x.visited[self.id]:
                y = x.backward()
                x.visited[self.id] = True
                if isinstance(y, tensor):
                    y.backward()
        self.parent.clear()
        self.children.clear()

    def reset_shape(self) -> None:
        self._host_memory = self._host_memory.reshape(self._init_shape)
        self.shape = self._init_shape
        if self._grad is not None:
            self._grad.reshape(self._init_shape)

    def flatten(self) -> None:
        self._host_memory = self._host_memory.reshape([self.shape[0], -1])
        if self._grad is not None:
            self._grad.host_data = self._grad.host_data.reshape([self.shape[0], -1])
        self.shape = list(self._host_memory.shape)

    def transpose(self, axis: List[int]) -> None:
        self._host_memory = self._host_memory.transpose(axis)
        self.shape = list(self._host_memory.shape)

    def zero_grad(self):
        if self._grad is not None:
            self.gradient.host_data = np.zeros_like(self.gradient.host_data)

    def onehot(self, label_count: Optional[int]=-1):
        if label_count > 0:
            _max = label_count
        else:
            _max = (np.max(self._host_memory) + 1).astype(int)
        y = np.zeros([self._host_memory.size, _max])
        self._host_memory = self._host_memory.flatten()
        for i in range(self.size):
            y[i][self._host_memory[i].astype(int)] = 1
        self._host_memory = self._host_memory.reshape(self._init_shape)
        if self._init_shape[-1] == 1:
            y = y.reshape(self._init_shape[:-1] + [_max])
        else:
            y = y.reshape(self._init_shape + [_max])
        return tensor(y, y.shape)