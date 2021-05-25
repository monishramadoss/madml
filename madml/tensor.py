from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import struct
from concurrent.futures import ThreadPoolExecutor
from typing import List, Union, Optional

import numpy as np

import vknn

global TENSOR_EXECUTOR
TENSOR_EXECUTOR = ThreadPoolExecutor(max_workers=os.cpu_count() - 1)


# from .nn.module import module_cache, execution_order

def _convert_to_np_dtype(type):
    if type == int:
        return np.int32
    elif type == float:
        return np.float32
    elif type == bool:
        return np.bool
    elif type == bytes:
        return np.chararray
    elif type == str:
        return np.str
    else:
        return np.float32

def _convert_to_float(size: int, arr: List[bytes]) -> List[float]:
    ret_data = []
    ret_data.extend([bytearray(arr[i:i + 4]) for i in range(0, size, 4)])
    for i in range(len(ret_data)):
        ret_data[i] = struct.unpack("f", ret_data[i])
    return ret_data

def _download(host: np.ndarray, device: vknn.tensor) -> None:
    if host.dtype == np.float32:
        return vknn.tensor_to_np_float(device, host)
    elif host.dtype == np.float64:
        host = host.astype(np.float32)
        return _download(host, device)
    elif host.dtype == np.int32 or host.dtype == np.uint32:
        return vknn.tensor_to_np_int(device, host)
    elif host.dtype == np.uint64:
        host = host.astype(np.uint32)
        return _download(host, device)
    elif host.dtype == np.int64:
        host = host.astype(np.int32)
        return _download(host, device)
    else:
        raise TypeError(" dtype: {0} is not implement.".format(host.dtype))

def _upload(host: np.ndarray, device: vknn.tensor) -> None:
    if host.dtype == np.float32:
        return vknn.np_to_tensor_float(device, host)
    elif host.dtype == np.float64:
        host = host.astype(np.float32)
        return _upload(host, device)
    elif host.dtype == np.int32 or host.dtype == np.uint32:
        return vknn.np_to_tensor_int(device, host)
    elif host.dtype == np.uint64:
        host = host.astype(np.uint32)
        return _upload(host, device)
    elif host.dtype == np.int64:
        host = host.astype(np.int32)
        return _upload(host, device)
    else:
        raise TypeError(" dtype: {0} is not Implemented".format(host.dtype))

class gpu_tensor(object):
    def __init__(self, data: np.ndarray):
        if data.dtype == np.float32:
            self.data = vknn.init_float(data)
        elif data.dtype == np.int32 or data.dtype == np.uint32:
            self.data = vknn.init_int(data)
        elif data.dtype == np.bool:
            self.data = vknn.init_bool(data)
        elif data.dtype == np.float64:
            data = data.astype(np.float32)
            self.data = vknn.init_float(data)
        elif data.dtype == np.int64 or data.dtype == np.uint64:
            data = data.astype(np.int32)
            self.data = vknn.init_int(data)
        elif data.dtype == np.bytes or data.dtype == np.uint8:
            self.data = vknn.init_char(data)
        else:
            raise TypeError(" dtype: {1} is not implement.".format(data.dtype))

    def reshape(self, new_shape):
        self.data.reshape(new_shape)

class tensor(object):
    shape : List[int]
    _init_shape : List[int]
    _host_memory : np.ndarray
    _device_memory : gpu_tensor
    gpu_access : bool
    cpu_access : bool
    id : int
    device_id : int

    def __init__(self, data: Union[List[Union[float, int, bytes, bool]], np.ndarray], shape=None,
                 requires_grad: bool=True, dtype=float, device_id=-1) -> None:
        if shape is None and not isinstance(data, np.ndarray):
            raise AttributeError("shape is undefined: must initalize with np.ndarray or flat list with shape")

        if not isinstance(data, np.ndarray):
            data = np.array(data)
        else:
            shape = data.shape   
        self.shape = [int(s) for s in shape]
        self._host_memory = data.astype(_convert_to_np_dtype(dtype)).reshape(self.shape)
        self._device_memory = gpu_tensor(self._host_memory)
        self._init_shape = self.shape
        self.size = 1

        self.gpu_access = False
        self.cpu_access = False

        for s in self.shape:
            self.size *= s

        self.previous = []
        self.next = []

        self.id = id(self)
        self.requires_grad = requires_grad
        assert (len(self.shape) > 0)
        assert (self._host_memory.size == self.size)
        self._future = None
        self.m_type = 'tensor'

        num_devices = vknn.number_physcial_devices()
        if num_devices != 0:
            device_id = -1
        self.device_id = device_id
        self._grad = None


    def __copy__(self):
        new = tensor(self._host_memory, self._init_shape, requires_grad=False)
        new._grad = self._grad
        return new

    def __len__(self):
        return self.shape[0]

    def __str__(self):
        self.download()
        return np.array2string(self._host_memory, formatter={'float_kind':lambda x: "%.4f" % x})

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
        if self.shape != shape:
            self.host_data = self.host_data.reshape(shape)
            if (self.host_data.size != self.size):
                raise Exception("reshape: cannot reshape size {(0},) does not match size ({1},)".format(self._host_memory.size, self.size))
            self.shape = list(self._host_memory.shape)
            self.device_data.reshape(self.shape)
            if self._grad is not None:
                self._grad.reshape(self.shape)

    @property
    def future(self):
        if self._future is None:
            return True
        else:
            self._future.result()
        return self._future.done()

    @future.setter
    def future(self, value) -> None:
        if self._future is None:
            self._future = value
        else:
            _ = self._future.result()
            self._future = value
        
    @property
    def gradient(self):
        if self._grad is None and self.requires_grad:
            self._grad = tensor([0 for i in range(self.size)], self.shape, requires_grad=False)
        elif not self.requires_grad:
            raise Exception("gradient: should not be asking for gradient")
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
        self.future
        if self.gpu_access:
            self.download()
            self.gpu_acess = False
        self.cpu_acess = True
        return self._host_memory

    @host_data.setter
    def host_data(self, value: np.ndarray) -> None:
        assert (value.size == self._host_memory.size)
        self.shape = list(value.shape)
        self._host_memory = value.astype(self._host_memory.dtype)
        self.cpu_access = True

    @property
    def device_data(self) -> vknn.tensor:
        self.future
        if self.cpu_access:
            self.upload()
            self.cpu_access = False
        self.gpu_access = True
        return self._device_memory.data

    @device_data.setter
    def device_data(self, value: vknn.tensor) -> None:
        self._device_memory.data = value
        self.gpu_access = True

    def forward(self, *args, **kwargs):
        for n in self.next:
            if type(n) == type(self):
                n.forward(None)
            else:
                n.forward()
        return self

    def backward(self):
        for n in self.previous:
            n.backward()
        return self._grad

    def reset_shape(self) -> None:
        self.reshape(self._init_shape)

    def zero_grad(self):
        if self.future is not None and not self.future.done():
            self.future.result()

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

    def download(self) -> np.ndarray:
        _download(self._host_memory, self._device_memory.data)
        return self._host_memory

    def upload(self) -> None:
        _upload(self._host_memory, self._device_memory.data)

    def squeeze(axis:int=None) -> None:
        new_shape = self.shape
        if axis is None:
            for i, s in enumerate(new_shape):
                if s == 1:
                    axis = i
                    break
        if self.shape[axis] == 1:
                del new_shape[axis]
        else:
            raise IndexError("Cannot squeeze element that is not one")
        self.reshape(new_shape)

    def unsqueeze(axis:int=0) -> None:
        new_shape = self.shape
        new_shape.insert(axis, 1)
        self.reshape(new_shape)

    def to(idx:int):
        self.device_id = idx
        return self

