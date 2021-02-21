from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import struct
from typing import List, Union, Optional

import numpy as np


# from .nn.module import module_cache, execution_order

def _convert_to_float(size: int, arr: List[bytes]) -> List[float]:
    ret_data = []
    ret_data.extend([bytearray(arr[i:i + 4]) for i in range(0, size, 4)])
    for i in range(len(ret_data)):
        ret_data[i] = struct.unpack("f", ret_data[i])
    return ret_data


class gpu_tensor(object):
    data: np.ndarray

    def __init__(self, data: np.ndarray):
        self.data = data


class tensor(object):
    shape: List[int]
    init_shape: List[int]
    _host_memory: np.ndarray
    _device_memory: gpu_tensor
    on_device: bool
    id: int

    def __init__(self, data: Union[List[Union[float, int, bytes, bool]], np.ndarray], shape=None,
                 requires_grad: bool = True) -> None:
        if shape is None:
            shape = []
        if isinstance(data, np.ndarray):
            self._host_memory = data.astype(np.float32)
            self.shape = list(data.shape)
        else:
            self._host_memory = np.array(data).reshape(shape).astype(np.float32)
            self.shape = shape

        self._device_memory = gpu_tensor(self._host_memory)
        self.init_shape = self.shape
        self.size = 1

        for s in self.shape:
            self.size *= s

        self.on_device = False
        self.parent = []
        self.children = []
        self.id = id(self)
        if requires_grad:
            self._grad = tensor([0 for i in range(self.size)], self.shape, requires_grad=False)
        else:
            self._grad = None
        assert (len(self.shape) > 0)
        assert (self._host_memory.size == self.size)

    def __copy__(self):
        new = tensor(self._host_memory, self.init_shape)
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

    def copy(self):
        return tensor(self._host_memory, self.shape)

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
        return self._host_memory

    @host_data.setter
    def host_data(self, value: np.ndarray) -> None:
        assert (value.size == self._host_memory.size)
        self.shape = list(value.shape)
        self._host_memory = value.astype(self._host_memory.dtype)

    @property
    def device_data(self) -> np.array:
        return self._device_memory

    @device_data.setter
    def device_data(self, value: np.ndarray) -> None:
        raise NotImplementedError

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
        self._host_memory = self._host_memory.reshape(self.init_shape)
        self.shape = self.init_shape
        if self._grad is not None:
            self._grad.reshape(self.init_shape)

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

    def onehot(self, label_count: Optional[int] = -1):
        if label_count > 0:
            _max = label_count
        else:
            _max = (np.max(self._host_memory) + 1).astype(int)
        y = np.zeros([self._host_memory.size, _max])
        self._host_memory = self._host_memory.flatten()
        for i in range(self.size):
            y[i][self._host_memory[i].astype(int)] = 1
        self._host_memory = self._host_memory.reshape(self.init_shape)
        if self.init_shape[-1] == 1:
            y = y.reshape(self.init_shape[:-1] + [_max])
        else:
            y = y.reshape(self.init_shape + [_max])
        return tensor(y, y.shape)
