from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import backend
from typing import List, Union
import numpy as np
import struct
from madml.nn import get_modules


def _size(shape: List[int]):
    size = 1
    for s in shape:
        size *= s
    return size



class tensor(object):
    shape: List[int]
    size: int
    host_data: List[Union[float, int, bytes, bool]]
    device_data: backend.tensor
    grad_data: [Union[backend.tensor, List[float]]]
    __module__ = 'madml'
    _use_gpu = 0

    def __init__(self, data: Union[np.ndarray, List[Union[float, int, bytes, bool]]], shape: List[int]) -> None:
        if type(data) == np.ndarray:
            self.host_side = data.reshape(-1).tolist()
        else:
            self.host_side = data

        self.size = _size(shape)
        self.shape = shape

        if type(self.host_side[0]) == int:
            for i in range(len(data)):
                self.host_side[i] = float(self.host_side[i])        
        

    def __len__(self) -> int:
        return self.shape[0]

    def size(self) -> int:
        return _size(self.shape)

    def __getitem__(self, idx: int):
        assert(self.size > idx)
        self.backend_layer = None
        new_shape = self.shape[1:]
        new_size = _size(new_shape)
        new_data = list()

        #_data = self._convert_to_float(self.byte_size, self.toHost())
        #for i in range(new_size):
        #    new_data.append(self._data[idx * new_size + i])
        #return tensor(new_data, new_shape)

    def T(self):
        self.host_side = self.host_side.reshape([self.shape[1], self.shape[0]])

    def _convert_to_float(self, size:int, arr:List[bytes]) -> List[float]:
        ret_data = []
        ret_data.extend([bytearray(arr[i:i + 4]) for i in range(0, size, 4)])
        for i in range(len(ret_data)):
            ret_data[i] = struct.unpack("f", ret_data[i])
        return ret_data

    def numpy(self):
        return np.array(self.host_side).reshape(self.shape)
    
    def backward(self):
        _modules = get_modules()
        print(len(_modules))
   


def zeros(shape: List[int]) -> tensor:
    data = [0 for _ in range(_size(shape))]
    return tensor(data, shape)

def zeros_like(T: tensor) -> tensor:
    return zeros(T.shape)

def ones(shape: List[int]) -> tensor:
    data = [1 for _ in range(_size(shape))]
    return tensor(data, shape)

def full_like(T: tensor, val: float) -> tensor:
    data = [val for _ in T.size]
    return tensor(data, T.shape)

def fill(shape: List[int], val: float) -> tensor:
    data = [val for _ in range(_size(shape))]
    return tensor(data, shape)

