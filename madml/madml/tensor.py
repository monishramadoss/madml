from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import backend
from typing import Union, List
import madml

class tensor:
    __constants__ = ['data', 'shape', 'size']
    data : List[float]
    shape : List[int]
    size : int
    def __init__(self, data: List[float], shape: List[int]=None) -> None:
        self.data = data
        if(shape is None):
            self.shape = [len(data)] if type(data) is list else data.shape
        else:
            self.shape = shape
        self.size = self._size(self.shape)

    def reshape(self, shape: List[List]):
        new_size = self._size(shape)
        new_shape = shape
        if new_size == self.size:
            self.shape = shape
        elif new_size < 0:
            t = []
            for i in range(len(shape)):
                if new_shape[i] == -1:
                    new_shape[i] = new_size // self.size
            self.shape = new_shape
        else:
            return ValueError("Cannot reshape, {0} into {1}".format(new_shape, self.shape))
        return self

    def _size(self, shape: List[int]) -> int:
        size = 1
        for s in shape:
            size *= s
        return size

    def __len__(self):
        return self.size

    def __getitem__(self, idx: int):
        assert(self.size > idx)
        new_shape = self.shape[1:]
        new_size = self._size(new_shape)
        new_data = list()
        for i in range(new_size):
            new_data.append(self.data[idx * new_size + i])
        return tensor(new_data, new_shape)

    def __setitem__(self, idx: int, val: Union[tensor, float]):
        assert(self.size > idx)
        if type(val) is float:
            assert(len(self.shape) == 1)
            self.data[idx] = val
        if type(val) is tensor:
            new_data = val.data
            new_shape = self.shape[1:]
            assert(new_shape == val.shape)
            new_size = self._size(new_shape)
            for i in range(new_size):
                self.data[idx * new_size + i] = new_data[i]
        return self

    def __add__(self, T: Union[tensor, float]) -> tensor:
        return madml.add(self, T)

    def __sub__(self, T: Union[tensor, float]) -> tensor:
        return madml.sub(self, T)

    def __mul__(self, T: Union[tensor, float]) -> tensor:
        return madml.mul(self, T)

    def __div__(self, T: Union[tensor, float]) -> tensor:
        return madml.div(self, T)

    def __abs__(self) -> tensor:
        return madml.abs(self)

    def __neg__(self) -> tensor:
        return madml.sign(self)

    def T(self) -> tensor:
        return self.reshape([self.shape[1], self.shape[0]])