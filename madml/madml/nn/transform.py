from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from typing import List
from .module import Module
from madml import tensor
import madml
import numpy as np
import backend

class Transpose(Module):
    __constants__ = ['axes']
    axes : List[int]

    def __init__(self, axes: List[int]=None) -> None:
        self.axes = axes
        #super(transpose, self).__init__(backend.transpose(axes))

    def forward_cpu(self, x: np.ndarray) -> np.ndarray:
        assert(len(x.shape) == len(self.axes))
        return np.transpose(x, axes=self.axes)

    def backward_cpu(self, dy: np.ndarray) -> np.ndarray:
        return np.transpose(dy, axes=self.axes)

class flatten(Module):
    __constants__ = ['old_shape']
    old_shape : List[int]

    def __init__(self) -> None:
        super(flatten, self).__init__()

    def forward_cpu(self, x: np.ndarray) -> np.ndarray:
        self.old_shape = x.shape
        return x.reshape((x.shape[0], -1))

    def backward_cpu(self, dy: np.ndarray) -> np.ndarray:
        return dy.reshape(self.old_shape)

    def forward_gpu(self, x: tensor) -> tensor:
        self.old_shape = x.shape
        return x.reshape((x.shape[0], -1))

    def backward_gpu(self, dy: tensor) -> tensor:
        return dy.reshape(self.old_shape)