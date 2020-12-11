from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

#from typing import Optional
from .module import Module, Parameter
from madml import tensor
import madml
import backend
import numpy as np

class Identity(Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward_cpu(self, input):
        return input
    def forward_gpu(self, input):
        return input
    def backward_cpu(self, input):
        return input
    def backward_gpu(self, input):
        return input


class Linear(Module):
    __constants__ = ['in_features', 'out_features']
    in_features : int
    out_features : int

    def __init__(self, in_features: int, out_features: int, bias: bool=True) -> None:
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter((out_features, in_features), self._use_gpu, True)
        if bias:
            self.bias = Parameter((out_features), self._use_gpu, False)

    def forward_cpu(self, x: np.ndarray) -> np.ndarray:
        y = np.zeros((x.shape[0], self.out_features))
        for i in range(x.shape[0]):
            y[i] = self.weight.data @ x[i]
            if self.bias is not None:
                y[i] = y[i] + self.bias.data   
        self.cache = [x]
        return y

    def forward_gpu(self, x: tensor) -> tensor:
        y = madml.zeros((x.shape[0], self.out_features))
        self.backend(y, x, self.weight.data, self.bias)
        self.cache = [x]
        return y

    def backward_cpu(self, dy: np.ndarray) -> np.ndarray:
        x = self.cache[0]
        if self.bias is not None:
            self.bias.gradient = np.sum(dy, axis=0)
        self.weight.gradient += dy.T @ x / x.shape[0]
        dx = dy @ self.weight.data #8, 84
        return dx

    def backward_gpu(self, dy: tensor) -> tensor:
        x = self.cache[0]
        return dy


class Bilinear(Module):
    __constants__ = ['in1_features', 'in2_features', 'out_features']
    in1_features : int
    in2_features : int
    out_features : int

    def __init__(self, in1_features: int, in2_features: int, out_features: int, bias: bool=True) -> None:
        super(Bilinear, self).__init__()
        self.in1_features = in1_features
        self.in2_features = in2_features
        self.out_features = out_features
        self.weight = Parameter([out_features, in1_features, in2_features], self._use_gpu, True)
        if bias:
            self.bias = Parameter([out_features], self._use_gpu, False)

    def forward_cpu(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        y = x1.T @ self.weight.data @ x2
        if self.bias is not None:
            y += self.bias.data
        self.cache = [x1, x2]
        return y

    def backward_cpu(self, dy):
        x1, x2 = self.cache
        self.bias.gradient = np.sum(dy, axis=0)
        dx1 = np.zeros(x1.shape)
        dx2 = np.zeros(x2.shape)
        for i in range(x1.shape[0]):
            self.weight.gradient[i] += x1[i].T @ dy[i] @ x2[i] / x1.shape[0]
            dx1[i] = dy[i] @ self.weight.data.T @ x2[i]
            dx2[i] = dy[i] @ self.weight.data.T @ x1[i]
       
        return dx1, dx2, 