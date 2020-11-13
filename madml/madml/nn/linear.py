from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

#from typing import Optional
from .module import Module
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
        if self._use_gpu:
            self.weight = madml.zeros((out_features, in_features))
            if bias:
                self.bias = madml.zeros((out_features))
        else:
            self.weight = np.zeros((out_features, in_features))
            if bias:
                self.bias = np.zeros((out_features))
        self._parameters['weights'] = [self.weight, self.bias]

    def forward_cpu(self, x: np.ndarray) -> np.ndarray:
        y = np.zeros((x.shape[0], self.out_features))
        for i in range(x.shape[0]):
            y[i] = self.weight @ x[i]
            if self.bias is not None:
                y[i] = y[i] + self.bias                
        return y

    def forward_gpu(self, x: tensor) -> tensor:
        y = madml.zeros((x.shape[0], self.out_features))
        self.backend(y, x, self.weight, self.bias)
        self.cache = [x]
        return y

    def backward_cpu(self, dy: np.ndarray) -> np.ndarray:
        x = self.cache[0]
        db = np.sum(dy, axis=0)
        dw = np.zeros(self.weight.shape)
        dx = np.zeros(x.shape)
        for i in range(x.shape[0]):
            dw[i] += x[i].T @ dy[i] / x.shape[0]
            dx[i] = dy[i] @ self.weight.T
        self._parameters['grads'] = [dw, db]
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
        if self._use_gpu:
            self.weight = madml.zeros((out_features, in1_features, in2_features))
            if bias:
                self.bias = madml.zeros((out_features))
        else:
            self.weight = np.zeros((out_features, in1_features, in2_features))
            if bias:
                self.bias = np.zeros((out_features))

    def forward_cpu(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        y = x1.T @ self.weight @ x2
        if self.bias is not None:
            y += self.bias
        self.cache = [x1, x2]
        return y

    def backward_cpu(self, dy):
        x1, x2 = self.cache
        db = np.sum(dy, axis=0)
        dw = np.zeros(self.weight.shape)
        dx1 = np.zeros(x1.shape)
        dx2 = np.zeros(x2.shape)
        for i in range(x1.shape[0]):
            dw[i] += x1[i].T @ dy[i] @ x2[i].T / x1.shape[0]
            dx1[i] = dy[i] @ self.weight.T @ x2[i]
            dx2[i] = dy[i] @ self.weight.T @ x1[i]
        self._parameters['grads'] = [dw, db]
        return dx1, dx2, 