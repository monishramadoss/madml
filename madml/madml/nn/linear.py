from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

#from typing import Optional
from .module import Module
from madml import tensor
import madml

class Identity(Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward_cpu(self, input):
        return input

class Linear(Module):
    __constants__ = ['in_features', 'out_features']
    in_features : int
    out_features : int

    def __init__(self, in_features: int, out_features: int, bias: bool=True) -> None:
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = madml.zeros((out_features, in_features))
        if bias:
            self.bias = madml.zeros((out_features))

    def forward_cpu(self, x: tensor) -> tensor:
        y = madml.zeros((x.shape[0], self.out_features))
        for b in range(x.shape[0]):
            y[b] = madml.matmul(self.weight, x[b,...])
            if self.bias is not None:
                y[b] = y[b] + self.bias
        return y

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
        self.weight = madml.zeros((out_features, in1_features, in2_features))
        if bias:
            self.bias = madml.zeros((out_features))

    def forward_cpu(self, x: tensor) -> tensor:
        y = madml.matmul(x, self.weight)
        if self.bias:
            y += self.bias
        return y