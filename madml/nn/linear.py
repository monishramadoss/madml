from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import math

import numpy as np

from madml import tensor, zeros
from madml.init import kaiming_uniform
from .module import Module, Parameter
from .testing import fc_forward, fc_backward

import vknn

class Linear(Module):
    __constants__ = ['in_features', 'out_features']
    in_features : int
    out_features : int

    def __init__(self, in_features: int, out_features: int, bias: bool=False) -> None:
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(kaiming_uniform(a=math.sqrt(5), nonlinearity='linear'), [in_features, out_features])
        self.use_gpu = False
        if bias:
            self.bias = Parameter(zeros, [out_features])
        else:
            self.bias = None

        if self.use_gpu:
            self.gpu_forward = vknn.gemm(1., 1., bias)
            self.gpu_backward1 = vknn.gemm(1., 1., False)
            self.gpu_backward2 = vknn.gemm(1., 1., False)
        

    def forward_cpu(self, x: tensor) -> tensor:
        assert len(x.shape) == 2
        y = zeros([x.shape[0], self.out_features])
        for i in range(x.shape[0]):
            y.host_data[i] = np.matmul(x.host_data[i], self.weight.param.host_data)
            if self.bias is Parameter:
                y.host_data[i] += self.bias.param.host_data
        self.cache = [x, y]
        return y

    def backward_cpu(self) -> tensor:
        x, y = self.cache
        dx, dy = x.gradient, y.gradient
        if self.bias is Parameter:
            self.bias.param.gradient.host_data = np.sum(dy.host_data, axis=0)
        self.weight.param.gradient.host_data = np.matmul(x.host_data.T, dy.host_data)
        for i in range(x.shape[0]):
            x.gradient.host_data[i] = np.matmul(dy.host_data[i], self.weight.param.host_data.T)
        y.zero_grad()
        return x

    def forward_gpu(self, x: tensor) -> tensor:
        assert len(x.shape) == 2
        y = zeros([x.shape[0], self.out_features])
        if self.bias is Parameter:
            self.gpu_forward.forward(y.device_data, x.device_data, self.weight.param.device_data, self.bias.param.device_data)
        else:
            self.gpu_forward.forward(y.device_data, x.device_data, self.weight.param.device_data, self._empty_gpu_tensor_obj)


        self.cache = [x, y]
        return y

    def backward_gpu(self) -> tensor:
        x, y = self.cache
        dx, dy = x.gradient, y.gradient
        if self.bias is Parameter:
            self.bias.param.gradient.host_data = np.sum(dy.host_data, axis=0)
        self.weight.param.gradient.device_data = self.gpu_backward1.forward(self.weight.param.gradient.device_data, x.device_data, dy.device_data, self._empty_gpu_tensor_obj) # Trans X
        x.gradient.device_data = self.gpu_backward2.forward(x.gradient.device_data, dy.device_data, self.weight.param.device_data,  self._empty_gpu_tensor_obj) # Trans W
        y.zero_grad()
        return x

    def print_l(self) -> None:
        x, y = self.cache
        super(Linear, self).print_l()
        print('\tmax input:', x.host_data.max(), 'g', x.gradient.host_data.max(),
              ' weight:', self.weight.param.host_data.max(), 'g', self.weight.param.gradient.host_data.max(),
              ' output:', y.host_data.max(), 'g', y.gradient.host_data.max())
        print('\tmin input:', x.host_data.min(), 'g', x.gradient.host_data.min(),
              ' weight:', self.weight.param.host_data.min(), 'g', self.weight.param.gradient.host_data.min(),
              ' output:', y.host_data.max(), 'g', y.gradient.host_data.min())
        self.test()

    def test(self):
        x, y = self.cache
        _y, c = fc_forward(x.host_data, self.weight.param.host_data, self.bias.param.host_data)
        _dx, _dw, _db = fc_backward(y.gradient.host_data, c)
        assert ((y.host_data == _y).all())
        assert ((_dx == x.gradient.host_data).all())
        assert ((_dw == self.weight.param.gradient.host_data).all())
        assert ((_db == self.bias.param.gradient.host_data).all())