from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import math

import numpy as np

import vknn
from madml import tensor, zeros
from madml.init import kaiming_uniform
from .module import Module
# from .testing import fc_forward, fc_backward

class linear(Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int

    def __init__(self, in_features: int, out_features: int, bias: bool = False) -> None:
        super(linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.w = self.register_weight(kaiming_uniform(a=math.sqrt(5), nonlinearity='linear'), [in_features, out_features])
        self.bias = self.register_bias(bias, zeros, [out_features])
        self.kernel_y = self.register_kernel(vknn.gemm, 1., 1., bias, False, False)
        self.kernel_dw = self.register_kernel(vknn.gemm, 1., 1., False, True, False)
        self.kernel_dx = self.register_kernel(vknn.gemm, 1., 1., False, False, True)

    def forward(self, x: tensor) -> tensor:
        self.register_output_shape([x.shape[0], self.out_features])
        self.register_forward_arg('x', x)
        self.register_forward_arg('w', self.w)

        self.register_backward_arg('x', x)
        self.register_backward_arg('w', self.w)
        self.register_backward_arg('y', self.y)
        super(linear, self).forward(x, self.w)
        return self.y

    def _forward_cpu(self, x: tensor, w: tensor) -> tensor:
        for i in range(x.shape[0]):
            self.y.host_data[i] = np.matmul(x.host_data[i], w.host_data)
        return self.y

    def _forward_gpu(self, x: tensor, w: tensor) -> tensor:
        self.kernel_y.forward(self.y.device_data, x.device_data, w.device_data, self.bias.device_data)
        self.kernel_y.run()
        return self.y

    def _backward_cpu(self, x: tensor, w: tensor, y: tensor) -> tensor:
        dx, dw, dy = x.gradient, w.gradient, y.gradient
        dw.host_data = np.matmul(x.host_data.T, dy.host_data)
        for i in range(x.shape[0]):
            x.gradient.host_data[i] = np.matmul(dy.host_data[i], self.w.host_data.T)
        return dx

    def _backward_gpu(self, x: tensor, w: tensor, y: tensor) -> tensor:
        dx, dw, dy = x.gradient, w.gradient, y.gradient
        self.kernel_dx.forward(dx.device_data, dy.device_data, w.device_data, self._empty_gpu_tensor_obj)
        self.kernel_dw.forward(dw.device_data, x.device_data, dy.device_data, self._empty_gpu_tensor_obj)

        self.kernel_dx.run()
        self.kernel_dw.run()
        return dx