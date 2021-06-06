from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

import vknn
from madml import tensor
from .module import Module
from .testing import relu_forward, relu_backward, dropout_forward, dropout_backward

class relu(Module):
    __constants__ = ['inplace']
    inplace: bool

    def __init__(self, inplace: bool = False) -> None:
        super(relu, self).__init__()
        self.inplace = inplace
        self.out = None
        self.kernel = vknn.relu(inplace, False)
        self.kernel_dx = vknn.relu(inplace, True)
        self.tmp = None

    def extra_repr(self) -> str:
        inplace_str = 'inplace=True' if self.inplace else ''
        return inplace_str

    def forward(self, x: tensor) -> tensor:
        if self.tmp is None:
            self.tmp = vknn.init_bool(np.zeros(x.shape).astype(bool)) if self.device_id != -1 else np.zeros(
                x.shape).astype(bool)
        if self.y is None:
            if self.inplace:
                self.y = x
            else:
                self.register_output_shape(x.shape)
        self.register_forward_arg('x', x)
        self.register_backward_arg('x', x)
        self.register_backward_arg('y', self.y)
        super(relu, self).forward(x)
        return self.y

    def _forward_cpu(self, x: tensor) -> tensor:
        self.tmp = x.host_data >= 0
        data = x.host_data * self.tmp
        self.y.host_data = data
        return self.y

    def _forward_gpu(self, x: tensor) -> tensor:
        self.kernel.forward(self.y.device_data, x.device_data, self.tmp)
        self.kernel.run()
        return self.y

    def _backward_cpu(self, x: tensor, y: tensor) -> tensor:
        dx, dy = x.gradient, y.gradient
        arr = dy.host_data.reshape(dx.shape) * self.tmp
        dx.host_data = arr.reshape(dx.shape)
        return dx

    def _backward_gpu(self, dx: tensor, dy: tensor) -> tensor:
        self.kernel_dx.forward(dx.device_data, dy.device_data, self.tmp)
        self.kernel_dx.run()
        return dx

    def test(self):
        x, tmp, y = self.cache
        _y, c = relu_forward(x.host_data)
        _dx = relu_backward(y.gradient.host_data, c)
        assert ((y.host_data == _y).all())
        assert ((_dx == x.gradient.host_data).all())

class dropout(Module):
    __constants__ = ['prob']
    prob: float

    def __init__(self, probability: float = 0.1, seed: int = None) -> None:
        super(dropout, self).__init__()
        if seed:
            np.random.seed(seed)
        self.prob = probability
        self.mask = None

    def forward(self, x: tensor) -> tensor:
        self.register_output_shape(x.shape)
        self.mask = tensor(np.random.rand(*x.shape), x.shape)
        self.register_forward_arg('x', x)
        self.register_backward_arg('dx', x.gradient)
        self.register_backward_arg('dy', self.y.gradient)
        super(dropout, self).forward(x)
        return self.y

    def _forward_cpu(self, x: tensor) -> tensor:
        self.mask.host_data = self.mask.host_data < self.prob
        tmp = x.host_data / (1 - self.prob)
        tmp[self.mask.host_data] = 0
        self.y.host_data = tmp
        self.cache = [x]
        return self.y

    def _backward_cpu(self, dx: tensor, dy: tensor) -> tensor:
        dx.host_data = dy.host_data / (1 - self.prob)
        dx[self.mask.host_data] = 0
        return dx

    def test(self):
        x, y = self.cache
        _y, c = dropout_forward(x.host_data, self.prob)
        _dx = dropout_backward(y.gradient.host_data, c)
        assert ((y.host_data == _y).all())
        assert ((_dx == x.gradient.host_data).all())

class softmax(Module):
    __constants__ = ['axis']
    axis: int

    def __init__(self, axis: int = 1):
        super(softmax, self).__init__()
        self.axis = axis

    def forward(self, x: tensor) -> tensor:
        self.register_forward_arg('x', x)
        self.register_output_shape(x.shape)
        super(softmax, self).forward(x)
        return self.y

    def _forward_cpu(self, x: tensor) -> tensor:
        x = x.host_data
        ex = np.exp((x.T - np.max(x, self.axis)).T)
        self.y.host_data = (ex.T / ex.sum(axis=self.axis)).T
        return self.y

    def _backward_cpu(self) -> tensor:
        return self.y