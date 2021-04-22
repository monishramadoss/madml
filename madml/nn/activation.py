from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

from madml import tensor, zeros_like
from .module import Module
from .testing import relu_forward, relu_backward, dropout_forward, dropout_backward
import vknn

class relu(Module):
    __constants__ = ['inplace']
    inplace : bool

    def __init__(self, inplace: bool=False) -> None:
        super(relu, self).__init__()
        self.inplace = inplace
        self.out = None
        self.kernel = vknn.relu(inplace, False)
        self.kernel_dx = vknn.relu(inplace, True)
        self.tmp = None

    def extra_repr(self) -> str:
        inplace_str = 'inplace=True' if self.inplace else ''
        return inplace_str

    def setup_forward(self, x: tensor) -> None:
        if self.tmp is None:
            if self.device_id != -1:
                self.tmp = vknn.init_bool(np.zeros(x.shape).astype(bool))
            else:
                self.tmp = x.host_data > 0
        if self.y is None:
            if self.inplace:
                self.y = x
            else:
                self.y = zeros_like(x)

    def forward_cpu(self, x: tensor) -> tensor:
        self.setup_forward(x)
        data = x.host_data * self.tmp
        self.y.host_data = data
        self.register_backward_arg('x', x)
        self.register_backward_arg('y', self.y)
        return self.y

    def forward_gpu(self, x: tensor) -> tensor:
        self.setup_forward(x)
        self.kernel.forward(self.y.device_data, x.device_data, self.tmp)
        self.register_backward_arg('x', x)
        self.register_backward_arg('y', self.y)
        return self.y

    def backward_cpu(self, x: tensor, y: tensor) -> tensor:
        dx, dy = x.gradient, y.gradient
        arr = dy.host_data.reshape(dx.shape) * self.tmp
        dx.host_data = arr.reshape(dx.shape)
        return dx

    def backward_gpu(self, x: tensor, y: tensor) -> tensor:
        dx, dy = x.gradient, y.gradient
        self.kernel_dx.forward(dx.device_data, dy.device_data, self.tmp)
        return dx

    def test(self):
        x, tmp, y = self.cache
        _y, c = relu_forward(x.host_data)
        _dx = relu_backward(y.gradient.host_data, c)
        assert ((y.host_data == _y).all())
        assert ((_dx == x.gradient.host_data).all())

class dropout(Module):
    __constants__ = ['prob']
    prob : float

    def __init__(self, probability: float=0.1, seed: int=None) -> None:
        super(dropout, self).__init__()
        if seed:
            np.random.seed(seed)
        self.prob = probability
        self.mask = None

    def forward_cpu(self, x: tensor) -> tensor:
        if self.y is None:
            self.y = zeros_like(x)
        self.mask = tensor(np.random.rand(*x.shape), x.shape)
        self.mask.host_data = self.mask.host_data < self.prob
        tmp = x.host_data / (1 - self.prob)
        tmp[self.mask.host_data] = 0
        self.cache = [x]
        return self.y

    def backward_cpu(self) -> tensor:
        x = self.cache[0]
        y = self.y
        dx, dy = x.gradient, y.gradient
        dx = dy / (1 - self.prob)
        dx[self.mask.host_data] = 0
        x.gradient = dx
        return x

    def test(self):
        x, y = self.cache
        _y, c = dropout_forward(x.host_data, self.prob)
        _dx = dropout_backward(y.gradient.host_data, c)
        assert ((y.host_data == _y).all())
        assert ((_dx == x.gradient.host_data).all())

class softmax(Module):
    __constants__ = ['axis']
    axis : int

    def __init__(self, axis: int=1):
        super(softmax, self).__init__()
        self.axis = axis

    def forward_cpu(self, x: tensor) -> tensor:
        if self.y is None:
            self.y = zeros_like(x)
        x = x.host_data
        ex = np.exp((x.T - np.max(x, self.axis)).T)
        self.y.host_data = (ex.T / ex.sum(axis=self.axis)).T
        return self.y

    def backward_cpu(self) -> tensor:
        return self.y