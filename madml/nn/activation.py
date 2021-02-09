from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

from madml import tensor, zeros_like
from .module import Module
from .testing import relu_forward, relu_backward, dropout_forward, dropout_backward


class ReLU(Module):
    __constants__ = ['inplace']
    inplace: bool

    def __init__(self, inplace: bool = False) -> None:
        super(ReLU, self).__init__()
        self.inplace = inplace
        self.out = None

    def extra_repr(self) -> str:
        inplace_str = 'inplace=True' if self.inplace else ''
        return inplace_str

    def forward_cpu(self, x: tensor) -> tensor:
        tmp = x.host_data > 0
        data = x.host_data * tmp
        if self.inplace:
            self.cache = [x, tmp, x]
            x.host_data = data
            return x
        else:
            y = zeros_like(x)
            self.cache = [x, tmp, y]
            y.host_data = data
            return y

    def backward_cpu(self) -> tensor:
        x, tmp, y = self.cache
        dx, dy = x.gradient, y.gradient
        arr = dy.host_data.reshape(x.shape) * tmp
        x.gradient.host_data = arr.reshape(x.shape)
        if not self.inplace:
            pass  # y.zero_grad()
        return x

    def print_l(self) -> None:
        x, t, y = self.cache
        super(ReLU, self).print_l()
        print('\tmax input:', x.host_data.max(), 'g', x.gradient.host_data.max(),
              ' output:', y.host_data.max(), 'g', y.gradient.host_data.max())
        print('\tmin input:', x.host_data.min(), 'g', x.gradient.host_data.min(),
              ' output:', y.host_data.min(), 'g', y.gradient.host_data.min())
        self.test()

    def test(self):
        x, tmp, y = self.cache
        _y, c = relu_forward(x.host_data)
        _dx = relu_backward(y.gradient.host_data, c)
        assert ((y.host_data == _y).all())
        assert ((_dx == x.gradient.host_data).all())


class Dropout(Module):
    __constants__ = ['prob']
    prob: float

    def __init__(self, probability: float = 0.1, seed: int = None) -> None:
        super(Dropout, self).__init__()
        if seed:
            np.random.seed(seed)
        self.prob = probability
        self.mask = None

    def forward_cpu(self, x: tensor) -> tensor:
        y = zeros_like(x)
        self.mask = tensor(np.random.rand(*x.shape), x.shape)
        self.mask.host_data = self.mask.host_data < self.prob
        tmp = x.host_data / (1 - self.prob)
        tmp[self.mask.host_data] = 0
        self.cache = [x, y]
        return y

    def backward_cpu(self) -> tensor:
        x, y = self.cache
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


