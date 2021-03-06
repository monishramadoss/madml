from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

from madml import tensor, zeros_like
from .module import Module
from .testing import tanh_forward, tanh_backward

class tanh(Module):
    __constants__ = ['inplace']

    def __init__(self, inplace: bool=False) -> None:
        super(tanh, self).__init__()
        self.inplace = inplace
        self.out = None

    def extra_repr(self) -> str:
        inplace_str = 'inplace=True' if self.inplace else ''
        return inplace_str

    def forward_cpu(self, x: tensor) -> tensor:
        data = np.tanh(x.host_data)
        if self.inplace:
            self.cache = [x, x]
            x.host_data = data
            return x
        else:
            y = zeros_like(x)
            self.cache = [x, y]
            y.host_data = data
            return y

    def backward_cpu(self) -> tensor:
        x, y = self.cache
        dx, dy = x.gradient, y.gradient
        x.gradient.host_data = (1 - y.host_data ** 2) * dy.host_data
        return x

    def print_l(self) -> None:
        x, t, y = self.cache
        super(tanh, self).print_l()
        print('\tmax input:', x.host_data.max(), 'g', x.gradient.host_data.max(),
              ' output:', y.host_data.max(), 'g', y.gradient.host_data.max())
        print('\tmin input:', x.host_data.min(), 'g', x.gradient.host_data.min(),
              ' output:', y.host_data.min(), 'g', y.gradient.host_data.min())
        self.test()

    def test(self):
        x, tmp, y = self.cache
        _y, c = tanh_forward(x.host_data)
        _dx = tanh_backward(y.gradient.host_data, c)
        assert ((y.host_data == _y).all())
        assert ((_dx == x.gradient.host_data).all())