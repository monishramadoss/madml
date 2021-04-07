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
        self.kernel_backward = vknn.relu(inplace, True)
        self.tmp = None

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
            if self.y is None:
                self.y = zeros_like(x)

            self.cache = [x, tmp]
            self.y.host_data = data
            return self.y

    def forward_gpu(self, x: tensor) -> tensor:
        if self.tmp is None:
            self.tmp = vknn.init_bool(np.zeros(x.shape).astype(bool))       
        if self.y is None and not self.inplace:
            self.y = zeros_like(x)
        elif self.inplace:
            self.y = x

        self.kernel.forward(self.y.device_data, x.device_data, self.tmp)
        self.cache = [x]
        return self.y
      

    def backward_cpu(self) -> tensor:
        x, tmp = self.cache
        y = self.y
        dx, dy = x.gradient, y.gradient
        arr = dy.host_data.reshape(x.shape) * tmp
        x.gradient.host_data = arr.reshape(x.shape)
        if not self.inplace:
            y.zero_grad()
        return x

    def backward_gpu(self) -> tensor:
        x = self.cache[0]
        dx, dy = x.gradient, self.y.gradient
        self.kernel_backward.forward(dx, dy, self.tmp)
        return x


    def print_l(self) -> None:
        x, t, y = self.cache
        super(relu, self).print_l()
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
    axis: int

    def __init__(self, axis: int=1):
        super(softmax, self).__init__()    
        self.axis = axis

    def forward_cpu(self, x: tensor) -> tensor:
        if self.y is None:
            self.y = zeros_like(x)
        x = x.host_data
        ex = np.exp((x.T - np.max(x, self.axis)).T)
        self.y.host_data = (ex.T / ex.sum(axis=self.axis)).T
        self.cache = [x]
        return self.y
    
    def backward_cpu(self) -> tensor:
        x = self.ca


        return x