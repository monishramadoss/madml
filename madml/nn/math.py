from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from madml import tensor
from madml import zeros
from .module import Module

class sigmoid(Module):
    def __init__(self):
        super(sigmoid, self).__init__()

    def forward(self, x: tensor) -> tensor:
        self.y = self.register_output_shape(x.shape)      
        super(sigmoid, self).forward(x)
        return self.y

    def _forward_cpu(self, x: tensor) -> tensor:
        self.y.host_data = 1. / (1 + np.exp(-(x.host_data)))
        return self.y

    def _backward_cpu(self, x:tensor, y:tensor) -> tensor:
        dx, dy = x.gradient, y.gradient
        dx.host_data = y.host_data * (1. - y.host_data) * dy.host_data
        return dx

class add(Module):
    def __init__(self):
        super(add, self).__init__()

    def forward(self, x: tensor, w: tensor) -> tensor:
        self.y = self.register_output_shape(x.shape)
        super(add, self).forward(x, w)
        return self.y

    def _forward_cpu(self, x: tensor, w: tensor) -> tensor:
        self.y.host_data = x.host_data + w.host_data
        return self.y

    def _backward_cpu(self, x: tensor, w: tensor, y: tensor) -> tensor:
        dx, dw, dy = x.gradient, w.gradient, y.gradient
        dx.host_data = dy.host_data
        dw.host_data = dy.host_data
        return dx

class sub(Module):
    def __init__(self):
        super(sub, self).__init__()

    def forward(self, x: tensor, w: tensor) -> tensor:
        self.y = self.register_output_shape(x.shape)      
        super(add, self).forward(x, w)
        return self.y

    def _forward_cpu(self, x: tensor, w: tensor) -> tensor:
        self.y.host_data = x.host_data - w.host_data
        return self.y

    def _backward_cpu(self, x: tensor, w: tensor, y: tensor) -> tensor:
        dx, dw, dy = x.gradient, w.gradient, y.gradient
        dx.host_data = dy.host_data
        dw.host_data = -dy.host_data
        return dx

class mul(Module):
    def __init__(self):
        super(mul, self).__init__()

    def forward(self, x: tensor, w: tensor) -> tensor:
        self.y = self.register_output_shape(x.shape)     
        super(mul, self).forward(x, w)
        return self.y

    def _forward_cpu(self, x: tensor, w: tensor) -> tensor:
        self.y.host_data = x.host_data * w.host_data
        return self.y

    def _backward_cpu(self, x: tensor, w: tensor, y: tensor) -> tensor:
        dx, dw, dy = x.gradient, w.gradient, y.gradient
        dx.host_data = dy.host_data / w.host_data
        dw.host_data = dy.host_data / x.host_data
        return dx

class div(Module):
    def __init__(self):
        super(div, self).__init__()

    def forward(self, x: tensor, w: tensor) -> tensor:
        self.y = self.register_output_shape(x.shape)      
        super(div, self).forward(x, w)
        return self.y

    def _forward_cpu(self, x: tensor, w: tensor) -> tensor:
        self.y.host_data = x.host_data / w.host_data
        return self.y

    def _backward_cpu(self, x: tensor, w: tensor, y: tensor) -> tensor:
        dx, dw, dy = x.gradient, w.gradient, y.gradient
        dx.host_data = dy.host_data * w.host_data
        dw.host_data = x.host_data / dy.host_data
        return dx