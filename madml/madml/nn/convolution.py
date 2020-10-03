import numpy as np
from .module import Module
from madml.utils import single, double, triple

class ConvNd(Module):
    def __init__(self, filters, kernel_size, stride, padding, dilation, size=-1):
        super(ConvNd, self).__init__()
        self.filters = triple(filters, 'Filters', 3)
        self.kernel_size = triple(kernel_size, 'Kernel_Size', 3)
        self.stride = triple(stride, 'Stride', 3)
        self.padding = triple(padding, 'Padding', 3)
        self.dilation = triple(dilation, 'Dilation', 3)

class Conv3d(ConvNd):
    def __init__(self, filters, kernel_size, stride, padding, dilation):
        super(Conv3d, self).__init__(filters, kernel_size, stride, padding, dilation)

class ConvTranspose3d(ConvNd):
    def __init__(self, filters, kernel_size, stride, padding, dilation):
        super(ConvTranspose3d, self).__init__(filters, kernel_size, stride, padding, dilation)

class Conv2d(ConvNd):
    def __init__(self, filters, kernel_size, stride, padding, dilation):
        self.filters = double(filters, 'Filters', 3)
        self.kernel_size = double(kernel_size, 'Kernel_Size', 3)
        self.stride = double(stride, 'Stride', 3)
        self.padding = double(padding, 'Padding', 3)
        self.dilation = double(dilation, 'Dilation', 3)
        super(Conv2d, self).__init__(filters, kernel_size, stride, padding, dilation)

class ConvTranspose2d(ConvNd):
    def __init__(self, filters, kernel_size, stride, padding, dilation):
        self.filters = double(filters, 'Filters', 3)
        self.kernel_size = double(kernel_size, 'Kernel_Size', 3)
        self.stride = double(stride, 'Stride', 3)
        self.padding = double(padding, 'Padding', 3)
        self.dilation = double(dilation, 'Dilation', 3)
        super(ConvTranspose2d, self).__init__(filters, kernel_size, stride, padding, dilation)

class Conv1d(ConvNd):
    def __init__(self, filters, kernel_size, stride, padding, dilation):
        self.filters = single(filters, 'Filters', 3)
        self.kernel_size = single(kernel_size, 'Kernel_Size', 3)
        self.stride = single(stride, 'Stride', 3)
        self.padding = single(padding, 'Padding', 3)
        self.dilation = single(dilation, 'Dilation', 3)
        super(Conv1d, self).__init__(filters, kernel_size, stride, padding, dilation)

class ConvTranspose1d(ConvNd):
    def __init__(self, filters, kernel_size, stride, padding, dilation):
        self.filters = single(filters, 'Filters', 3)
        self.kernel_size = single(kernel_size, 'Kernel_Size', 3)
        self.stride = single(stride, 'Stride', 3)
        self.padding = single(padding, 'Padding', 3)
        self.dilation = single(dilation, 'Dilation', 3)
        super(ConvTranspose3d, self).__init__(filters, kernel_size, stride, padding, dilation)