from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from typing import List, Optional, Union

import numpy as np
from .module import Module
from madml.utils import single, double, triple

class _ConvNd(Module):
    __constants__ = ['stride', 'padding', 'dilation', 'groups',
                     'padding_mode', 'output_padding', 'in_channels',
                     'out_channels', 'kernel_size']
    __annotations__ = {'bias': Optional[List]}

    _in_channels : int
    out_channels : int
    kernel_size : List[int]
    stride : List[int]
    padding : List[int]
    dilation : List[int]
    transposed : bool
    output_padding : List[int]
    groups : int
    padding_mode : str
    weight : List[float]
    bias : Optional[List[float]]

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Optional[int, List[int]],
                 stride: Optional[int, List[int]],
                 padding: Optional[int, List[int]],
                 dilation: Optional[int, List[int]],
                 transposed: bool,
                 output_padding: Optional[int, List[int]],
                 groups: int,
                 bias: Optional[List[float]],
                 padding_mode: str) -> None:
        super(_ConvNd, self).__init__()

        if groups != 1:
            raise NotImplementedError("dilation not implemented in conv")

        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        valid_padding_modes = {'zeros'} #, 'reflect', 'replicate', 'circular'} # needs to be implemented
        if padding_mode not in valid_padding_modes:
            raise ValueError("padding_mode must be one of {}, but got padding_mode='{}'".format(valid_padding_modes, padding_mode))
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        self.padding_mode = padding_mode
        #self._reversed_padding_repeated_twice = _reverse_repeat_tuple(self.padding, 2)
        if transposed:
            self.weight = np.zeros((in_channels, out_channels // groups, *kernel_size))
        else:
            self.weight = np.zeros((out_channels, in_channels // groups, *kernel_size))

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        if self.padding_mode != 'zeros':
            s += ', padding_mode={padding_mode}'
        return s.format(**self.__dict__)

    def __setstate__(self, state):
        #super(_ConvNd, self).__setstate__(state)
        if not hasattr(self, 'padding_mode'):
            self.padding_mode = 'zeros'

class Conv1d(_ConvNd):
    def __init__(self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, List[int]],
        stride: Union[int, List[int]]=1,
        padding: Union[int, List[int]]=0,
        dilation: Union[int, List[int]]=1,
        groups: Union[int, List[int]]=1,
        bias: bool=True,
        padding_mode: str='zeros'):

        kernel_size = single(kernel_size,)
        stride = single(stride)
        padding = single(padding)
        dilation = single(dilation)
        super(Conv1d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, False, single(0), groups, bias, padding_mode)

class Conv2d(_ConvNd):
    def __init__(self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, List[int]],
        stride: Union[int, List[int]]=1,
        padding: Union[int, List[int]]=0,
        dilation: Union[int, List[int]]=1,
        groups: Union[int, List[int]]=1,
        bias: bool=True,
        padding_mode: str='zeros'):

        kernel_size = double(kernel_size,)
        stride = double(stride)
        padding = double(padding)
        dilation = double(dilation)
        super(Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, False, double(0), groups, bias, padding_mode)

class Conv3d(_ConvNd):
    def __init__(self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, List[int]],
        stride: Union[int, List[int]]=1,
        padding: Union[int, List[int]]=0,
        dilation: Union[int, List[int]]=1,
        groups: Union[int, List[int]]=1,
        bias: bool=True,
        padding_mode: str='zeros'):

        kernel_size = triple(kernel_size,)
        stride = triple(stride)
        padding = triple(padding)
        dilation = triple(dilation)
        super(Conv3d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, False, triple(0), groups, bias, padding_mode)

class _ConvTransposeNd(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, transposed, output_padding,
                 groups, bias, padding_mode):
        if padding_mode != 'zeros':
            raise ValueError('Only "zeros" padding mode is supported for {}'.format(self.__class__.__name__))
        super(_ConvTransposeNd, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, transposed, output_padding, groups, bias, padding_mode)

    def _output_padding(self, input, output_size, stride, padding, kernel_size):
        if output_size is None:
            ret = single(self.output_padding)
        else:
            k = input.dim() - 2
            if len(output_size) == k + 2:
                output_size = output_size[2:]
            if len(output_size) != k:
                raise ValueError("output_size must have {} or {} elements (got {})"
                    .format(k, k + 2, len(output_size)))

            min_sizes = []
            max_sizes = []
            for d in range(k):
                dim_size = ((input.size(d + 2) - 1) * stride[d] - 2 * padding[d] + kernel_size[d])
                min_sizes.append(dim_size)
                max_sizes.append(min_sizes[d] + stride[d] - 1)

            for i in range(len(output_size)):
                size = output_size[i]
                min_size = min_sizes[i]
                max_size = max_sizes[i]
                if size < min_size or size > max_size:
                    raise ValueError(("requested an output size of {}, but valid sizes range from {} to {} (for an input of {})").format(output_size, min_sizes, max_sizes, input.size()[2:]))

            res = []
            for d in range(k):
                res.append(output_size[d] - min_sizes[d])
            ret = res
        return ret

class ConvTranspose1d(_ConvTransposeNd):
    def __init__(self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, List[int]],
        stride: Union[int, List[int]]=1,
        padding: Union[int, List[int]]=0,
        output_padding: Union[int, List[int]]=0,
        groups: int=1,
        bias: bool=True,
        dilation: Union[int, List[int]]=1,
        padding_mode: str='zeros'):

        kernel_size = single(kernel_size)
        stride = single(stride)
        padding = single(padding)
        dilation = single(dilation)
        output_padding = single(output_padding)
        super(ConvTranspose1d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, True, output_padding, groups, bias, padding_mode)

    def forward(self, input):
        #output_padding = self._output_padding(input, self.output_size, self.stride, self.padding, self.kernel_size)
        pass

class ConvTranspose2d(_ConvTransposeNd):
    def __init__(self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, List[int]],
        stride: Union[int, List[int]]=1,
        padding: Union[int, List[int]]=0,
        output_padding: Union[int, List[int]]=0,
        groups: int=1,
        bias: bool=True,
        dilation: Union[int, List[int]]=1,
        padding_mode: str='zeros'):

        kernel_size = double(kernel_size)
        stride = double(stride)
        padding = double(padding)
        dilation = double(dilation)
        output_padding = double(output_padding)
        super(ConvTranspose2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, True, output_padding, groups, bias, padding_mode)

    def forward(self, input):
        #output_padding = self._output_padding(input, output_size, self.stride, self.padding, self.kernel_size)
        pass

class ConvTranspose3d(_ConvTransposeNd):
    def __init__(self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, List[int]],
        stride: Union[int, List[int]]=1,
        padding: Union[int, List[int]]=0,
        output_padding: Union[int, List[int]]=0,
        groups: int=1,
        bias: bool=True,
        dilation: Union[int, List[int]]=1,
        padding_mode: str='zeros'):

        kernel_size = triple(kernel_size)
        stride = triple(stride)
        padding = triple(padding)
        dilation = triple(dilation)
        output_padding = triple(output_padding)
        super(ConvTranspose3d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, True, output_padding, groups, bias, padding_mode)

    def forward(self, input):
        #output_padding = self._output_padding(input, output_size, self.stride, self.padding, self.kernel_size)
        pass