from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import math
from typing import Union, List, Optional

import numpy as np

from madml import tensor
from madml.init import kaiming_uniform, zeros, ones, xavier_uniform
from .module import Module, Parameter
from .testing import conv_forward, conv_backward
from .transform import vol2col, transpose
import vknn

MAX_DIMS = 3


def _dim_fix(arr, arg_arr, pi):
    def parse(x):
        return [x for _ in range(pi)] if isinstance(x, int) else [x[t] for t in range(pi)]

    if isinstance(arg_arr, int):
        arg_arr = parse(arg_arr)
    j = 0
    for i in range(len(arg_arr) - 1, len(arr)):
        arr[i] = arg_arr[j]
        j += 1
    return arr

class ConvNd(Module):
    __constants__ = ['dims', 'stride', 'padding', 'dilation', 'groups', 'padding_mode', 'output_padding', 'in_channels',
                     'out_channels', 'kernel_size']
    __annotations__ = {'bias': Optional[List]}
    dims : int
    in_channels : int
    out_channels : int
    kernel_size : List[int]
    stride : List[int]
    padding : List[int]
    dilation : List[int]
    transposed : bool
    output_padding : List[int]
    groups : int
    padding_mode : str
    weight : Parameter
    bias : Optional[Parameter]

    def __init__(self, dims, in_channels: int, out_channels: int, kernel_size: Union[int, List[int]],
                 stride: Union[int, List[int]], padding: Union[int, List[int]], dilation: Union[int, List[int]],
                 transposed: bool, output_padding: Union[int, List[int]],
                 groups: int, bias: bool, padding_mode: str, weight_init='kaiming_uniform') -> None:
        super(ConvNd, self).__init__()

        if groups != 1:
            raise NotImplementedError("groups not implemented in conv")

        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        valid_padding_modes = {'zeros'}  # , 'reflect', 'replicate', 'circular'} # needs to be implemented
        if padding_mode not in valid_padding_modes:
            raise ValueError("padding_mode must be one of {}, but got padding_mode='{}'".format(valid_padding_modes, padding_mode))
        #self.use_gpu = True
        self.dims = dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _dim_fix([1 for _ in range(MAX_DIMS)], kernel_size, dims)
        self.stride = _dim_fix([1 for _ in range(MAX_DIMS)], stride, dims)
        self.padding = _dim_fix([0 for _ in range(MAX_DIMS)], padding, dims)
        self.dilation = _dim_fix([1 for _ in range(MAX_DIMS)], dilation, dims)
        self.transposed = transposed
        self.output_padding = _dim_fix([0 for _ in range(MAX_DIMS)], output_padding, dims)
        self.groups = groups
        self.padding_mode = padding_mode
        self._col = []
        self._vol = []
        self.use_bias = bias
        self.batch_size = 1
        self.col = None
        if transposed:
            weight_shape = [in_channels, out_channels // groups, *self.kernel_size]
        else:
            weight_shape = [out_channels, in_channels // groups, *self.kernel_size]

        if weight_init == 'xavier_uniform':
            self.weight = self.register_weight(xavier_uniform(), weight_shape)
        elif weight_init == 'kaiming_uniform':
            self.weight = self.register_weight(kaiming_uniform(a=math.sqrt(5), nonlinearity='conv3d'), weight_shape)
        else:
            self.weight = self.register_weight(ones, weight_shape)

        self.vol_col = None
        self._TP = [1, 0] + [i + 2 for i in range(dims)]
        self.gemm_y = self.register_kernel(vknn.gemm, 1.0,1.0, bias, False, False)
        self.gemm_dw = self.register_kernel(vknn.gemm, 1.0, 1.0, False, False, True)
        self.gemm_dc = self.register_kernel(vknn.gemm, 1.0, 1.0, False, True, False)
        self.transpose_y = self.register_module(transpose, self._TP, True)
        self.output_shape = []

    def forward(self, x: tensor, w: tensor) -> None:
        self.batch_size = x.shape[0]
        self._col = [1 for _ in range(MAX_DIMS)]
        self._vol = [1 for _ in range(MAX_DIMS)]
        for i in range(1, self.dims+1):
            self._col[-i] = int((x.shape[-i] + 2 * self.padding[-i] - self.dilation[-i] * (self.kernel_size[-i] - 1) - 1) // self.stride[-i]) + 1
            self._vol[-i] = x.shape[-i]
        self.output_shape = [self._col[i] for i in range(-1, -(self.dims+1), -1)]
        self.bias = self.register_bias(self.use_bias, zeros, [self.out_channels, *self.output_shape])
        self.register_output_shape([self.batch_size, self.out_channels, *self.output_shape])

        if self.vol_col is None:
            self.vol_col = self.register_module(vol2col, self.batch_size, self.in_channels, self._vol, self._col, self.kernel_size, self.stride, self.padding, self.dilation)

        self.col = self.vol_col(x)
        self.register_forward_arg('x', x)
        self.register_forward_arg('w', w)

        self.register_backward_arg('x', x)
        self.register_backward_arg('w', w)
        self.register_backward_arg('y', self.y)
        self.register_backward_arg('col', self.col)       
        
        super(ConvNd, self).forward(x, w)
        return self.y            


    def forward_cpu(self, x: tensor, w: tensor) -> tensor:
        w.reshape([w.shape[0], -1])
        self.y.reshape([self.weight.shape[0], -1])       
        self.y.host_data = np.matmul(w.host_data, self.col.host_data)
        self.y.reshape([self.out_channels, self.batch_size, *self.output_shape])
        self.transpose_y.forward(self.y)
        return self.y

    def forward_gpu(self, x: tensor, w: tensor) -> tensor:
        w.reshape([w.shape[0], -1])
        self.y.reshape([self.weight.shape[0], -1])
        self.gemm_y.forward(self.y.device_data, w.device_data, self.col.device_data, self.bias.device_data)        
        self.y.reshape([self.out_channels, self.batch_size, *self.output_shape])
        self.gemm_y.run()
        self.transpose_y.forward(self.y)
        return self.y

    def backward_cpu(self, x: tensor, w: tensor, y: tensor, col: tensor) -> tensor:
        y.reset_shape()
        dx, dw, dy, dcol = x.gradient, w.gradient, y.gradient, col.gradient
        _dy = self.transpose_y.backward()
        dy.reshape([self.out_channels, -1])

        dw.host_data = np.matmul(dy.host_data, col.host_data.T)
        w.reset_shape()

        w_reshaped = w.host_data.reshape([self.out_channels, -1])
        dcol.host_data = np.matmul(w_reshaped.T, dy_reshaped)

        _dx = self.vol_col.backward()
        return dx

    def backward_gpu(self, x: tensor, w: tensor, y: tensor, col: tensor) -> tensor:
        y.reset_shape()
        dx, dw, dy, dcol = x.gradient, w.gradient, y.gradient, col.gradient
        _dy = self.transpose_y.backward()
        dy.reshape([self.out_channels, -1])

        self.gemm_dx.forward(dcol.device_data, w.device_data, dy.device_data, self._empty_gpu_tensor_obj)
        self.gemm_dw.forward(dw.device_data, dy.device_data, dcol.device_data, self._empty_gpu_tensor_obj)
        
        self.gemm_dx.run()
        self.gemm_dw.run()

        _dx = self.vol_col.backward()
        return dx
       

class conv1d(ConvNd):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, List[int]],
                 stride: Union[int, List[int]]=1,
                 padding: Union[int, List[int]]=0,
                 dilation: Union[int, List[int]]=1,
                 groups: Union[int, List[int]]=1,
                 bias: bool=False,
                 padding_mode: str='zeros',
                 weight_init: str='kaiming_uniform') -> None:
        super(conv1d, self).__init__(1, in_channels, out_channels, kernel_size, stride, padding, dilation, False, 0,
                                     groups, bias, padding_mode, weight_init)


class conv2d(ConvNd):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, List[int]],
                 stride: Union[int, List[int]]=1,
                 padding: Union[int, List[int]]=0,
                 dilation: Union[int, List[int]]=1,
                 groups: Union[int, List[int]]=1,
                 bias: bool=False,
                 padding_mode: str='zeros',
                 weight_init: str='xavier_uniform') -> None:
        super(conv2d, self).__init__(2, in_channels, out_channels, kernel_size, stride, padding, dilation, False, 0,
                                     groups, bias, padding_mode, weight_init)

   
class conv3d(ConvNd):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, List[int]],
                 stride: Union[int, List[int]]=1,
                 padding: Union[int, List[int]]=0,
                 dilation: Union[int, List[int]]=1,
                 groups: Union[int, List[int]]=1,
                 bias: bool=False,
                 padding_mode: str='zeros'):
        super(conv3d, self).__init__(3, in_channels, out_channels, kernel_size, stride, padding, dilation, False, 0,
                                     groups, bias, padding_mode)


class ConvTransposeNd(ConvNd):
    def __init__(self, dims, in_channels, out_channels, kernel_size, stride,
                  padding, dilation, output_padding,
                  groups, bias, padding_mode, weight_init):
        if padding_mode != 'zeros':
            raise ValueError('Only "zeros" padding mode is supported for {}'.format(self.__class__.__name__))

        super(ConvTransposeNd, self).__init__( dims, in_channels, out_channels, kernel_size, stride,  padding, dilation, True, output_padding,
            groups, bias, padding_mode, weight_init)
    

class convtranspose1d(ConvTransposeNd):
    def __init__(self, 
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, List[int]],
                 stride: Union[int, List[int]]=1,
                 padding: Union[int, List[int]]=0,
                 dilation: Union[int, List[int]]=1,
                 groups: Union[int, List[int]]=1,
                 bias: bool=False,
                 padding_mode: str='zeros',
                 weight_init: str='xavier_uniform') -> None:
        super(convtranspose1d, self).__init__(1, in_channels, out_channels, kernel_size, stride, padding, dilation, output_padding,
            groups, bias, padding_mode, weight_init)


class convtranspose2d(ConvTransposeNd):
    def __init__(self, 
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, List[int]],
                 stride: Union[int, List[int]]=1,
                 padding: Union[int, List[int]]=0,
                 dilation: Union[int, List[int]]=1,
                 groups: Union[int, List[int]]=1,
                 bias: bool=False,
                 padding_mode: str='zeros',
                 weight_init: str='xavier_uniform') -> None:
        super(convtranspose2d, self).__init__(2, in_channels, out_channels, kernel_size, stride, padding, dilation, output_padding,
            groups, bias, padding_mode, weight_init)


   
class convtranspose3d(ConvTransposeNd):
    def __init__(self, 
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, List[int]],
                 stride: Union[int, List[int]]=1,
                 padding: Union[int, List[int]]=0,
                 dilation: Union[int, List[int]]=1,
                 groups: Union[int, List[int]]=1,
                 bias: bool=False,
                 padding_mode: str='zeros',
                 weight_init: str='xavier_uniform') -> None:
        super(convtranspose3d, self).__init__(3, in_channels, out_channels, kernel_size, stride, padding, dilation, output_padding,
            groups, bias, padding_mode, weight_init)
