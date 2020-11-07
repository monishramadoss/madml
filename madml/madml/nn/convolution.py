from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from typing import List, Optional, Union
from .module import Module
from ..utils import *
from madml import tensor
import madml
import backend
import numpy as np

def dim_fix(arr, arg_arr):
    j = 0
    for i in range(len(arg_arr) - 1, len(arr)):
        arr[i] = arg_arr[j]
        j+=1
    return arr

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
                 kernel_size: List[int],
                 stride: List[int],
                 padding: List[int],
                 dilation:  List[int],
                 transposed: bool,
                 output_padding: List[int],
                 groups: int,
                 bias: bool,
                 padding_mode: str) -> None:
        super(_ConvNd, self).__init__(None,)

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
        self.kernel_size = dim_fix([1,1,1], kernel_size)
        self.stride = dim_fix([1,1,1], stride)
        self.padding = dim_fix([0,0,0], padding)
        self.dilation = dim_fix([1,1,1], dilation)
        self.transposed = transposed
        self.output_padding = dim_fix([0,0,0], output_padding)
        self.groups = groups
        self.padding_mode = padding_mode
        self._col = [1,1,1]
        self._vol = [1,1,1]
        self.params = []
        #self._reversed_padding_repeated_twice =
        #_reverse_repeat_tuple(self.padding, 2)
        if self._use_gpu:
            if transposed:
                self.weight = madml.ones((in_channels, out_channels // groups, *self.kernel_size))
            else:
                self.weight = madml.ones((out_channels, in_channels // groups, *self.kernel_size))
            self._use_bias = bias
            self._vol2col = backend.vol2col(in_channels, [*self.kernel_size, *self.padding, *self.stride,  *self.dilation])
            self._matmul = backend.gemm(1.0, 1.0, bias)
            self._T = backend.transpose([1,0,2,3,4])
            self._n_output_plane = out_channels * self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2]
            self._temp = None
        else:
            if transposed:
                self.weight = np.ones((in_channels, out_channels // groups, *self.kernel_size))
            else:
                self.weight = np.ones((out_channels, in_channels // groups, *self.kernel_size))
        self.bias = None
        self._parameters['weights'] = [self.weight, self.bias]

    def forward_gpu(self, x: tensor) -> tensor:
        if(len(x.shape) >= 3):
            self._col[2] = int((x.shape[-1] + 2 * self.padding[2] - self.dilation[2] * (self.kernel_size[2] - 1) - 1) // self.stride[2]) + 1
            self._vol[2] = x.shape[-1]
        if (len(x.shape) >= 4):
            self._col[1] = int((x.shape[-2] + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) // self.stride[1]) + 1
            self._vol[1] = x.shape[-2]
        if(len(x.shape) >= 5):
            self._col[0] = int((x.shape[-3] + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) // self.stride[0]) + 1
            self._vol[0] = x.shape[-3]

        self.batch_size = x.shape[0]
        if self._temp is None:
            self._temp = madml.zeros([1])

        self._vol2col(self._temp, x)
        self.weight.reshape(-1, self._n_output_plane) 
        if self.bias is None:
            self.bias = madml.zeros([self.weight.shape[0], self._temp.shape[1]])

        y = self._matmul(self.weight, self._temp, self.bias)
        y = y.reshape((self.out_channels, self.batch_size, *self._col))
        y = self._T(y)
        self.cache = [x]
        return y

    def forward_cpu(self, x: np.ndarray) -> np.ndarray:
        if(len(x.shape) >= 3):
            self._col[2] = int((x.shape[-1] + 2 * self.padding[2] - self.dilation[2] * (self.kernel_size[2] - 1) - 1) // self.stride[2]) + 1
            self._vol[2] = x.shape[-1]
        if (len(x.shape) >= 4):
            self._col[1] = int((x.shape[-2] + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) // self.stride[1]) + 1
            self._vol[1] = x.shape[-2]
        if(len(x.shape) >= 5):
            self._col[0] = int((x.shape[-3] + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) // self.stride[0]) + 1
            self._vol[0] = x.shape[-3]

        self.batch_size = x.shape[0]
        B, n_output_plane, output_length = im2col_cpu(x, self.batch_size, self.in_channels, self._vol, self._col, self.kernel_size, self.stride, self.padding, self.dilation)
        y = np.matmul(self.weight.reshape(-1, n_output_plane), B)
        y = y.reshape((self.out_channels, self.batch_size, *self._col))
        y = np.transpose(y, (1,0,2,3,4))

        if self.bias is not None:
            self.bias = np.zeros(y.shape[1:])
            for b in range(self.batch_size):
                y[b] += self.bias

        self.cache = [x, B]
        return y

    def backward_cpu(self, dy: np.ndarray) -> np.ndarray:
        x, B = self.cache
        if self.backend:
            self.db = np.sum(dy, axis=(0, 2, 3, 4))
        else:
            self.db = None

        n_filter, c_filter, d_filter, h_filter, w_filter = self.weight.shape
        dy_reshaped = dy.transpose([1, 2, 3, 4, 0]).reshape(n_filter, -1)
        self.dw = dy_reshaped @ B.T
        self.dw = self.dw.reshape(self.weight.shape)
        w_reshape = self.weight.reshape(n_filter, -1)
        self._parameters['grads'] = [self.dw, self.db]
        dx_col = w_reshape.T @ dy_reshaped
        dx = col2im_cpu(dx_col, self.batch_size, n_filter, self._vol, self._col, self.kernel_size, self.stride, self.padding, self.dilation)
        return dx

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

class Conv1d(_ConvNd):
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
        bias: bool=False,
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
        bias: bool=False,
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
            ret = self.output_padding
        else:
            k = input.dim() - 2
            if len(output_size) == k + 2:
                output_size = output_size[2:]
            if len(output_size) != k:
                raise ValueError("output_size must have {} or {} elements (got {})".format(k, k + 2, len(output_size)))

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

    def forward_cpu(self, x: tensor) -> tensor:
        if(len(x.shape) >= 3):
            self._col[2] = int(x.shape[-1] + 2 * self.padding[2] - (self.dilation[2] * (self.kernel_size[2] - 1) + 1) // self.stride[2] + 1)
            self._vol[2] = x.shape[-1]
        if (len(x.shape) >= 4):
            self._col[1] = int(x.shape[-2] + 2 * self.padding[1] - (self.dilation[1] * (self.kernel_size[1] - 1) + 1) // self.stride[1] + 1)
            self._vol[1] = x.shape[-2]
        if(len(x.shape) == 5):
            self._col[0] = int(x.shape[-3] + 2 * self.padding[0] - (self.dilation[0] * (self.kernel_size[0] - 1) + 1) // self.stride[0] + 1)
            self._vol[0] = x.shape[-3]

        #output_padding = self._output_padding(x, self._col, self.stride,
        #self.padding, self.kernel_size)
class ConvTranspose1d(_ConvTransposeNd):
    def __init__(self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, List[int]],
        stride: Union[int, List[int]]=1,
        padding: Union[int, List[int]]=0,
        output_padding: Union[int, List[int]]=0,
        groups: int=1,
        bias: bool=False,
        dilation: Union[int, List[int]]=1,
        padding_mode: str='zeros'):

        kernel_size = single(kernel_size)
        stride = single(stride)
        padding = single(padding)
        dilation = single(dilation)
        output_padding = single(output_padding)
        super(ConvTranspose1d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, True, output_padding, groups, bias, padding_mode)

class ConvTranspose2d(_ConvTransposeNd):
    def __init__(self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, List[int]],
        stride: Union[int, List[int]]=1,
        padding: Union[int, List[int]]=0,
        output_padding: Union[int, List[int]]=0,
        groups: int=1,
        bias: bool=False,
        dilation: Union[int, List[int]]=1,
        padding_mode: str='zeros'):

        kernel_size = double(kernel_size)
        stride = double(stride)
        padding = double(padding)
        dilation = double(dilation)
        output_padding = double(output_padding)
        super(ConvTranspose2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, True, output_padding, groups, bias, padding_mode)

class ConvTranspose3d(_ConvTransposeNd):
    def __init__(self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, List[int]],
        stride: Union[int, List[int]]=1,
        padding: Union[int, List[int]]=0,
        output_padding: Union[int, List[int]]=0,
        groups: int=1,
        bias: bool=False,
        dilation: Union[int, List[int]]=1,
        padding_mode: str='zeros'):

        kernel_size = triple(kernel_size)
        stride = triple(stride)
        padding = triple(padding)
        dilation = triple(dilation)
        output_padding = triple(output_padding)
        super(ConvTranspose3d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, True, output_padding, groups, bias, padding_mode)