from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from typing import List, Optional, Union
import madml
from .module import Module
from ..utils import *
import numpy as np

def dim_fix(arr, arg_arr):
    j = 0
    for i in range(len(arg_arr) - 1, len(arr)):
        arr[i] = arg_arr[j]
        j+=1
    return arr

class _MaxPoolNd(Module):
    __constants__ = ['kernel_size', 'stride', 'padding', 'dilation', 'return_indices', 'ceil_mode']

    return_indices : bool
    ceil_mode : bool

    def __init__(self, kernel_size, stride=None,
                 padding=0, dilation=1,
                 return_indices=False, ceil_mode=False) -> None:
        super(_MaxPoolNd, self).__init__()
        self.kernel_size = dim_fix([1,1,1], kernel_size)
        self.stride = dim_fix([1,1,1], stride)
        self.padding = dim_fix([0,0,0], padding)
        self.dilation = dim_fix([1,1,1], dilation)
        self.return_indices = return_indices
        self.ceil_mode = ceil_mode
        self._col = []
        self._vol = []
        if self._use_gpu:
            self._kernel = backend.vol2col(1, [*self.kernel_size, *self.stride, *self.padding, *self.dilation])
            self._trans = backend.transpose([1, 0, 2, 3, 4])
    def extra_repr(self) -> str:
        return 'kernel_size={kernel_size}, stride={stride}, padding={padding}, dilation={dilation}, ceil_mode={ceil_mode}'.format(**self.__dict__)

    def forward_cpu(self, x: np.ndarray) -> np.ndarray:
        if len(self._col) != 3 or len(self._vol) != 3:
            self._col = [1,1,1]
            self._vol = [1,1,1]
            if(len(x.shape) >= 3):
                self._col[2] = int((x.shape[-1] + 2 * self.padding[2] - self.dilation[2] * (self.kernel_size[2] - 1) - 1) // self.stride[2]) + 1
                self._vol[2] = x.shape[-1]
            if (len(x.shape) >= 4):
                self._col[1] = int((x.shape[-2] + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) // self.stride[1]) + 1
                self._vol[1] = x.shape[-2]
            if(len(x.shape) == 5):
                self._col[0] = int((x.shape[-3] + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) // self.stride[0]) + 1
                self._vol[0] = x.shape[-3]

        self.batch_size = x.shape[0] * x.shape[1]
        self.in_channels = 1
        B, n_output_plane, output_length = im2col_cpu(x, self.batch_size, self.in_channels, self._vol, self._col, self.kernel_size, self.stride, self.padding, self.dilation)
        # 4 X 32x1x25
        max_idx = np.argmax(B, axis=0)
        y = B[max_idx, range(max_idx.size)]
        y = y.reshape(x.shape[1], x.shape[0], *self._col)
        y = np.transpose(y, (1, 0, 2, 3, 4))
        self.cache = [x, max_idx, B]
        return y

    def backward_cpu(self, dy: np.ndarray) -> np.ndarray:
        x, max_idx, B = self.cache
        dx_col = np.zeros(B.shape)
        dy_col = np.transpose(dy, (2,3,4,0,1)).ravel() # (72128,)
        dx_col[max_idx, range(dy_col.size)] = dy_col
        
        dx = col2im_cpu(dx_col, self.batch_size, self.in_channels, self._vol, self._col, self.kernel_size, self.stride, self.padding, self.dilation)
        dx = dx.reshape(x.shape)
        return dx

class MaxPool1d(_MaxPoolNd):
    kernel_size : int
    stride : int
    padding : int
    dilation : int

    def __init__(self, kernel_size: int, stride: Optional[int]=None,
                 padding: int=0, dilation: int=1,
                 return_indices: bool=False, ceil_mode: bool=False) -> None:
        kernel_size = single(kernel_size)
        stride = single(stride or kernel_size)
        padding = single(padding)
        dilation = single(dilation)
        super(MaxPool1d, self).__init__(kernel_size, stride, padding, dilation, return_indices, ceil_mode)

class MaxPool2d(_MaxPoolNd):
    kernel_size : Union[int, List[int]]
    stride : Union[int, List[int]]
    padding : Union[int, List[int]]
    dilation : Union[int, List[int]]
    def __init__(self, kernel_size: Union[int, List[int]], stride: Optional[Union[int, List[int]]]=None,
                 padding: Union[int, List[int]]=0, dilation: Union[int, List[int]]=1,
                 return_indices: bool=False, ceil_mode: bool=False) -> None:
        kernel_size = double(kernel_size)
        stride = double(stride or kernel_size)
        padding = double(padding)
        dilation = double(dilation)
        super(MaxPool2d, self).__init__(kernel_size, stride, padding, dilation, return_indices, ceil_mode)

class MaxPool3d(_MaxPoolNd):
    kernel_size : Union[int, List[int]]
    stride : Union[int, List[int]]
    padding : Union[int, List[int]]
    dilation : Union[int, List[int]]
    def __init__(self, kernel_size: Union[int, List[int]], stride: Optional[Union[int, List[int]]]=None,
                 padding: Union[int, List[int]]=0, dilation: Union[int, List[int]]=1,
                 return_indices: bool=False, ceil_mode: bool=False) -> None:
        kernel_size = triple(kernel_size)
        stride = triple(stride or kernel_size)
        padding = triple(padding)
        dilation = triple(dilation)
        super(MaxPool3d, self).__init__(kernel_size, stride, padding, dilation, return_indices, ceil_mode)

class _MaxUnpoolNd(Module):
    __constants__ = ['kernel_size', 'stride', 'padding']

    kernel_size : Union[int, List[int]]
    stride : Optional[Union[int, List[int]]]
    padding : Union[int, List[int]]

    def __init__(self, kernel_size, stride, padding):
        super(_MaxUnpoolNd, self).__init__()
        self.kernel_size = kernel_size
        self.stride = dim_fix([1,1,1], stride)
        self.padding = dim_fix([0,0,0], padding)
        self.dilation = [1,1,1]
        self._vol = []
        self._col = []

        if self._use_gpu:
            self._vol2col = backend.vol2col(1, [*self.kernel_size, *self.stride, *self.padding, *self.dilation])
            self._T = backend.transpose([1, 0, 2, 3, 4])

    def extra_repr(self) -> str:
        return 'kernel_size={}, stride={}, padding={}'.format(self.kernel_size, self.stride, self.padding)

    def forward_cpu(self, x: np.ndarray) -> np.ndarray:
        t = np.transpose(x, (1, 0, 2, 3, 4))

class MaxUnpool1d(_MaxUnpoolNd):
    def __init__(self, kernel_size: int, stride: Optional[int]=None, padding: int=0) -> None:
        kernel_size = single(kernel_size)
        padding = single(padding)
        if stride:
            stride = single(stride or kernel_size)
        super(MaxUnpool1d, self).__init__(kernel_size, stride, padding)

class MaxUnpool2d(_MaxUnpoolNd):
    def __init__(self, kernel_size: Union[int, List[int]], stride: Optional[Union[int, List[int]]]=None, padding: Union[int, List[int]]=0) -> None:
        kernel_size = double(kernel_size)
        padding = double(padding)
        if stride:
            stride = double(stride or kernel_size)
        super(MaxUnpool2d, self).__init__(kernel_size, stride, padding)

class MaxUnpool3d(_MaxUnpoolNd):
    def __init__(self, kernel_size: Union[int, List[int]], stride: Optional[Union[int, List[int]]]=None, padding: Union[int, List[int]]=0) -> None:
        kernel_size = triple(kernel_size)
        padding = triple(padding)
        if stride:
            stride = triple(stride or kernel_size)
        super(MaxUnpool3d, self).__init__(kernel_size, stride, padding)

class _AvgPoolNd(Module):
    __constants__ = ['kernel_size', 'stride', 'padding', 'ceil_mode', 'count_include_pad', 'divisor_override']
    kernel_size : Union[int, List[int]]
    stride : Union[int, List[int]]
    padding : Union[int, List[int]]
    ceil_mode : bool
    count_include_pad : bool
    divisor_override : bool

    def __init__(self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override) -> None:
        super(_AvgPoolNd, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad
        self.divisor_override = divisor_override
        self.dilation = [1,1,1]
        self._col = []
        self._vol = []
        if self._use_gpu:
            self._vol2col = backend.vol2col(1, [*self.kernel_size, *self.stride, *self.padding, *self.dilation])
            #self._T = backend.transpose([1, 0, 2, 3, 4])
    def extra_repr(self) -> str:
        return 'kernel_size={}, stride={}, padding={}'.format(self.kernel_size, self.stride, self.padding)

    def forward_cpu(self, x: np.ndarray) -> np.ndarray:
        if len(self._col) != 3 or len(self._vol) != 3:
            self._col = [1,1,1]
            self._vol = [1,1,1]
            if(len(x.shape) >= 3):
                self._col[2] = int((x.shape[-1] + 2 * self.padding[2] - self.dilation[2] * (self.kernel_size[2] - 1) + 1) // self.stride[2] + 1)
                self._vol[2] = x.shape[-1]
            if (len(x.shape) >= 4):
                self._col[1] = int((x.shape[-2] + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) + 1) // self.stride[1] + 1)
                self._vol[1] = x.shape[-2]
            if(len(x.shape) == 5):
                self._col[0] = int((x.shape[-3] + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) + 1) // self.stride[0] + 1)
                self._vol[0] = x.shape[-3]

        self.batch_size = x.shape[0] * x.shape[1]       
        B, n_output_plane, output_length = im2col_cpu(x, self.batch_size, 1, self._col, self._vol, self.kernel_size, self.stride, self.padding, self.dilation)

        mean = np.mean(B, axis=0)
        y = mean.reshape(x.shape[1], self.batch_size, *self._col)
        y = np.transpose(y, (1, 0, 2, 3, 4))
        self.cache = [x, mean, B]
        return y

    def backward_cpu(self, dy:  np.ndarray) -> np.ndarray:
        x, max_idx, B = self.cache
        n, c, d, h, w = x.shape

        dx_col = np.zero(B.shape)
        dy_col = np.transpose(dy, (2,3,4,0,1)).ravel()
        dx_col = dx_col[:, range(dy_col.size)] = 1. / B.shape[0] * dy_col
        
        dx = col2im_cpu(dx_col, self.batch_size, n, self._vol, self._col, self.kernel_size, self.stride, self.padding, self.dilation)
        dx = dx.reshape(x.shape)
        return dx

class AvgPool1d(_AvgPoolNd):
    def __init__(self, kernel_size: int, stride: int=None, padding: int=0, ceil_mode: bool=False, count_include_pad: bool=True) -> None:
        kernel_size = single(kernel_size)
        stride = single(stride if stride is not None else kernel_size)
        padding = single(padding)
        super(AvgPool1d, self).__init__(kernel_size, stride, padding, ceil_mode, count_include_pad, False)

class AvgPool2d(_AvgPoolNd):
    def __init__(self, kernel_size: Union[int, List[int]], stride: Optional[Union[int, List[int]]]=None, padding: Union[int, List[int]]=0,
                 ceil_mode: bool=False, count_include_pad: bool=True, divisor_override: bool=None) -> None:
        kernel_size = double(kernel_size)
        stride = double(stride if stride is not None else kernel_size)
        padding = double(padding)
        super(AvgPool2d, self).__init__(kernel_size, stride, padding, ceil_mode, count_include_pad,)

    def __setstate__(self, d):
        #super(AvgPool2d, self).__setstate__(d)
        self.__dict__.setdefault('padding', 0)
        self.__dict__.setdefault('ceil_mode', False)
        self.__dict__.setdefault('count_include_pad', True)

class AvgPool3d(_AvgPoolNd):
    def __init__(self, kernel_size: Union[int, List[int]], stride: Optional[Union[int, List[int]]]=None, padding: Union[int, List[int]]=0,
                 ceil_mode: bool=False, count_include_pad: bool=True, divisor_override: bool=None) -> None:
        kernel_size = triple(kernel_size)
        stride = triple(stride if stride is not None else kernel_size)
        padding = triple(padding)
        super(AvgPool3d, self).__init__(kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override)

    def __setstate__(self, d):
        #super(AvgPool3d, self).__setstate__(d)
        self.__dict__.setdefault('padding', 0)
        self.__dict__.setdefault('ceil_mode', False)
        self.__dict__.setdefault('count_include_pad', True)