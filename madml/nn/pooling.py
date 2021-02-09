from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from typing import Union, List, Optional

import numpy as np

from madml import tensor, zeros
from .module import Module
from .testing import maxpool_forward, maxpool_backward
from .transform import vol2col


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


class _MaxPoolNd(Module):
    __constants__ = ['kernel_size', 'stride', 'padding', 'dilation', 'return_indices', 'ceil_mode']

    return_indices: bool
    ceil_mode: bool

    def __init__(self, dims, kernel_size: Union[int, List[int]], stride: Union[int, List[int]] = None,
                 padding: Union[int, List[int]] = 0, dilation: Union[int, List[int]] = 1, return_indices: bool = False,
                 ceil_mode: bool = False) -> None:
        super(_MaxPoolNd, self).__init__()
        self.dims = 3

        self.kernel_size = _dim_fix([1 for _ in range(self.dims)], kernel_size, dims)
        self.stride = _dim_fix([1 for _ in range(self.dims)], stride, dims)
        self.padding = _dim_fix([0 for _ in range(self.dims)], padding, dims)
        self.dilation = _dim_fix([1 for _ in range(self.dims)], dilation, dims)
        self.return_indices = return_indices
        self.ceil_mode = ceil_mode
        self.channel_offset = 1
        self._col = []
        self._vol = []
        self.batch_size = 0
        self.in_channels = 0
        self.col = None
        self.kernel = None

    def forward_cpu(self, x: tensor) -> tensor:
        if self._col == [] or self._vol == []:
            self._col = [1 for _ in range(self.dims)]
            self._vol = [1 for _ in range(self.dims)]

            for i in range(self.dims - 1, 0, -1):
                self._col[i] = int((x.shape[i + 2] + 2 * self.padding[i] - self.dilation[i]
                                    * (self.kernel_size[i] - 1) - 1) // self.stride[i]) + 1
                self._vol[i] = x.shape[i + 2]
                self.channel_offset *= self.kernel_size[i]

            self.batch_size = x.shape[0]
            self.in_channels = x.shape[1]
            self.kernel = vol2col(self.batch_size, self.in_channels, self._vol, self._col, self.kernel_size,
                                  self.stride, self.padding, self.dilation)
        y = zeros([x.shape[0], x.shape[1], *self._col])
        y.reshape([self.in_channels * self.batch_size, -1])
        self.col = self.kernel.forward_cpu(x)
        self.col.reshape([self.in_channels * self.batch_size, self.channel_offset, -1])
        max_idx = []
        for i in range(self.in_channels * self.batch_size):
            tmp = self.col.host_data[i]
            m_idx = np.argmax(tmp, axis=0)
            max_idx.append(m_idx)
            y.host_data[i] = self.col.host_data[i][m_idx, range(m_idx.size)]

        y.reshape([self.batch_size, self.in_channels, self._col[0], self._col[1], self._col[2]])
        x.reset_shape()

        self.cache = [x, y, max_idx]
        return y

    def backward_cpu(self) -> tensor:
        x, y, max_idx = self.cache
        dx, dy = x.gradient, y.gradient
        dy_col = dy.host_data.ravel()
        self.col.gradient.reshape([self.in_channels * self.batch_size, self.channel_offset, -1])
        dy_col = dy_col.reshape([self.in_channels * self.batch_size, -1])
        d_col = self.col.gradient.host_data
        for i in range(self.in_channels * self.batch_size):
            m_idx = max_idx[i]
            d_col[i][m_idx, range(m_idx.size)] = dy_col[i]

        self.col.gradient.host_data = d_col
        self.col.reset_shape()
        _ = self.kernel.backward_cpu()
        return x

    def print_l(self) -> None:
        x, y, _ = self.cache
        super(_MaxPoolNd, self).print_l()
        print('\tmax input:', x.host_data.max(), 'g', x.gradient.host_data.max(),
              ' output:', y.host_data.max(), 'g', y.gradient.host_data.max())
        print('\tmin input:', x.host_data.min(), 'g', x.gradient.host_data.min(),
              ' output:', y.host_data.min(), 'g', y.gradient.host_data.min())

    def test(self) -> None:

        return


class MaxPool1d(_MaxPoolNd):
    kernel_size: int
    stride: int
    padding: int
    dilation: int

    def __init__(self, kernel_size: int, stride: Optional[int] = None,
                 padding: int = 0, dilation: int = 1,
                 return_indices: bool = False, ceil_mode: bool = False) -> None:
        super(MaxPool1d, self).__init__(1, kernel_size, stride, padding, dilation, return_indices, ceil_mode)

    def forward_cpu(self, x: tensor) -> tensor:
        x.reshape([x.shape[0], x.shape[1], 1, 1, x.shape[2]])
        y = super(MaxPool1d, self).forward_cpu(x)
        x.reset_shape()
        y.reshape([x.shape[0], y.shape[1], y.shape[-1]])
        return y

    def backward_cpu(self) -> tensor:
        x, y, _ = self.cache
        y.reshape([x.shape[0], y.shape[1], 1, 1, y.shape[2]])
        x.reshape([x.shape[0], x.shape[1], 1, 1, x.shape[2]])
        x = super(MaxPool1d, self).backward_cpu()
        x.reset_shape()
        y.reset_shape()
        return x


class MaxPool2d(_MaxPoolNd):
    kernel_size: Union[int, List[int]]
    stride: Union[int, List[int]]
    padding: Union[int, List[int]]
    dilation: Union[int, List[int]]

    def __init__(self, kernel_size: Union[int, List[int]], stride: Optional[Union[int, List[int]]] = None,
                 padding: Union[int, List[int]] = 0, dilation: Union[int, List[int]] = 1,
                 return_indices: bool = False, ceil_mode: bool = False) -> None:
        super(MaxPool2d, self).__init__(2, kernel_size, stride, padding, dilation, return_indices, ceil_mode)

    def forward_cpu(self, x: tensor) -> tensor:
        x.reshape([x.shape[0], x.shape[1], 1, x.shape[2], x.shape[3]])
        y = super(MaxPool2d, self).forward_cpu(x)
        x.reset_shape()
        y.reshape([x.shape[0], y.shape[1], y.shape[3], y.shape[4]])
        y.init_shape = y.shape
        return y

    def backward_cpu(self) -> tensor:
        x, y, _ = self.cache
        y.reshape([y.shape[0], y.shape[1], 1, y.shape[2], y.shape[3]])
        x.reshape([x.shape[0], x.shape[1], 1, x.shape[2], x.shape[3]])
        x = super(MaxPool2d, self).backward_cpu()
        x.reset_shape()
        y.reset_shape()
        return x

    def test(self) -> None:
        x, y, mx = self.cache
        _y, c = maxpool_forward(x.host_data, self.kernel_size[-1], self.stride[-1])
        _dx = maxpool_backward(y.gradient.host_data, c)
        assert ((y.host_data == _y).all())
        assert ((_dx == x.gradient.host_data).all())


class MaxPool3d(_MaxPoolNd):
    kernel_size: Union[int, List[int]]
    stride: Union[int, List[int]]
    padding: Union[int, List[int]]
    dilation: Union[int, List[int]]

    def __init__(self, kernel_size: Union[int, List[int]], stride: Optional[Union[int, List[int]]] = None,
                 padding: Union[int, List[int]] = 0, dilation: Union[int, List[int]] = 1,
                 return_indices: bool = False, ceil_mode: bool = False) -> None:
        super(MaxPool3d, self).__init__(3, kernel_size, stride, padding, dilation, return_indices, ceil_mode)
