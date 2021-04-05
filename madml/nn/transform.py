from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import math
from typing import List

import numba.typed as nbt
import numpy as np
from numba import njit, prange

from madml import tensor, zeros
from .module import Module
import vknn

class vol2col(Module):
    def __init__(self,
                 batch_size: int,
                 in_channels: int,
                 _vol: List,
                 _col: List,
                 kernel_size: List,
                 stride: List,
                 padding: List,
                 dilation: List):
        super(vol2col, self).__init__()
        self.batch_size = batch_size
        self.in_channels = in_channels
        self._vol = _vol
        self._col = _col
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.n_output_plane = self.in_channels
        self.output_length = self.batch_size
        self.index_length = self.in_channels
        self._c = 1
        for k in self.kernel_size:
            self.n_output_plane *= k
        for c in self._col:
            self.output_length *= c
            self.index_length *= c
            self._c *= c
        self.col = zeros([self.n_output_plane, self.output_length])
        self.kernel = vknn.vol2col([batch_size, in_channels, *kernel_size, *padding, *stride, *dilation, *_col, *_vol,])
        self.kernel_backward = vknn.col2vol([batch_size, in_channels, *kernel_size, *padding, *stride, *dilation, *_col, *_vol,])


    def forward_cpu(self, x: tensor) -> tensor:
        _vol2col(x.host_data.ravel(), self.col.host_data.ravel(), self.batch_size, self.in_channels,
                 self.n_output_plane, self.index_length, nbt.List(self._vol), nbt.List(self._col),
                 nbt.List(self.kernel_size), nbt.List(self.stride), nbt.List(self.padding), nbt.List(self.dilation))
        self.cache = [x]
        return self.col

    def forward_gpu(self, x: tensor) -> tensor:
        self.kernel.forward(x.device_data, self.col.device_data)
        self.cache = [x]
        return self.col

    def backward_cpu(self):
        x = self.cache[0]
        dx = x.gradient.host_data.ravel()
        dcol = self.col.gradient.host_data.ravel()
        _col2vol(dx, dcol, self.batch_size, self.in_channels,
                 self.n_output_plane, self.index_length, nbt.List(self._vol), nbt.List(self._col),
                 nbt.List(self.kernel_size), nbt.List(self.stride), nbt.List(self.padding), nbt.List(self.dilation))
        x.gradient.host_data = dx.reshape(x.shape)
        self.col = zeros([self.n_output_plane, self.output_length])
        return x

    def backward_gpu(self):
        x = self.cache[0]
        dx = x.gradient.device_data
        dcol = self.col.gradient.device_data
        self.kernel_backward.forward(dx, dcol)
        return x

    def print_l(self):
        x, y = self.cache[0], self.col
        super(vol2col, self).print_l()
        print('\tmax input:', x.host_data.max(), 'g', x.gradient.host_data.max(),
              ' output:', y.host_data.max(), 'g', y.gradient.host_data.max())
        print('\tmin input:', x.host_data.min(), 'g', x.gradient.host_data.min(),
              ' output:', y.host_data.min(), 'g', y.gradient.host_data.min())

class transpose(Module):
    __constants__ = ['axes']
    axes : List
    stride : List
    old_shape : List
    new_shape : List

    def __init__(self, axes: List) -> None:
        super(transpose, self).__init__()
        self.axes = axes
        self.old_shape = []
        self.new_shape = []
        self.stride = [1 for _ in range(len(axes) * 3)]

    def forward_cpu(self, x: tensor) -> tensor:
        assert (len(x.shape) == len(self.axes))
        if self.old_shape == [] or self.new_shape == []:
            self.old_shape = [s for s in x.shape]
            self.new_shape = [self.old_shape[self.axes[i]] for i in range(len(self.axes))]
            self.prepare_stride(self.old_shape, self.new_shape)

        y = zeros(self.new_shape)
        for i in range(x.size):
            old_pos = 0
            new_pos = i
            for j in range(len(x.shape)):
                order = self.stride[j]
                old_pos += (new_pos / self.stride[len(x.shape) + j] * self.stride[len(x.shape) * 2 + order])
                new_pos %= self.stride[len(x.shape) + j]

            y.host_data[i] = x.host_data[old_pos]
        self.cache.append(x)
        self.cache.append(y)
        return y

    def backward_cpu(self) -> None:
        x, y = self.cache
        self.prepare_stride(self.new_shape, self.old_shape)
        for i in range(x.size):
            old_pos = 0
            new_pos = i
            for j in range(len(x.shape)):
                order = self.stride[j]
                old_pos += (new_pos / self.stride[len(x.shape) + j] * self.stride[len(x.shape) * 2 + order])
                new_pos %= self.stride[len(x.shape) + j]
            x.gradient.host_data[i] = y.gradient.host_data[old_pos]
        return x

    def prepare_stride(self, shape_before: List, shape_after: List) -> None:
        dims = len(self.axes)
        self.stride[2 * dims - 1] = 1
        self.stride[3 * dims - 1] = 1
        for i in range(dims - 2, 0, -1):
            self.stride[dims * 2 + i] = self.stride[dims * 2 + i + 1] * shape_before[i + 1]
            self.stride[dims + i] = self.stride[dims + i + 1] * shape_after[i + 1]

@njit(parallel=True)
def _vol2col(vol: np.ndarray, col: np.ndarray, batch_size: int, in_channels: int, n_output_plane: int,
             index_length: int,
             _vol: nbt.List, _col: nbt.List, kernel_size: nbt.List, stride: nbt.List, padding: nbt.List,
             dilation: nbt.List):
    for elt in prange(batch_size):
        data_vol = elt * in_channels * _vol[0] * _vol[1] * _vol[2]
        data_col = elt * n_output_plane * _col[0] * _col[1] * _col[2]
        for c_col in range(index_length):
            w_offset = c_col % kernel_size[2]
            h_offset = int(c_col / kernel_size[2]) % kernel_size[1]
            d_offset = int(c_col / kernel_size[2] / kernel_size[1]) % kernel_size[0]
            c_vol = int(c_col / kernel_size[2] / kernel_size[1] / kernel_size[0])
            for d_col in range(_col[0]):
                d_vol = d_col * stride[0] - padding[0] + d_offset * dilation[0]
                for h_col in range(_col[1]):
                    h_vol = h_col * stride[1] - padding[1] + h_offset * dilation[1]
                    for w_col in range(_col[2]):
                        w_vol = w_col * stride[2] - padding[2] + w_offset * dilation[2]
                        if 0 <= d_vol < _vol[0] and 0 <= h_vol < _vol[1] and 0 <= w_vol < _vol[2]:
                            data_vol_idx = math.floor(data_vol + (((c_vol * _vol[0] + d_vol) * _vol[1] + h_vol) * _vol[2] + w_vol))
                            data_col_idx = math.floor(data_col + (((c_col * _col[0] + d_col) * _col[1] + h_col) * _col[2] + w_col))
                            data_vol_idx = int(data_vol_idx)
                            data_col_idx = int(data_col_idx)
                            if data_col_idx < col.size and data_vol_idx < vol.size:
                                col[data_col_idx] = vol[data_vol_idx]

@njit(parallel=True)
def _col2vol(vol: np.ndarray, col: np.ndarray, batch_size: int, in_channels: int, n_output_plane: int,
             index_length: int,
             _vol: nbt.List, _col: nbt.List, kernel_size: nbt.List, stride: nbt.List, padding: nbt.List,
             dilation: nbt.List):
    for elt in prange(batch_size):
        data_vol = elt * in_channels * _vol[0] * _vol[1] * _vol[2]
        data_col = elt * n_output_plane * _col[0] * _col[1] * _col[2]
        for c_col in range(index_length):
            w_offset = c_col % kernel_size[2]
            h_offset = int(c_col / kernel_size[2]) % kernel_size[1]
            d_offset = int(c_col / kernel_size[2] / kernel_size[1]) % kernel_size[0]
            c_vol = int(c_col / kernel_size[2] / kernel_size[1] / kernel_size[0])
            for d_col in range(_col[0]):
                d_vol = d_col * stride[0] - padding[0] + d_offset * dilation[0]
                for h_col in range(_col[1]):
                    h_vol = h_col * stride[1] - padding[1] + h_offset * dilation[1]
                    for w_col in range(_col[2]):
                        w_vol = w_col * stride[2] - padding[2] + w_offset * dilation[2]
                        if 0 <= d_vol < _vol[0] and 0 <= h_vol < _vol[1] and 0 <= w_vol < _vol[2]:
                            data_vol_idx = math.floor(data_vol + (((c_vol * _vol[0] + d_vol) * _vol[1] + h_vol) * _vol[2] + w_vol))
                            data_col_idx = math.floor(data_col + (((c_col * _col[0] + d_col) * _col[1] + h_col) * _col[2] + w_col))
                            data_vol_idx = int(data_vol_idx)
                            data_col_idx = int(data_col_idx)
                            if data_col_idx < col.size and data_vol_idx < vol.size:
                                vol[data_vol_idx] += col[data_col_idx]