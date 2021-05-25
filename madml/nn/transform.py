from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import math
from typing import List

import numba.typed as nbt
import numpy as np
from numba import njit, prange

import vknn
from madml import tensor
from .module import Module


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
        self.vol_col = self.register_kernel(vknn.vol2col, [batch_size, in_channels, *kernel_size, *padding, *stride,
                                                           *dilation, *_col, *_vol, ])
        self.col_vol = self.register_kernel(vknn.col2vol, [batch_size, in_channels, *kernel_size, *padding, *stride,
                                                           *dilation, *_col, *_vol, ])

    def forward(self, x: tensor) -> tensor:
        self.register_output_shape([self.n_output_plane, self.output_length])
        self.register_forward_arg('x', x)
        self.register_backward_arg('x', x)
        self.register_backward_arg('y', self.y)
        super(vol2col, self).forward(x)
        return self.y

    def _forward_cpu(self, x: tensor) -> tensor:
        _vol2col(x.host_data.ravel(), self.y.host_data.ravel(), self.batch_size, self.in_channels,
                 self.n_output_plane, self.index_length, nbt.List(self._vol), nbt.List(self._col),
                 nbt.List(self.kernel_size), nbt.List(self.stride), nbt.List(self.padding), nbt.List(self.dilation))
        return self.y

    def _forward_gpu(self, x: tensor) -> tensor:
        self.vol_col.forward(self.y.device_data, x.device_data)
        self.vol_col.run()
        return self.y

    def _backward_cpu(self, x: tensor, y: tensor):
        dx = x.gradient
        npdx = x.gradient.host_data.ravel()
        dcol = y.gradient.host_data.ravel()
        _col2vol(npdx, dcol, self.batch_size, self.in_channels,
                 self.n_output_plane, self.index_length, nbt.List(self._vol), nbt.List(self._col),
                 nbt.List(self.kernel_size), nbt.List(self.stride), nbt.List(self.padding), nbt.List(self.dilation))
        dx.host_data = npdx.reshape(x.shape)
        return dx

    def _backward_gpu(self, x: tensor, y: tensor):
        dx = x.gradient
        dcol = y.gradient
        self.col_vol.forward(dx.device_data, dcol.device_data)
        self.col_vol.run()
        return dx

    def print_l(self):
        x, y = self.cache[0], self.col
        super(vol2col, self).print_l()
        print('\tmax input:', x.host_data.max(), 'g', x.gradient.host_data.max(),
              ' output:', y.host_data.max(), 'g', y.gradient.host_data.max())
        print('\tmin input:', x.host_data.min(), 'g', x.gradient.host_data.min(),
              ' output:', y.host_data.min(), 'g', y.gradient.host_data.min())


class col2vol(vol2col):
    def __init__(self,
                 batch_size: int,
                 in_channels: int,
                 _vol: List,
                 _col: List,
                 kernel_size: List,
                 stride: List,
                 padding: List,
                 dilation: List):
        super(col2vol, self).__init__(batch_size, in_channels, _vol, _col, kernel_size, stride, padding, dilation)

    def _forward_cpu(self, x: tensor) -> tensor:
        _col2vol(x.host_data.ravel(), self.y.host_data.ravel(), self.batch_size, self.in_channels,
                 self.n_output_plane, self.index_length, nbt.List(self._vol), nbt.List(self._col),
                 nbt.List(self.kernel_size), nbt.List(self.stride), nbt.List(self.padding), nbt.List(self.dilation))

    def _forward_gpu(self, x: tensor) -> tensor:
        self.col_vol.forward(self.y.device_data, x.device_data)
        self.col_vol.run()
        return self.y

    def _backward_cpu(self, x: tensor, y: tensor) -> tensor:
        dx = x.gradient
        npdx = x.gradient.host_data.ravel()
        dcol = y.gradient.host_data.ravel()
        _vol2col(npdx, dcol, self.batch_size, self.in_channels,
                 self.n_output_plane, self.index_length, nbt.List(self._vol), nbt.List(self._col),
                 nbt.List(self.kernel_size), nbt.List(self.stride), nbt.List(self.padding), nbt.List(self.dilation))
        dx.host_data = npdx.reshape(x.shape)
        return dx

    def _backward_gpu(self, x: tensor, y: tensor) -> tensor:
        dx = x.gradient
        dvol = y.gradient
        self.vol_col.forward(dx.device_data, dvol.device_data)
        self.vol_col.run()
        return dx


class transpose(Module):
    __constants__ = ['axes']
    axes: List
    stride: List
    old_shape: List
    new_shape: List

    def __init__(self, axes: List[int], in_place: bool) -> None:
        super(transpose, self).__init__()
        self.axes = axes
        self.old_shape = []
        self.new_shape = []
        self.stride = [1 for _ in range(len(axes) * 3)]
        self.in_place = in_place
        self.kernel = self.register_kernel(vknn.transpose, self.axes)
        self.kernel_dx = self.register_kernel(vknn.transpose, self.axes)

    def forward(self, x: tensor) -> tensor:
        assert (len(x.shape) == len(self.axes))
        if self.in_place:
            self.y = x
        elif self.y is None:
            self.register_output_shape(self.new_shape)
        if not self.new_shape:
            self.old_shape = [s for s in x.shape]
            self.new_shape = [self.old_shape[self.axes[i]] for i in range(len(self.axes))]
        self.register_forward_arg('x', x)
        self.register_backward_arg('dx', x.gradient)
        self.register_backward_arg('dy', self.y.gradient)
        super(transpose, self).forward(x)
        return self.y

    def _forward_cpu(self, x: tensor) -> tensor:
        self.y.host_data = np.transpose(x.host_data, self.axes)
        return self.y

    def _forward_gpu(self, x: tensor) -> tensor:
        self.kernel.forward(self.y.device_data, x.device_data)
        self.kernel.run()
        self.y.reshape(self.new_shape)
        return self.y

    def _backward_cpu(self, dx: tensor, dy: tensor) -> tensor:
        dx.host_data = np.transpose(dy.host_data, self.axes)
        return dx

    def _backward_gpu(self, dx: tensor, dy: tensor) -> tensor:
        self.kernel_dx.forward(dx.device_data, dy.device_data)
        self.kernel_dx.run()
        return dx

    def prepare_stride(self, shape_before: List, shape_after: List) -> None:
        dims = len(self.axes)
        self.stride[2 * dims - 1] = 1
        self.stride[3 * dims - 1] = 1
        for i in range(dims - 2, 0, -1):
            self.stride[dims * 2 + i] = self.stride[dims * 2 + i + 1] * shape_before[i + 1]
            self.stride[dims + i] = self.stride[dims + i + 1] * shape_after[i + 1]


class flatten(Module):
    def __init__(self) -> None:
        super(flatten, self).__init__()
        self.old_shape = []

    def forward(self, x: tensor) -> tensor:
        self.old_shape = x.shape
        self.register_forward_arg('x', x)
        self.register_backward_arg('dx', x.gradient)
        self.y = x
        super(flatten, self).forward(x)
        return x

    def _forward_cpu(self, x: tensor) -> tensor:
        x.reshape([x.shape[0], -1])
        return x

    def _forward_gpu(self, x: tensor) -> tensor:
        x.reshape([x.shape[0], -1])
        return x

    def _backward_cpu(self, dx: tensor) -> tensor:
        dx.reshape(self.old_shape)
        return dx

    def _backward_gpu(self, dx: tensor) -> tensor:
        dx.reshape(self.old_shape)
        return dx


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
                            data_vol_idx = math.floor(
                                data_vol + (((c_vol * _vol[0] + d_vol) * _vol[1] + h_vol) * _vol[2] + w_vol))
                            data_col_idx = math.floor(
                                data_col + (((c_col * _col[0] + d_col) * _col[1] + h_col) * _col[2] + w_col))
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
                            data_vol_idx = math.floor(
                                data_vol + (((c_vol * _vol[0] + d_vol) * _vol[1] + h_vol) * _vol[2] + w_vol))
                            data_col_idx = math.floor(
                                data_col + (((c_col * _col[0] + d_col) * _col[1] + h_col) * _col[2] + w_col))
                            data_vol_idx = int(data_vol_idx)
                            data_col_idx = int(data_col_idx)
                            if data_col_idx < col.size and data_vol_idx < vol.size:
                                vol[data_vol_idx] += col[data_col_idx]
