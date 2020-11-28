from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from .errors import InvalidArguementError

from multiprocessing import Pool, Array
import backend
import numpy as np

def _ntuple(i):
    def parse(x):
        if type(x) is int:
            return [x for j in range(i)]
        else:
            return [x[j] for j in range(i)]
    return parse

single = _ntuple(1)
double = _ntuple(2)
triple = _ntuple(3)

def im2col_cpu(x, batch_size, in_channels, _vol, _col, kernel_size, stride, padding, dilation):
    n_output_plane = int(in_channels * kernel_size[0] * kernel_size[1] * kernel_size[2])
    output_length = int(batch_size * _col[0] * _col[1] * _col[2])
    B = np.zeros(n_output_plane * output_length)
    X = x.flatten()
    args = [batch_size, in_channels, *_vol, *_col, *kernel_size, *stride, *padding, *dilation]
    B = backend.im2col_cpu(X, B, args)
    B = B.reshape(n_output_plane, output_length)
    return B, n_output_plane, output_length

def col2im_cpu(x, batch_size, in_channels, _vol, _col, kernel_size, stride, padding, dilation):
    B = np.zeros(batch_size* in_channels * _vol[0] * _vol[1] * _vol[2])
    X = x.flatten()
    args = [batch_size, in_channels, *_vol, *_col, *kernel_size, *stride, *padding, *dilation]
    B = backend.col2im_cpu(X, B, args)
    B = B.reshape(batch_size, in_channels, *_vol)
    return B

def _reverse_repeat_tuple(t, n):
    return tuple(x for x in reversed(t) for _ in range(n))