from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from .errors import InvalidArguementError

from multiprocessing import Pool, Array
import backend
import madml

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

def im2col(x, batch_size, in_channels, _vol, _col, kernel_size, stride, padding, dilation):
    n_output_plane = int(in_channels * kernel_size[0] * kernel_size[1] * kernel_size[2])
    output_length = int(batch_size * _col[0] * _col[1] * _col[2])
    B = madml.zeros(n_output_plane * output_length)
    X = x.flatten()
    args = [batch_size, in_channels, *_vol, *_col, *kernel_size, *stride, *padding, *dilation]
    B = backend.im2col_cpu(X, B, args)
    B = B.reshape(n_output_plane, output_length)
    return B, n_output_plane, output_length

def col2im(x, batch_size, in_channels, _vol, _col, kernel_size, stride, padding, dilation):
    n_output_plane = int(in_channels * kernel_size[0] * kernel_size[1] * kernel_size[2])
    output_length = int(batch_size * _col[0] * _col[1] * _col[2])
    B = madml.zeros(n_output_plane * output_length)
    X = x.flatten()
    args = [batch_size, in_channels, *_col, *_vol, *kernel_size, *stride, *padding, *dilation]
    B = backend.im2col_cpu(X, B, args)
    B = B.reshape(batch_size, in_channels, *_col)
    return B, n_output_plane, output_length

def bias_add(x, bias, bias_shape):
    y = x
    if bias is None:
        bias = madml.ones(bias_shape)
    y += bias
    return y

def _reverse_repeat_tuple(t, n):
    return tuple(x for x in reversed(t) for _ in range(n))