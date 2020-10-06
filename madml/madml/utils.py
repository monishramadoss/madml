from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from .errors import InvalidArguementError
import numpy as np
import math 

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
    B = np.zeros(shape=int(n_output_plane * output_length))
    X = x.flatten()
    for elt in range(batch_size):
        data_vol = elt * in_channels * _vol[0] * _vol[1] * _vol[2]
        data_col = elt * n_output_plane * _col[0] * _col[1] * _col[2]

        for index in range(in_channels * _col[0] * _col[1] * _col[2]):
            w_offset =  int(index) % kernel_size[2]
            h_offset =  int(index / kernel_size[2]) % kernel_size[1]
            d_offset =  int(index / kernel_size[2] / kernel_size[1]) % kernel_size[0]
            c_vol =     int(index / kernel_size[2] / kernel_size[1] / kernel_size[0])

            for d_col in range(_col[0]):
                d_vol = d_col * stride[0] - padding[0] + d_offset * dilation[0]
                for h_col in range(_col[1]):
                    h_vol = h_col * stride[1] - padding[1] + h_offset * dilation[1]
                    for w_col in range(_col[2]):
                        w_vol = w_col * stride[2] - padding[2] + w_offset * dilation[2]
            
                        if d_vol >= 0 and d_vol < _vol[0] and h_vol >= 0 and  h_vol < _vol[1] and w_vol >= 0 and w_vol < _vol[2]:
                            data_col_idx = data_col + ((index * _col[0] + d_col) * _col[1] + h_col) * _col[2] + w_col                     
                            data_vol_idx = data_vol + ((c_vol * _vol[0] + d_vol) * _vol[1] + h_vol) * _vol[2] + w_vol
                            if(data_col_idx < B.size and data_vol_idx < x.size):
                                B[int(data_col_idx)] = X[int(data_vol_idx)]
    B = B.reshape(n_output_plane, output_length)
    return B, n_output_plane, output_length

def bias_add(x, bias, bias_shape):
    y = x
    if bias is None:
        bias = np.ones(bias_shape)
    y += bias
    return y

def _reverse_repeat_tuple(t, n):
    return tuple(x for x in reversed(t) for _ in range(n))