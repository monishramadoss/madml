from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import math
import random
from typing import List, Union
import numpy as np
from numba import njit

from .tensor import tensor

def _size(shape: List[int]) -> int:
    size = 1
    for s in shape:
        size *= s
    return size

def zeros(shape: List[int]) -> tensor:
    data = np.zeros(shape=shape)
    return tensor(data, shape)

def zeros_like(t: tensor) -> tensor:
    return zeros(t.shape)

def ones(shape: List[int]) -> tensor:
    data = np.ones(shape=shape)
    return tensor(data, shape)

def full_like(t: tensor, val: float) -> tensor:
    data = [val for _ in t.size]
    return tensor(data, t.shape)

def fill(shape: List[int], val: float) -> tensor:
    data = [val for _ in range(_size(shape))]
    return tensor(data, shape)

def calc_gain(nonlinearity: str, param: Union[float, int]=None):
    linear_fns = ['linear', 'conv1d', 'conv2d', 'conv3d', 'conv_transpose1d', 'conv_transpose2d', 'conv_transpose3d']
    if nonlinearity in linear_fns or nonlinearity == 'sigmoid':
        return 1
    elif nonlinearity == 'tanh':
        return 5.0 / 3
    elif nonlinearity == 'relu':
        return math.sqrt(2.0)
    elif nonlinearity == 'leaky_relu':
        if param is None:
            negative_slope = 0.01
        elif isinstance(param, int) or isinstance(param, float):
            # True/False are instances of int, hence check above
            negative_slope = param
        else:
            raise ValueError("negative_slope {} not a valid number".format(param))
        return math.sqrt(2.0 / (1 + negative_slope ** 2))
    else:
        raise ValueError("Unsupported nonlinearity {}".format(nonlinearity))

def uniform(a: float=0., b: float=1.):
    def init(shape: List[int]) -> tensor:
        sz = _size(shape)
        data = [random.uniform(a, b) for _ in range(sz)]
        return tensor(data, shape)

    return init

def normal(mean=0., std=1.):
    def init(shape: List[int]) -> tensor:
        sz = _size(shape)
        data = [random.normalvariate(mean, std) for _ in range(sz)]
        return tensor(data, shape)

    return init

def _calculate_fan_in_and_fan_out(shape: List[int]) -> List[int]:
    dim = len(shape)
    if dim < 2:
        raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")

    num_input_fmaps = _size(shape[1:])
    num_output_fmaps = _size(shape)
    receptive_field_size = 1
    if dim > 2:
        receptive_field_size = _size(shape[2:])
    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size
    return [fan_in, fan_out]

def xavier_uniform(gain: float=1.):
    def init(shape: List[int]) -> tensor:
        fan_in, fan_out = _calculate_fan_in_and_fan_out(shape)
        std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
        a = math.sqrt(3.) * std
        return uniform(-a, a)(shape)

    return init

def xavier_normal(gain: float=1.):
    def init(shape: List[int]) -> tensor:
        fan_in, fan_out = _calculate_fan_in_and_fan_out(shape)
        std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
        return normal(0., std)(shape)

    return init

def _calculate_correct_fan(shape: List[int], mode: str) -> int:
    mode = mode.lower()
    valid_modes = ['fan_in', 'fan_out']
    if mode not in valid_modes:
        raise ValueError("Mode {} not supported, please use one of {}".format(mode, valid_modes))
    fan_in, fan_out = _calculate_fan_in_and_fan_out(shape)
    return fan_in if mode == 'fan_in' else fan_out

def kaiming_uniform(a: Union[int, float]=0, mode: str='fan_in', nonlinearity: str='leaky_relu'):
    def init(shape: List[int]) -> tensor:
        fan = _calculate_correct_fan(shape, mode)
        gain = calc_gain(nonlinearity, a)
        std = gain / math.sqrt(fan)
        bound = math.sqrt(3.) * std
        return uniform(-bound, bound)(shape)

    return init

def kaiming_normal(a: Union[int, float]=0, mode: str='fan_in', nonlinearity: str='leaky_relu'):
    def init(shape: List[int]) -> tensor:
        fan = _calculate_correct_fan(shape, mode)
        gain = calc_gain(nonlinearity, a)
        std = gain / math.sqrt(fan)
        return normal(0, std)(shape)

    return init