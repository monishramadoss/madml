from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from typing import Union, List
import math
from madml import tensor
from madml import nn

def _size(shape: List[int]):
    size = 1
    for s in shape:
        size *= s
    return size

def zeros(shape: List[int]) -> tensor:
    data = [0 for _ in range(_size(shape))]
    return tensor(data, shape)

def zeros_like(T: tensor) -> tensor:
    return zeros(T.shape)

def ones(shape: List[int]) -> tensor:
    data = [1 for _ in range(_size(shape))]
    return tensor(data, shape)

def full_like(T: tensor, val: float) -> tensor:
    data = [val for _ in T.size]
    return tensor(data, T.shape)

def fill(shape: List[int], val: float) -> tensor:
    data = [val for _ in range(_size(shape))]
    return tensor(data, shape)

def reduce_sum(T: tensor, axis: Union[List[int], int]=None) -> tensor:
    def _sum(data: List[float], size: int) -> int:
        s = 0
        for x in data:
            s+=x
        return s

    sum = None
    if axis == None:
        sum = tensor([_sum(T.data, T.size)], [1])

    if type(axis) is int and axis == 0:
        A = T.shape[axis]
        top_range = T.shape[:axis]
        bot_range = T.shape[axis + 1:]
        sum = zeros(A)
        for idx in A:
            sum[idx] += _sum(T[idx].data, sum.size)
    else:
        raise NotImplementedError

    return sum

def mean(T1: tensor, axis: int=None) -> tensor:
    r_sum = reduce_sum(T1, axis)
    if len(sum) == 1:
        r_sum /= T1.size
    else:
        r_sum /= r_sum.size
    return r_sum

def var(T1: tensor, axis: int=None) -> tensor:
    m = mean(T1, axis)

    if axis == None:
        m = T1 - m
    else:
        raise NotImplementedError
    m = pow(m, 2.)
    m /= m.size
    return m