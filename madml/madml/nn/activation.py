from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from typing import List, Optional, Union

import numpy as np
from .module import Module

class Threshold(Module):
    __constants__ = ['threshold', 'value', 'inplace']
    threshold : float
    value : float
    inplace : bool

    def __init__(self, threshold: float, value: float, inplace: bool=False) -> None:
        super(Threshold, self).__init__()
        self.threshold = threshold
        self.value = value
        self.inplace = inplace

    def extra_repr(self):
        inplace_str = ', inplace=True' if self.inplace else ''
        return 'threshold={}, value={}{}'.format(self.threshold, self.value, inplace_str)

class ReLU(Module):
    __constatns__ = ['inplace']
    inplace : bool

    def __init__(self, inplace: bool=False) -> None:
        super(ReLU, self).__init__()
        self.inplace = inplace

    def extra_repr(self) -> str:
        inplace_str = 'inplace=True' if self.inplace else ''
        return inplace_str

class RReLU(Module):
    __constants__ = ['lower', 'upper', 'inplace']

    lower : float
    upper : float
    inplace : bool

    def __init__(self, lower: float=1. / 8, upper: float=1. / 3, inplace: bool=False):
        super(RReLU, self).__init__()
        self.lower = lower
        self.upper = upper
        self.inplace = inplace

    def extra_repr(self):
        inplace_str = ', inplace=True' if self.inplace else ''
        return 'lower={}, upper={}{}'.format(self.lower, self.upper, inplace_str)

class Hardtanh(Module):
    __constants__ = ['min_val', 'max_val', 'inplace']

    min_val : float
    max_val : float
    inplace : bool

    def __init__(self, min_val: float=-1., max_val: float=1., inplace: bool=False) -> None:
        super(Hardtanh, self).__init__()
        self.min_val = min_val
        self.max_val = max_val
        self.inplace = inplace

    def extra_repr(self):
        inplace_str = ', inplace=True' if self.inplace else ''
        return 'min_val={}, max_val={}{}'.format(self.min_val, self.max_val, inplace_str)

class ReLU6(Hardtanh):
    def __init__(self, inplace: bool=False):
        super(ReLU6, self).__init__(0., 6., inplace)

    def extra_repr(self) -> str:
        inplace_str = 'inplace=True' if self.inplace else ''
        return inplace_str

class Sigmoid(Module):
    pass

class Hardsigmoid(Module):
    pass

class Hardswish(Module):
    pass

class ELU(Module):
    __constants__ = ['alpha', 'inplace']
    alpha : float
    inplace : bool
    def __init__(self, alpha: float=1., inplace: bool=False) -> None:
        super(ELU, self).__init__()
        self.alpha = alpha
        self.inplace = inplace

    def extra_repr(self) -> str:
        inplace_str = ', inplace=True' if self.inplace else ''
        return 'alpha={}{}'.format(self.alpha, inplace_str)

class CELU(Module):
    __constants__ = ['alpha', 'inplace']
    alpha : float
    inplace : bool

    def __init__(self, alpha: float=1., inplace: bool=False) -> None:
        super(CELU, self).__init__()
        self.alpha = alpha
        self.inplace = inplace

    def extra_repr(self) -> str:
        inplace_str = ', inplace=True' if self.inplace else ''
        return 'alpha={}{}'.format(self.alpha, inplace_str)

class SELU(Module):
    __constants__ = ['inplace']
    inplace : bool

    def __init__(self, inplace: bool=False) -> None:
        super(SELU, self).__init__()
        self.inplace = inplace

    def extra_repr(self) -> str:
        inplace_str = 'inplace=True' if self.inplace else ''
        return inplace_str

class GLU(Module):
    __constants__ = ['dim']
    dim : int

    def __init__(self, dim: int=-1) -> None:
        super(GLU, self).__init__()
        self.dim = dim

    def extra_repr(self) -> str:
        return 'dim={}'.format(self.dim)

class GELU(Module):
    pass

class Hardshrink(Module):
    __constants__ = ['lambd']
    lambd : float

    def  __init__(self, lambd: float=0.5) -> None:
        super(Hardshrink, self).__init__()
        self.lambd = lambd

    def extra_repr(self) -> str:
        return '{}'.format(self.lambd)

class LeakyReLU(Module):
    __constants__ = ['inplace', 'negative_slope']
    inplace : bool
    negative_slope : float

    def __init__(self, negative_slope: float=1e-2, inplace: bool=False) -> None:
        super(LeakyReLU, self).__init__()
        self.negative_slope = negative_slope
        self.inplace = inplace

    def extra_repr(self) -> str:
        inplace_str = ', inplace=True' if self.inplace else ''
        return 'negative_slope={}{}'.format(self.negative_slope, inplace_str)

class LogSigmoid(Module):
    pass

class Softplus(Module):
    __constants__ = ['beta', 'threshold']
    beta : int
    threshold : int

    def __init__(self, beta: int=1, threshold: int=20) -> None:
        super(Softplus, self).__init__()
        self.beta = beta
        self.threshold = threshold

    def extra_repr(self) -> str:
        return 'beta={}, threshold={}'.format(self.beta, self.threshold)

class PReLU(Module):
    __constants__ = ['num_parameters']
    num_parameters : int

    def __init__(self, num_parameters: int=1, init: float=0.25) -> None:
        super(PReLU, self).__init__()
        self.num_parameters = num_parameters
        self.weight = 0#Parameter(torch.Tensor(num_parameters).fill_(init))

    def extra_repr(self) -> str:
        return 'num_parameters={}'.format(self.num_parameters)

class Softsign(Module):
    pass

class Tanhshrink(Module):
    pass

class Softmin(Module):
    __constants__ = ['dim']
    dim : Optional[int]

    def __init__(self, dim: Optional[int]=None) -> None:
        super(Softmin, self).__init__()
        self.dim = dim

    def __setstate__(self, state):
        self.__dict__.update(state)
        if not hasattr(self, 'dim'):
            self.dim = None

    def extra_repr(self):
        return 'dim={dim}'.format(dim=self.dim)

class Softmax(Module):
    __constants__ = ['dim']
    dim : Optional[int]

    def __init__(self, dim: Optional[int]=None) -> None:
        super(Softmax, self).__init__()
        self.dim = dim

    def __setstate__(self, state):
        self.__dict__.update(state)
        if not hasattr(self, 'dim'):
            self.dim = None

    def extra_repr(self) -> str:
        return 'dim={dim}'.format(dim=self.dim)