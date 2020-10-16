from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from typing import  Optional
from .module import Module
from madml import tensor
import madml
import random
import backend

class Threshold(Module):
    __constants__ = ['threshold', 'value', 'inplace']
    threshold : float
    value : float
    inplace : bool

    def __init__(self, threshold: float, value: float, inplace: bool=False) -> None:
        super(Threshold, self).__init__(None)
        self.threshold = threshold
        self.value = value
        self.inplace = inplace

    def extra_repr(self):
        inplace_str = ', inplace=True' if self.inplace else ''
        return 'threshold={}, value={}{}'.format(self.threshold, self.value, inplace_str)

    def forward_cpu(self, x: tensor) -> tensor:
        self.cache = [x]
        #x[x <= self.threshold] = self.value
        return x

    def backward_cpu(self, dy: tensor) -> tensor:
        dx = dy
        #dx[self.cache[0] <= self.threshold] = 0
        return dx

class ReLU(Module):
    __constatns__ = ['inplace']
    inplace : bool

    def __init__(self, inplace: bool=False) -> None:
        super(ReLU, self).__init__(backend.relu(inplace))
        self.inplace = inplace

    def extra_repr(self) -> str:
        inplace_str = 'inplace=True' if self.inplace else ''
        return inplace_str

    def forward_cpu(self, x: tensor) -> tensor:
        self.cache = [x]
        return madml.max(x, axis=0)

    def backward_cpu(self, dy: tensor) -> tensor:
        dx = dy.copy()
        dx[self.cache[0] <= 0] = 0
        return dx

class RReLU(Module):
    __constants__ = ['lower', 'upper', 'inplace']

    lower : float
    upper : float
    inplace : bool

    def __init__(self, lower: float=1. / 8, upper: float=1. / 3, inplace: bool=False):
        super(RReLU, self).__init__(backend.rrelu(lower, upper, inplace))
        self.lower = lower
        self.upper = upper
        self.inplace = inplace

    def extra_repr(self):
        inplace_str = ', inplace=True' if self.inplace else ''
        return 'lower={}, upper={}{}'.format(self.lower, self.upper, inplace_str)

    def forward_cpu(self, x: tensor) -> tensor:
        self.cache[0]
        x[x < 0] = x * random.uniform(self.lower, self.upper)
        return x
    def backward_cpu(self, dy: tensor) -> tensor:
        dx = dy.copy()
        dx[self.cache[0] <= 0] = 0
        return dx

class Hardtanh(Module):
    __constants__ = ['min_val', 'max_val', 'inplace']

    min_val : float
    max_val : float
    inplace : bool

    def __init__(self, min_val: float=-1., max_val: float=1., inplace: bool=False) -> None:
        super(Hardtanh, self).__init__(backend.hardtanh(min_val, max_val, inplace))
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
    __constant__ = ['inplace']
    inplace : bool
    def __init__(self, inplace: bool=True):
        super(Sigmoid, self).__init__(backend.sigmoid(inplace))
        self.inplace = inplace

    def forward_cpu(self, x: tensor) -> tensor:
        y = 1. / (1 + madml.exp(-x))
        self.cache = [y]
        return y

    def backward_cpu(self, dy: tensor) -> tensor:
        return self.cache[0] * (1. - self.cache[0]) * dy

class Hardsigmoid(Module):
    inplace : bool
    def __init__(self, inplace: bool=True):
        super(Hardsigmoid, self).__init__(backend.hardsigmoid(inplace))
        self.inplace = inplace

class Hardswish(Module):
    inplace : bool
    def __init__(self, inplace: bool=True):
        super(Hardswish, self).__init__(backend.hardswish(inplace))
        self.inplace = inplace

class ELU(Module):
    __constants__ = ['alpha', 'inplace']
    alpha : float
    inplace : bool
    def __init__(self, alpha: float=1., inplace: bool=False) -> None:
        super(ELU, self).__init__(backend.elu(alpha, inplace))
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
        super(CELU, self).__init__(backend.celu(alpha, inplace))
        self.alpha = alpha
        self.inplace = inplace

    def extra_repr(self) -> str:
        inplace_str = ', inplace=True' if self.inplace else ''
        return 'alpha={}{}'.format(self.alpha, inplace_str)

class SELU(Module):
    __constants__ = ['inplace']
    inplace : bool

    def __init__(self, inplace: bool=False) -> None:
        super(SELU, self).__init__(backend.selu(inplace))
        self.inplace = inplace

    def extra_repr(self) -> str:
        inplace_str = 'inplace=True' if self.inplace else ''
        return inplace_str

class GLU(Module):
    __constants__ = ['dim']
    dim : int

    def __init__(self, dim: int=-1) -> None:
        super(GLU, self).__init__(None)
        self.dim = dim

    def extra_repr(self) -> str:
        return 'dim={}'.format(self.dim)

class GELU(Module):
    def __init__(self, inplace: bool=False) -> None:
        super(GELU, self).__init__(None)
        self.inplace = inplace

class Hardshrink(Module):
    __constants__ = ['lambd', 'inplace']
    inplace : bool
    lambd : float
    def  __init__(self, lambd: float=0.5, inplace: bool=False) -> None:
        super(Hardshrink, self).__init__(backend.hardshrink(lambd, inplace))
        self.lambd = lambd
        self.inplace = inplace

    def extra_repr(self) -> str:
        return '{}'.format(self.lambd)

class LeakyReLU(Module):
    __constants__ = ['inplace', 'negative_slope']
    inplace : bool
    negative_slope : float

    def __init__(self, negative_slope: float=1e-2, inplace: bool=False) -> None:
        super(LeakyReLU, self).__init__(backend.leakyrelu(negative_slope, inplace))
        self.negative_slope = negative_slope
        self.inplace = inplace

    def extra_repr(self) -> str:
        inplace_str = ', inplace=True' if self.inplace else ''
        return 'negative_slope={}{}'.format(self.negative_slope, inplace_str)

class LogSigmoid(Module):
    __constants__ = ['alpha', 'inplace']
    inplace : bool
    alpha : float
    def __init__(self, alpha: float=-0.01, inplace: bool=False) -> None:
        super(LogSigmoid, self).__init__(backend.logsigmoid(alpha, inplace))
        self.alpha = alpha
        self.inplace = inplace

    def extra_repr(self) -> str:
        inplace_str = ', inplace=True' if self.inplace else ''
        return 'alpha={}{}'.format(self.alpha, inplace_str)

class Softplus(Module):
    __constants__ = ['beta', 'threshold']
    beta : int
    threshold : int
    def __init__(self, beta: int=1, threshold: int=20) -> None:
        super(Softplus, self).__init__(backend.softplus(float(beta), False))
        self.beta = beta
        self.threshold = threshold

    def extra_repr(self) -> str:
        return 'beta={}, threshold={}'.format(self.beta, self.threshold)

class PReLU(Module):
    __constants__ = ['num_parameters']
    num_parameters : int

    def __init__(self, num_parameters: int=1, init: float=0.25) -> None:
        super(PReLU, self).__init__(backend.prelu(-0.01, False))
        self.num_parameters = num_parameters
        self.weight = madml.fill(num_parameters, init)

    def extra_repr(self) -> str:
        return 'num_parameters={}'.format(self.num_parameters)

class Softsign(Module):
    __constants__ = ['alpha']
    alpha : float

    def __init__(self, alpha: float=0.01) -> None:
        super(Softsign, self).__init__(backend.softsign(alpha, False))
        self.alpha = alpha

class Tanhshrink(Module):
    __constants__ = ['inplace']
    inplace : bool

    def __init__(self, inplace: bool=False) -> None:
        super(Tanhshrink, self).__init__(backend.tanhshrink(inplace))
        self.inplace = inplace

class Softmin(Module):
    __constants__ = ['dim']
    dim : Optional[int]

    def __init__(self, dim: Optional[int]=None) -> None:
        super(Softmin, self).__init__(None)
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
        super(Softmax, self).__init__(None)
        self.dim = dim

    def __setstate__(self, state):
        self.__dict__.update(state)
        if not hasattr(self, 'dim'):
            self.dim = None

    def extra_repr(self) -> str:
        return 'dim={dim}'.format(dim=self.dim)