from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from typing import  Optional
from .module import Module, Parameter
from madml import tensor
import madml
import random
import numpy as np
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

    def forward_cpu(self, x: np.ndarray) -> np.ndarray:
        self.cahce = [x]
        x [x<= self.threshold] = self.value
        return x

    def backward_cpu(self, dy: np.ndarray) -> np.ndarray:
        dx = dy.copy()
        dx[self.cache[0] <= self.threshold] = 0
        return dx

    def forward_gpu(self, x: tensor) -> tensor:
        return x

    def backward_gpu(self, dy: tensor) -> tensor:
        return dy

class ReLU(Module):
    __constatns__ = ['inplace']
    inplace : bool

    def __init__(self, inplace: bool=False) -> None:
        super(ReLU, self).__init__(backend.relu(inplace))
        #super(ReLU, self).__init__()
        self.inplace = inplace

    def extra_repr(self) -> str:
        inplace_str = 'inplace=True' if self.inplace else ''
        return inplace_str

    def forward_cpu(self, x: np.ndarray) -> np.ndarray:
        self.cache = [x]
        return np.maximum(x, 0)

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

    def forward_cpu(self, x: np.ndarray) -> np.ndarray:
        self.cache = [x]
        y = x.copy()
        y[x < 0] *= random.uniform(self.lower, self.upper)
        return y

    def backward_cpu(self, dy: np.ndarray) -> np.ndarray:
        dx = dy.copy()
        dx[self.cache[0] <= 0] *= random.uniform(self.lower, self.upper)
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

    def forward_cpu(self, x: np.ndarray) -> np.ndarray:
        y = x.copy()
        y[x > 1] = 1
        y[x < -1] = -1
        self.cache = [x]
        return x        
    
    def backward_cpu(self, dy: np.ndarray) -> np.ndarray:
        x = self.cache
        dx = dy.copy()
        dx[x > 1] = 0
        dx[x < -1] = 0
        return dx

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

    def forward_cpu(self, x: np.ndarray) -> np.ndarray:
        y = 1. / (1 + np.exp(-x))
        self.cache = [y]
        return y

    def backward_cpu(self, dy: np.ndarray) -> np.ndarray:
        return self.cache[0] * (1. - self.cache[0]) * dy

class Hardsigmoid(Module):
    inplace : bool
    def __init__(self, inplace: bool=True):
        super(Hardsigmoid, self).__init__(backend.hardsigmoid(inplace))
        self.inplace = inplace

    def forward_cpu(self, x: np.ndarray) -> np.ndarray:
        y = x.copy()
        y[x >= 3] = 1
        y[x <= -3] = 0
        y[not(x>=3 and x <= -3)] = x / 6 + 0.5 
        self.cache = [x]
        return x       

    def backward_cpu(self, dy: np.ndarray) -> np.ndarray:
        x = self.cache[0]
        dx = dy.copy()
        dx[x >= 3] = 0
        dx[x <= -3] = 0
        dx[not(x >= 3 and x <= -3)] = 1 / 6 
        return dx       


class Hardswish(Module):
    inplace : bool
    def __init__(self, inplace: bool=True):
        super(Hardswish, self).__init__(backend.hardswish(inplace))
        self.inplace = inplace

    def forward_cpu(self, x: np.ndarray) -> np.ndarray:
        y = x.copy()
        y[x <= -3] = 0
      
        y[not(x<=-3 and x>=3)] = x * (x + 3) / 6
    
    def backward_cpu(self, dy: np.ndarray) -> np.ndarray:
        x = self.cache[0]
        dx = dy.copy()
        dx[x >= 3] = 0
        dx[x <= -3] = 1
        dx[not(x >= 3 and x <= -3)] = 1 / 6 * (2 * dx + 3)
        return dx   

class ELU(Module):
    __constants__ = ['alpha', 'inplace']
    alpha: float
    inplace: bool
    def __init__(self, alpha: float=1., inplace: bool=False) -> None:
        super(ELU, self).__init__(backend.elu(alpha, inplace))
        self.alpha = alpha
        self.inplace = inplace

    def extra_repr(self) -> str:
        inplace_str = ', inplace=True' if self.inplace else ''
        return 'alpha={}{}'.format(self.alpha, inplace_str)

    def forward_cpu(self, x: np.ndarray) -> np.ndarray:
        y = x.copy()
        y[x <= 3] = self.alpha * np.exp(x) - 1.0
        self.cache = [x]
        return y

    def backward_cpu(self, dy: np.ndarray) -> np.ndarray:
        x = self.cache[0]
        dx = dy.copy()
        dx[x>0] = 1
        dx[x<=0] = self.alpha * np.exp(dx)
        return dx   


class CELU(Module):
    __constants__ = ['alpha', 'inplace']
    alpha: float
    inplace: bool

    def __init__(self, alpha: float=1., inplace: bool=False) -> None:
        super(CELU, self).__init__(backend.celu(alpha, inplace))
        self.alpha = alpha
        self.inplace = inplace

    def extra_repr(self) -> str:
        inplace_str = ', inplace=True' if self.inplace else ''
        return 'alpha={}{}'.format(self.alpha, inplace_str)

    def forward_cpu(self, x: np.ndarray) -> np.ndarray:
        y = np.maximum(0, x) + np.minimum(0, self.alpha * (np.exp(x / self.alpha) - 1))
        self.cache = [x]
        return y

    def backward_cpu(self, dy: np.ndarray) -> np.ndarray:
        x = self.cache[0]
        dx = dy.copy()
        dx[x>=0] = 1
        dx[x<0] = np.exp(dx / self.alpha)
        return dx

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

    def forward_cpu(self, x: np.ndarray) -> np.ndarray:
        y = np.zero_like(x)
        y[x > self.lambd or x < -self.lambd] = x
        self.cache = [x]
        return y

    def backward_cpu(self, dy: np.ndarray) -> np.ndarray:
        x = self.cache[0]
        dx = np.zero_like(dy)
        dx[x > self.lambd or x < -self.lambd] = dx
        return dx

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

    def forward_cpu(self, x: np.ndarray) -> np.ndarray:
        out = np.maximum(self.negative_slope * x, x)
        self.cache = [x]

    def backward_cpu(self, dy: np.ndarray) -> np.ndarray:
        x = self.cache[0]
        dx = dy.copy()
        dx[x < 0] *= self.negative_slope
        return dx

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
        self.weight = Parameter(num_parameters, self._use_gpu)

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

    def forward_cpu(self, x: np.ndarray) -> np.ndarray:
        self.cache = [x]
        exps = np.exp(x - np.max(x))
        return exps / np.sum(exps, axis=self.dim)
    
    def backward_cpu(self, dy: np.ndarray) -> np.ndarray:
        x = self.cache[0]
        s = dy.reshape(-1,1)
        return np.diagflat(s) - np.dot(s, s.T)
