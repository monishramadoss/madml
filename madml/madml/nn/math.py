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
import numpy as np



class abs(Module):
    __constants__ = ['inplace']
    inplace : bool

    def __init__(self, inplace: bool=False) -> None:
        super(abs, self).__init__(backend.abs(inplace))
        self.inplace = inplace

    def forward_cpu(self, x: np.ndarray) -> np.ndarray:
        self.cache = [x]
        return np.abs(x)

    def backward_cpu(self, dy: np.ndarray) -> np.ndarray:
        x = self.cache[0]
        return x / np.abs(x)

class ceil(Module):
    __constants__ = ['inplace']
    inplace : bool

    def __init__(self, inplace: bool=False) -> None:
        super(ceil, self).__init__(backend.ceil(inplace))
        self.inplace = inplace

    def forward_cpu(self, x: np.ndarray) -> np.ndarray:
        self.cache = [x]
        return np.ceil(x)

    def backward_cpu(self, dy: np.ndarray) -> np.ndarray:
        x = self.cache[0]
        return np.zeros_like(x, dtype=dy.dtype)

class clip(Module):
    __constants__ = ['min_val', 'max_val', 'inplace']
    inplace : bool
    max_val : float
    min_val : float
    def __init__(self, min_val: float, max_val: float, inplace: bool=False) -> None:
        super(clip, self).__init__(backend.clip(min_val, max_val, inplace))
        self.min_val = min_val
        self.max_val = max_val
        self.inplace = inplace

    def forward_cpu(self, x: np.ndarray) -> np.ndarray:
        self.cache = [x]
        return np.clip(x, self.min_val , self.max_val)

    def backward_cpu(self, dy: np.ndarray) -> np.ndarray:
        x = self.cache[0]
        return np.zeros_like(x, dtype=dy.dtype)

class exp(Module):
    __constants__ = ['inplace']
    inplace : bool

    def __init__(self, inplace: bool=False) -> None:
        super(exp, self).__init__(backend.exp(inplace))
        self.inplace = inplace

    def forward_cpu(self, x: np.ndarray) -> np.ndarray:
        self.cache = [x]
        return np.exp(x)

    def backward_cpu(self, dy: np.ndarray) -> np.ndarray:
        x = self.cache[0]
        return np.exp(x)

class floor(Module):
    __constants__ = ['inplace']
    inplace : bool

    def __init__(self, inplace: bool=False) -> None:
        super(floor, self).__init__(backend.floor(inplace))
        self.inplace = inplace

    def forward_cpu(self, x: np.ndarray) -> np.ndarray:
        self.cache = [x]
        return np.floor(x)

    def backward_cpu(self, dy: np.ndarray) -> np.ndarray:
        x = self.cache[0]
        return np.zeros_like(x, x.dtype)

class ln(Module):
    __constants__ = ['inplace']
    inplace : bool

    def __init__(self, inplace: bool=False) -> None:
        super(ln, self).__init__(backend.ln(inplace))
        self.inplace = inplace

    def forward_cpu(self, x: np.ndarray) -> np.ndarray:
        self.cache = [x]
        return np.ln(x)

    def backward_cpu(self, dy: np.ndarray) -> np.ndarray:
        x = self.cache[0]
        return 1 / x

class round(Module):
    __constants__ = ['inplace']
    inplace : bool

    def __init__(self, inplace: bool=False) -> None:
        super(round, self).__init__(backend.round(inplace))
        self.inplace = inplace

    def forward_cpu(self, x: np.ndarray) -> np.ndarray:
        self.cache = [x]
        return np.round(x)

    def backward_cpu(self, dy: np.ndarray) -> np.ndarray:
        x = self.cache[0]
        return np.zeros_like(x, dy.dtype)

class sqrt(Module):
    __constants__ = ['inplace']
    inplace : bool

    def __init__(self, inplace: bool=False) -> None:
        super(sqrt, self).__init__(backend.sqrt(inplace))
        self.inplace = inplace

    def forward_cpu(self, x: np.ndarray) -> np.ndarray:
        self.cache = [x]
        return np.sqrt(x)

    def backward_cpu(self, dy: np.ndarray) -> np.ndarray:
        x = self.cache[0]
        return 0.5 / np.sqrt(x)

class acos(Module):
    __constants__ = ['inplace']
    inplace : bool

    def __init__(self, inplace: bool=False) -> None:
        super(acos, self).__init__(backend.acos(inplace))
        self.inplace = inplace

    def forward_cpu(self, x: np.ndarray) -> np.ndarray:
        self.cache = [x]
        return np.arccos(x)

    def backward_cpu(self, dy: np.ndarray) -> np.ndarray:
        x = self.cache[0]
        return -1. / np.sqrt(1. - x ** 2)

class acosh(Module):
    __constants__ = ['inplace']
    inplace : bool

    def __init__(self, inplace: bool=False) -> None:
        super(acosh, self).__init__(backend.acosh(inplace))
        self.inplace = inplace

    def forward_cpu(self, x: np.ndarray) -> np.ndarray:
        self.cache = [x]
        return np.arccosh(x)

    def backward_cpu(self, dy: np.ndarray) -> np.ndarray:
        x = self.cache[0]
        return -1. / np.sqrt(x ** 2 - 1.)

class asin(Module):
    __constants__ = ['inplace']
    inplace : bool

    def __init__(self, inplace: bool=False) -> None:
        super(asin, self).__init__(backend.asin(inplace))
        self.inplace = inplace

    def forward_cpu(self, x: np.ndarray) -> np.ndarray:
        self.cache = [x]
        return np.arcsin(x)

    def backward_cpu(self, dy: np.ndarray) -> np.ndarray:
        x = self.cache[0]
        return 1. / np.sqrt(1 - x ** 2)

class asinh(Module):
    __constants__ = ['inplace']
    inplace : bool

    def __init__(self, inplace: bool=False) -> None:
        super(asinh, self).__init__(backend.asinh(inplace))
        self.inplace = inplace

    def forward_cpu(self, x: np.ndarray) -> np.ndarray:
        self.cache = [x]
        return np.arcsinh(x)

    def backward_cpu(self, dy: np.ndarray) -> np.ndarray:
        x = self.cache[0]
        return -1. / np.sqrt(x ** 2 + 1)

class atan(Module):
    __constants__ = ['inplace']
    inplace : bool

    def __init__(self, inplace: bool=False) -> None:
        super(atan, self).__init__(backend.atan(inplace))
        self.inplace = inplace

    def forward_cpu(self, x: np.ndarray) -> np.ndarray:
        self.cache = [x]
        return np.arctan(x)

    def backward_cpu(self, dy: np.ndarray) -> np.ndarray:
        x = self.cache[0]
        return -1. / np.sqrt(1 + x ** 2)

class atanh(Module):
    __constants__ = ['inplace']
    inplace : bool

    def __init__(self, inplace: bool=False) -> None:
        super(atanh, self).__init__(backend.atanh(inplace))
        self.inplace = inplace

    def forward_cpu(self, x: np.ndarray) -> np.ndarray:
        self.cache = [x]
        return np.arctanh(x)

    def backward_cpu(self, dy: np.ndarray) -> np.ndarray:
        x = self.cache[0]
        return -1. / (1 - x ** 2)

class cos(Module):
    __constants__ = ['inplace']
    inplace : bool

    def __init__(self, inplace: bool=False) -> None:
        super(cos, self).__init__(backend.cos(inplace))
        self.inplace = inplace

    def forward_cpu(self, x: np.ndarray) -> np.ndarray:
        self.cache = [x]
        return np.cos(x)

    def backward_cpu(self, dy: np.ndarray) -> np.ndarray:
        x = self.cache[0]
        return -np.sin(x)

class cosh(Module):
    __constants__ = ['inplace']
    inplace : bool

    def __init__(self, inplace: bool=False) -> None:
        super(cosh, self).__init__(backend.cosh(inplace))
        self.inplace = inplace

    def forward_cpu(self, x: np.ndarray) -> np.ndarray:
        self.cache = [x]
        return np.cosh(x)

    def backward_cpu(self, dy: np.ndarray) -> np.ndarray:
        x = self.cache[0]
        return np.sinh(x)

class sin(Module):
    __constants__ = ['inplace']
    inplace : bool

    def __init__(self, inplace: bool=False) -> None:
        super(sin, self).__init__(backend.sin(inplace))
        self.inplace = inplace

    def forward_cpu(self, x: np.ndarray) -> np.ndarray:
        self.cache = [x]
        return np.sin(x)

    def backward_cpu(self, dy: np.ndarray) -> np.ndarray:
        x = self.cache[0]
        return np.cos(x)

class sinh(Module):
    __constants__ = ['inplace']
    inplace : bool

    def __init__(self, inplace: bool=False) -> None:
        super(sinh, self).__init__(backend.sinh(inplace))
        self.inplace = inplace

    def forward_cpu(self, x: np.ndarray) -> np.ndarray:
        self.cache = [x]
        return np.sinh(x)

    def backward_cpu(self, dy: np.ndarray) -> np.ndarray:
        x = self.cache[0]
        return np.cosh(x)

class tan(Module):
    __constants__ = ['inplace']
    inplace : bool

    def __init__(self, inplace: bool=False) -> None:
        super(tan, self).__init__(backend.tan(inplace))
        self.inplace = inplace

    def forward_cpu(self, x: np.ndarray) -> np.ndarray:
        self.cache = [x]
        return np.tan(x)

    def backward_cpu(self, dy: np.ndarray) -> np.ndarray:
        x = self.cache[0]
        return (1 / np.cos(x)) ** 2

class tanh(Module):
    __constants__ = ['inplace']
    inplace : bool

    def __init__(self, inplace: bool=False) -> None:
        super(tanh, self).__init__(backend.tanh(inplace))
        self.inplace = inplace

    def forward_cpu(self, x: np.ndarray) -> np.ndarray:
        self.cache = [x]
        return np.tanh(x)

    def backward_cpu(self, dy: np.ndarray) -> np.ndarray:
        x = self.cache[0]
        return (1 / np.cosh(x)) ** 2

class add(Module):
    __constants__ = ['inplace']
    inplace : bool

    def __init__(self, inplace: bool=False) -> None:
        super(add, self).__init__(backend.add(inplace))
        self.inplace = inplace

    def forward_cpu(self, x: np.ndarray, w: np.ndarray) -> np.ndarray:
        self.cache = [x, w]
        return x + w

    def backward_cpu(self, dy: np.ndarray) -> np.ndarray:
        x, w = self.cache
        return np.ones_like(dy), np.ones_like(dy)

class sub(Module):
    __constants__ = ['inplace']
    inplace : bool

    def __init__(self, inplace: bool=False) -> None:
        super(sub, self).__init__(backend.sub(inplace))
        self.inplace = inplace

    def forward_cpu(self, x: np.ndarray, w: np.ndarray) -> np.ndarray:
        self.cache = [x, w]
        return x - w

    def backward_cpu(self, dy: np.ndarray) -> np.ndarray:
        x, w = self.cache
        return np.ones_like(dy), -1. * np.ones_like(dy)

class mul(Module):
    __constants__ = ['inplace']
    inplace : bool

    def __init__(self, inplace: bool=False) -> None:
        super(mul, self).__init__(backend.mul(inplace))
        self.inplace = inplace

    def forward_cpu(self, x: np.ndarray, w: np.ndarray) -> np.ndarray:
        self.cache = [x, w]
        return np.multiply(x,  w)

    def backward_cpu(self, dy: np.ndarray) -> np.ndarray:
        x, w = self.cache
        return w,  x

class div(Module):
    __constants__ = ['inplace']
    inplace : bool

    def __init__(self, inplace: bool=False) -> None:
        super(div, self).__init__(backend.div(inplace))
        self.inplace = inplace

    def forward_cpu(self, x: np.ndarray, w: np.ndarray) -> np.ndarray:
        self.cache = [x, w]
        return np.divide(x, w)

    def backward_cpu(self, dy: np.ndarray) -> np.ndarray:
        x, w = self.cache
        return w, np.multiply(x, -(1 / (w) ** 2))

class mod(Module):
    __constants__ = ['inplace']
    inplace : bool

    def __init__(self, inplace: bool=False) -> None:
        super(mod, self).__init__(backend.mod(inplace))
        self.inplace = inplace

    def forward_cpu(self, x: np.ndarray, w: np.ndarray) -> np.ndarray:
        self.cache = [x, w]
        return np.mod(x, w)

    def backward_cpu(self, dy: np.ndarray) -> np.ndarray:
        x, w = self.cache
        return np.ones_like(dy), np.ones_like(dy)

class pow(Module):
    __constants__ = ['inplace']
    inplace : bool

    def __init__(self, inplace: bool=False) -> None:
        super(pow, self).__init__(backend.pow(inplace))
        self.inplace = inplace

    def forward_cpu(self, x: np.ndarray, w: np.ndarray) -> np.ndarray:
        y = np.power(x, w)
        self.cache = [x, w, y]
        return y

    def backward_cpu(self, dy: np.ndarray) -> np.ndarray:
        x, w, y = self.cache
        return np.multiply(w, np.power(x, w - 1)),  (np.multiply(y, np.ln(x)))

class max(Module):
    __constants__ = ['inplace']
    inplace : bool

    def __init__(self, inplace: bool=False) -> None:
        super(max, self).__init__(backend.max(inplace))
        self.inplace = inplace

    def forward_cpu(self, x: np.ndarray, w: np.ndarray) -> np.ndarray:
        self.cache = [x, w]
        return np.max(x, w)

    def backward_cpu(self, dy: np.ndarray) -> np.ndarray:
        x, w = self.cache
        return np.ones_like(dy), np.ones_like(dy)

class min(Module):
    __constants__ = ['inplace']
    inplace : bool

    def __init__(self, inplace: bool=False) -> None:
        super(min, self).__init__(backend.min(inplace))
        self.inplace = inplace

    def forward_cpu(self, x: np.ndarray, w: np.ndarray) -> np.ndarray:
        self.cache = [x, w]
        return np.min(x, w)

    def backward_cpu(self, dy: np.ndarray) -> np.ndarray:
        x, w = self.cache
        return np.ones_like(dy), np.ones_like(dy)

class eq(Module):
    __constants__ = ['inplace']
    inplace : bool

    def __init__(self, inplace: bool=False) -> None:
        super(eq, self).__init__(backend.eq(inplace))
        self.inplace = inplace

    def forward_cpu(self, x: np.ndarray, w: np.ndarray) -> np.ndarray:
        self.cache = [x, w]
        return x == w

    def backward_cpu(self, dy: np.ndarray) -> np.ndarray:
        x, w = self.cache
        return np.ones_like(dy), np.ones_like(dy)

class ne(Module):
    __constants__ = ['inplace']
    inplace : bool

    def __init__(self, inplace: bool=False) -> None:
        super(ne, self).__init__(backend.ne(inplace))
        self.inplace = inplace

    def forward_cpu(self, x: np.ndarray, w: np.ndarray) -> np.ndarray:
        self.cache = [x, w]
        return x != w

    def backward_cpu(self, dy: np.ndarray) -> np.ndarray:
        x, w = self.cache
        return np.ones_like(dy), np.ones_like(dy)

class lt(Module):
    __constants__ = ['inplace']
    inplace : bool

    def __init__(self, inplace: bool=False) -> None:
        super(lt, self).__init__(backend.lt(inplace))
        self.inplace = inplace

    def forward_cpu(self, x: np.ndarray, w: np.ndarray) -> np.ndarray:
        self.cache = [x, w]
        return x < w

    def backward_cpu(self, dy: np.ndarray) -> np.ndarray:
        x, w = self.cache
        return np.ones_like(dy), np.ones_like(dy)

class le(Module):
    __constants__ = ['inplace']
    inplace : bool

    def __init__(self, inplace: bool=False) -> None:
        super(le, self).__init__(backend.le(inplace))
        self.inplace = inplace

    def forward_cpu(self, x: np.ndarray, w: np.ndarray) -> np.ndarray:
        self.cache = [x, w]
        return x <= w

    def backward_cpu(self, dy: np.ndarray) -> np.ndarray:
        x, w = self.cache
        return np.ones_like(dy), np.ones_like(dy)

class gt(Module):
    __constants__ = ['inplace']
    inplace : bool

    def __init__(self, inplace: bool=False) -> None:
        super(gt, self).__init__(backend.gt(inplace))
        self.inplace = inplace

    def forward_cpu(self, x: np.ndarray, w: np.ndarray) -> np.ndarray:
        self.cache = [x, w]
        return x > w

    def backward_cpu(self, dy: np.ndarray) -> np.ndarray:
        x, w = self.cache
        return np.ones_like(dy), np.ones_like(dy)

class ge(Module):
    __constants__ = ['inplace']
    inplace : bool

    def __init__(self, inplace: bool=False) -> None:
        super(ge, self).__init__(backend.ge(inplace))
        self.inplace = inplace

    def forward_cpu(self, x: np.ndarray, w: np.ndarray) -> np.ndarray:
        self.cache = [x, w]
        return x >= w

    def backward_cpu(self, dy: np.ndarray) -> np.ndarray:
        x, w = self.cache
        return np.ones_like(dy), np.ones_like(dy)

class xr(Module):
    __constants__ = ['inplace']
    inplace : bool

    def __init__(self, inplace: bool=False) -> None:
        super(xr, self).__init__(backend.xr(inplace))
        self.inplace = inplace

    def forward_cpu(self, x: np.ndarray, w: np.ndarray) -> np.ndarray:
        self.cache = [x, w]
        return np.logical_xor(x, w)

    def backward_cpu(self, dy: np.ndarray) -> np.ndarray:
        x, w = self.cache
        return np.ones_like(dy), np.ones_like(dy)