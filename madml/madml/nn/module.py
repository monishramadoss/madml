from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import random
from typing import List
import numpy as np
import madml

_modules = list()
module_count = 0

def get_modules():
    return _modules if len(_modules) != 0 else []


class Parameter: 
    __constants__ = ['shape']
    shape: List[int]
    _use_gpu: bool

    def __init__(self, shape:List[int], use_gpu: bool) -> None:
        self.shape = shape
        self._use_gpu = use_gpu
        if use_gpu:
            self.gradient = madml.zeros(shape)
            self.data = madml.zeros(shape)
        else:
            self.gradient = np.zeros(shape)
            self.data = np.zeros(shape)

    def init(self, shape: List[int], data: List, gradients: List) -> None:
        self.shape = shape
        self.data = data
        self.gradient = gradients

    def reshape(self, shape: List[int]) -> None:
        self.shape = shape
        self.data.reshape(shape)
        self.gradient.reshape(shape)

    def zero_grad(self):
        if self._use_gpu:
            self.gradient = madml.zeros(self.shape)
        else:
            self.gradient = np.zeros(self.shape)
    
    def update_grad(self, lam: float) -> None:
        if self._use_gpu:
            raise NotImplementedError
        else:
            self.weight -= lam*self.gradient

class Module:
    def __init__(self, backend=None):
        self.cache = []
        self.backend = backend
        self._registered = False
        self._use_gpu = False #backend == None
        self._hash = random.getrandbits(128)
        global module_count
        module_count += 1
      

    def forward(self, *args):
        if self.backend is not None and self._use_gpu:
            y =  self.forward_gpu(*args)
        else:
            y = self.forward_cpu(*args)
        for i in range(len(y)):
            print(self, y[i].shape)
        _modules.append(self)
        return y

    def forward_gpu(self, *args):
        if self.backend is not None:
            return self.backend(*args)
        raise NotImplementedError("forward_gpu not implemented")

    def forward_cpu(self, *args):
        raise NotImplementedError("{} forward_cpu for layer not Implemented".format(self))

    def backward(self, *args):
        x = _modules[-1].backward_hook()
        
        for m in range(len(_modules)-3, 1, -2):
            m = _modules[m]
            if isinstance(m, Module):
                print(m)
                x = m.backward_hook(x)
                print(type(x))



    def backward_hook(self, *args):
        if self._use_gpu:
            return self.backward_gpu(*args)
        return self.backward_cpu(*args)

    def backward_gpu(self, *args):
        raise NotImplementedError("backward_gpu not implemented")

    def backward_cpu(self, *args):
        raise NotImplementedError("backward_cpu not implemented")

    def parameters(self):
        p = []
        for m in _modules:
            if isinstance(m, Module):
                for k, v in m.__dict__.items():
                    if isinstance(v, Parameter):
                        p.append(v)
        return p

    def __call__(self, *args):
        if not self._registered:
            _modules.append(self)
            self._registered = True

        return self.forward(*args)

    