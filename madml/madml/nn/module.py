from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collection import namedtuple
from typing import List

_modules = list()
_input_output = dict()

Parameter = namedtuple('Parameter', ['gradient', 'weight', 'shape'])

class Module:
    def __init__(self, backend=None):
        self.cache = []
        self.backend = backend
        self._registered = False
        self._parameters = []
        self._use_gpu = False #backend == None

    def forward(self, *args):
        if self.backend is not None and self._use_gpu:
            y =  self.forward_gpu(*args)
        else:
            y = self.forward_cpu(*args)
        
        for x in args:
            _input_output[x] = y

        return y


    def forward_gpu(self, *args):
        if self.backend is not None:
            return self.backend(*args)
        raise NotImplementedError("forward_gpu not implemented")

    def forward_cpu(self, *args):
        raise NotImplementedError("{} forward_cpu for layer not Implemented".format(self))

    def backward(self, *args):
        if self._use_gpu:
            return self.backward_gpu(*args)
        return self.backward_cpu(*args)

    def backward_gpu(self, *args):
        raise NotImplementedError("backward_gpu not implemented")

    def backward_cpu(self, *args):
        raise NotImplementedError("backward_cpu not implemented")

    def param(self):
        return _parameters

    def parameters(self):
        pass

    def __call__(self, *args):
        if not self._registered:
            _modules.append(self)
            self._registered = True
        return self.forward(*args)

    