from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from typing import List

_modules = list()

class Module:
    def __init__(self, backend=None):
        self.cache = []
        self.backend = backend
        self._parameters = {}
        self._use_gpu = False #backend == None

    def forward(self, *args):
        if self.backend is not None and self._use_gpu:
            return self.forward_gpu(*args)
        return self.forward_cpu(*args)

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

    def parameters(self):
        return self._parameters

    def __call__(self, *args):
        return self.forward(*args)

    def register_module(self):
        _modules.append(self)