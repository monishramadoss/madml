from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from typing import List

class Module:

    def __init__(self, backend=None):
        self.cache = []
        self.backend = backend
    def forward(self, *args):
        return self.forward_cpu(*args)

    def forward_cpu(self, args):
        raise NotImplementedError("{} forward_cpu for layer not Implemented".format(self))

    def __call__(self, *args):
        return self.forward(*args)