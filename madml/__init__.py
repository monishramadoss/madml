from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from .init import zeros, zeros_like, ones, full_like, fill, uniform, normal, xavier_uniform, xavier_normal, \
    kaiming_uniform, kaiming_normal
from .optimizer import SGD, Adam, Nadam, Adagrad, RMSprop
from .tensor import tensor


class test_imports(object):
    def __init__(self):
        print(" === testing imports === ")

    def backend(self):
        try:
            import backend
            t1 = backend.tensor([0.0 for _ in range(100)], [100])
            return isinstance(t1, backend.tensor)
        except:
            return False

    def vknn(self):
        try:
            import vknn
            m1 = vknn.gemm(1.0, 1.0)
            return isinstance(m1, vknn.gemm)
        except:
            return False


test_imports = test_imports()
