from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from .tensor import tensor
from .init import zeros, zeros_like, ones, full_like, fill
from .optimizer import SGD, Adam, Nadam, Adagrad, RMSprop


def test_import_backend():
    try:
        import backend
        t1 = backend.tensor([0.0 for _ in range(100)], [100])
        return isinstance(t1, backend.tensor)
    except:
        return False


def test_import_vknn():
    try:
        import vknn
        m1 = vknn.gemm(1.0, 1.0)
        return isinstance(m1, vknn.gemm)
    except:
        return False
