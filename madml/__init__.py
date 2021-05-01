from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from .tensor import tensor
from .init import zeros, zeros_like, ones, full_like, fill
from .optimizer import SGD, adam, Adagrad, RMSprop

def test_import_vknn():
    try:
        import vknn
        m1 = vknn.gemm(1.0, 1.0, False)
        return isinstance(m1, vknn.gemm)
    except Exception as e:
        print(e)
        return False

def test_pipeline():
    try:
        import vknn

        m1 = vknn.gemm(1.0, 1.0, False, False, False)
        x = vknn.tensor([float(1) for _ in range(100)], [10, 10])
        w = vknn.tensor([float(1) for _ in range(100)], [10, 10])
        b = vknn.tensor([0], [1])
        y = vknn.tensor([float(0) for _ in range(100)], [10, 10])

        m1.forward(y, x, w, b)

        print("Pipeline Testing Done...")

        input()
    except Exception as e:
        print(e)
        return False

def print_graph_next(obj):
    if isinstance(obj, tensor):
        print(type(obj),obj.m_type, obj.shape)
    else:
        print(type(obj),obj.m_type)

    next = obj.next
    if next == []:
        return obj
    for ne in obj.next:
        print_graph_next(ne)
    print()


def print_graph_prev(obj):
    if isinstance(obj, tensor):
        print(type(obj),obj.m_type, obj.shape)
    else:
        print(type(obj),obj.m_type)

    previous = obj.previous
    if previous == []:
        return obj
    for prev in obj.previous:
        print_graph_prev(prev)
    print()

   

def transpose(x: tensor, axis:list):
    layer = nn.transpose(axis)
    return layer(x)


def flatten(x: tensor):
    layer = nn.flatten()
    return layer(x)