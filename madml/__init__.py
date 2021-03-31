from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from .tensor import tensor
from .init import zeros, zeros_like, ones, full_like, fill
from .optimizer import SGD, Adam, Nadam, Adagrad, RMSprop

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
     
           
        
      
        m1 = vknn.gemm(1.0, 1.0, False)
        x = vknn.tensor([float(1) for _ in range(100)], [10, 10])
        w = vknn.tensor([float(1) for _ in range(100)], [10, 10])
        b = vknn.tensor([0], [1])
        y = vknn.tensor([float(0) for _ in range(100)], [10, 10])

        m1.forward(y, x, w, b)       

        print(":::::::")

        input()
    except Exception as e:
        print(e)
        return False