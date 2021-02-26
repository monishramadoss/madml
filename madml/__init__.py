from .tensor import tensor
from .init import zeros, zeros_like, ones, full_like, fill, uniform, normal, xavier_uniform, xavier_normal, \
    kaiming_uniform, kaiming_normal
from .optimizer import SGD, Adam, Nadam, Adagrad, RMSprop


try:
    import backend
except ImportError:
    print("=== NO BACKEND DLL LOADED ===")

try:
    import vknn 
except ImportError:
    print("=== NO VKNN DLL LOADED ===")