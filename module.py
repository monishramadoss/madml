import madml
import madml.nn as nn
import numpy as np


def test_linear():
    a = np.random.ranf([3, 5]).astype(np.float32)
    t1 = madml.tensor(a)
    module = nn.Linear(5, 5, use_gpu=True)

    t2 = module.forward_cpu(t1)
    y = t2.host_data

    t3 = module.forward_gpu(t1)
    y_hat = t3.download()
    
    t1.gradient.host_data = a
    
    print(y_hat == y)

    input()

def test_convolution():
    kernel_shape = [3, 3]
    stride = [1, 1]
    padding = [1, 1]
    dilation = [1, 1]
    x = np.array([[[[0., 1., 2., 3., 4.],
                    [5., 6., 7., 8., 9.],
                    [10., 11., 12., 13., 14.],
                    [15., 16., 17., 18., 19.],
                    [20., 21., 22., 23., 24.]]]]).astype(np.float32)
    y_with_padding = np.array([[[12., 21., 27., 33., 24.],
                                [33., 54., 63., 72., 51.],
                                [63., 99., 108., 117., 81.],
                                [93., 144., 153., 162., 111.],
                                [72., 111., 117., 123., 84.]]]).astype(np.float32).reshape([1, 1, 5, 5])

    t1 = madml.tensor(x)
    
    module = nn.Conv2d(1, 1, kernel_shape, stride, padding, dilation, weight_init='ones')

    t2 = module.forward_cpu(t1)
    y = t2.host_data

    t3 = module.forward_gpu(t1)
    y_hat = t3.download()

    print(y_hat == y)
    input()
    
def test_maxpool():
    kernel_shape = [2, 2]
    stride = [1, 1]
    padding = [0, 0]
    dilation = [1, 1]

    x = np.arange(0, 100).astype(np.float32).reshape([2, 2, 5, 5])

    t1 = madml.tensor(x)
    module = nn.MaxPool2d(kernel_shape, stride, padding, dilation)
    t2 = module.forward_cpu(t1)
    y = t2.host_data
    print(y)
    t3 = module.forward_gpu(t1)
    y_hat = t3.download()
    print()
    print(y_hat)
    input()

def test_relu():
    x = np.random.uniform(-2, 2, size=81).reshape([9, 9])
    t1 = madml.tensor(x)
    module = nn.ReLU()
    t3 = module.forward_gpu(t1)
    y_hat = t3.download()
    print(y_hat)
    print()

    t2 = module.forward_cpu(t1)
    y = t2.host_data
    print(y)
    input()

if __name__ == "__main__":
    test_relu()