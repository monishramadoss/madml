import numpy as np

def _batchnorm_test_mode(x, s, bias, mean, var, epsilon=1e-5):  # type: ignore
    dims_x = len(x.shape)
    dim_ones = (1,) * (dims_x - 2)
    s = s.reshape(-1, *dim_ones)

    print(x)
    bias = bias.reshape(-1, *dim_ones)
    mean = mean.reshape(-1, *dim_ones)
    var = var.reshape(-1, *dim_ones)
    print(x)
    return s * (x - mean) / np.sqrt(var + epsilon) + bias

# input size: (1, 2, 1, 3)
x = np.array([[[[-1, 0, 1]], [[2, 3, 4]]]]).astype(np.float32)
s = np.array([1.0, 1.5]).astype(np.float32)
bias = np.array([0, 1]).astype(np.float32)
mean = np.array([0, 3]).astype(np.float32)
var = np.array([1, 1.5]).astype(np.float32)
y = _batchnorm_test_mode(x, s, bias, mean, var).astype(np.float32)