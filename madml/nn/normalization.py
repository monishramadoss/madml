from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from madml import tensor
from madml.init import kaiming_uniform, zeros, ones, xavier_uniform
from .module import Module, Parameter
import vknn

def _dim_fix(arr, arg_arr, pi):
    def parse(x):
        return [x for _ in range(pi)] if isinstance(x, int) else [x[t] for t in range(pi)]
    if isinstance(arg_arr, int):
        arg_arr = parse(arg_arr)
    j = 0
    for i in range(len(arg_arr) - 1, len(arr)):
        arr[i] = arg_arr[j]
        j += 1
    return arr

def exp_running_avg(running, new, gamma=.9):
    return gamma * running + (1. - gamma) * new

class _NormBase(Module):
    __constants__ = ['track_running_stats', 'momentum', 'eps', 'num_features', 'affine']
    num_features : int
    eps : float
    momentum : float
    affine : bool
    track_running_stats : bool

    def __init__(self, num_features: int, eps: float=1e-5, momentum: float=0.1, affine: bool=True, track_running_stats: bool=True) -> None:
        super(_NormBase, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            self.weight = register_weight(zeros, [num_features])
            self.bias = register_bias(True, zeros, [num_features])
        if self.track_running_stats:
            self.running_mean = register_weight(zeros, [num_features])
            self.running_var = register_weight(zeros, [num_features])
            self.num_batches_tracked = 0

#mu = np.mean(X, axis=0)
#var = np.var(X, axis=0)

#X_norm = (X - mu) / np.sqrt(var + 1e-8)
#out = gamma * X_norm + beta

#cache = (X, X_norm, mu, var, gamma, beta)

#running_mean = exp_running_avg(running_mean, mu, momentum)
#running_var = exp_running_avg(running_var, var, momentum)

class BatchNorm(_NormBase):
    def __init__(self, num_features: int, eps: float=1e-5, momentum: float=0.1, affine: bool=True, track_running_stat: bools=True):
        super(BatchNorm, self).__init__(num_features, eps, momentum, affine, track_running_stats)

    def forward_cpu(self, x: tensor) -> tensor:
        if self.y is None:
            self.y = zeros(x.shape)

        mu = np.mean(x.host_data, axis=0)
        var = np.var(x.host_data, axis=0)
        x_norm = (x.host_data - mu) / np.sqrt(var + self.esp)

        self.y.host_data = self.weight.host_data * x_norm + self.bias.host_data
        self.running_mean.host_data = self.momentum * self.running_mean.host_data + (1. - self.momentum) * mu
        self.running_var.host_data = self.momentum * self.running_var.host_data + (1. - self.momentum) * var
        self.num_batches.tracked += 1

        self.register_backward_arg('x', x)
        self.register_backward_arg('dx', x.gradient)
        self.register_backward_arg('dy', self.y.gradient)

        return self.y

    def backward_cpu(self, x: tensor,  dx:tensor, dy: tensor):
        mu = np.mean(x.host_data, axis=0)
        var = np.var(x.host_data, axis=0)
        x_norm = (x.host_data - mu) / np.sqrt(var + self.esp)

        N, C = x.shape[:2]
        x_mu = x.host_data - mu

        std_inv = 1. / np.sqrt(var + self.eps)
        dx_norm = dx.host_data * self.weight.host_data
        dvar = np.sum(dx_norm * x_mu, axis=0) * -.5 * std_inv ** 3
        dmu = np.sum(dx_norm * -std_inv, axis=0) + dvar * np.mean(-2. * x_mu, axis=0)

        dx.host_data = (dx_norm * std_inv) + (dvar * 2 * x_mu / N) + (dmu / N)
        self.weight.gradient.host_data = np.sum(dy.host_data * x_norm, axis=0)
        self.bias.gradient.host_data = np.sum(dy.host_data, axis=0)

        return dx

class BatchNorm1d(BatchNorm):
    def __init__(self, num_features: int, eps: float=1e-5, momentum: float=0.1, affine: bool=True, track_running_stat: bools=True):
        super(BatchNorm1d, self).__init__(num_features, eps, momentum, affine, track_running_stats)

class BatchNorm2d(BatchNorm):
    def __init__(self, num_features: int, eps: float=1e-5, momentum: float=0.1, affine: bool=True, track_running_stat: bools=True):
        super(BatchNorm2d, self).__init__(num_features, eps, momentum, affine, track_running_stats)

class BatchNorm3d(BatchNorm):
    def __init__(self, num_features: int, eps: float=1e-5, momentum: float=0.1, affine: bool=True, track_running_stat: bools=True):
        super(BatchNorm3d, self).__init__(num_features, eps, momentum, affine, track_running_stats)

class InstanceNorm(_NormBase):
    def __init__(self, num_features: int, eps: float=1e-5, momentum: float=0.1, affine: bool=False, track_running_stats: bool=False) -> None:
        super(InstanceNorm, self).__init__(num_features, eps, momentum, affine, track_running_stats)

    def forward_cpu(self, x: tensor) -> tensor:
        if self.y is None:
            self.y = zeros(x.shape)
        return self.y

class InstanceNorm1d(_NormBase):
    def __init__(self, num_features: int, eps: float=1e-5, momentum: float=0.1, affine: bool=False, track_running_stats: bool=False) -> None:
        super(InstanceNorm1d, self).__init__(num_features, eps, momentum, affine, track_running_stats)

class InstanceNorm2d(_NormBase):
    def __init__(self, num_features: int, eps: float=1e-5, momentum: float=0.1, affine: bool=False, track_running_stats: bool=False) -> None:
        super(InstanceNorm2d, self).__init__(num_features, eps, momentum, affine, track_running_stats)

class InstanceNorm3d(_NormBase):
    def __init__(self, num_features: int, eps: float=1e-5, momentum: float=0.1, affine: bool=False, track_running_stats: bool=False) -> None:
        super(InstanceNorm3d, self).__init__(num_features, eps, momentum, affine, track_running_stats)