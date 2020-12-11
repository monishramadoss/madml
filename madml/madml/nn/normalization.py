from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

#from typing import List, Optional, Union
from .module import Module, Parameter
from madml import tensor
import madml
import numpy as np

def exp_running_avg(running, new, gamma=.9):
    return gamma * running + (1. - gamma) * new

class _NormBase(Module):
    __constants__ = ['track_running_stats', 'momentum', 'eps', 'num_features', 'affine']
    num_features : int
    eps : float
    momentum : float
    affine : bool
    track_running_stats : bool

    def __init__(self, num_features: int, eps: float=1e-5, momentum: float=0.1, affine: bool=True, track_running_stats: bool=True, backend=None) -> None:
        super(_NormBase, self).__init__(backend)
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            self.weight = Parameter([num_features], self._use_gpu, True)
            self.bias = Parameter([num_features], self._use_gpu, False)
        if self.track_running_stats:
            self.running_mean = Parameter([num_features], self._use_gpu, False)
            self.running_var = Parameter([num_features], self._use_gpu, False)
        self.num_batches_tracked = 0

    def extra_repr(self):
        return '{num_features}, eps={eps}, momentum={momentum}, affine={affine}, track_running_stats={track_running_stats}'.format(**self.__dict__)

class BatchNorm(_NormBase):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super(BatchNorm, self).__init__(num_features, eps, momentum, affine, track_running_stats)

    def forward_cpu(self, x:  np.ndarray) ->  np.ndarray:
        mu = np.mean(x, axis=0)
        var = np.var(x, axis=0)
        x_norm = (x - mu) / np.sqrt(var + self.eps)
        y = self.weight.data * x_norm + self.bias.data

        self.running_mean.data = exp_running_avg(self.running_mean.data, mu, self.momentum)
        self.running_var.data = exp_running_avg(self.running_var.data, var, self.momentum)
        self.cache = [x, x_norm, mu, var]
        return y

    def backward_cpu(self, dy:  np.ndarray) ->  np.ndarray:
        x, x_norm, mu, var = self.cache
        N, C = x.shape
        x_mu = x - mu
        std_inv = 1. / np.sqrt(var + self.eps)
        dx_norm = dy * self.weight.data
        dvar = np.sum(dx_norm * x_mu, axis=0) * -.5 * std_inv ** 3
        dmu = np.sum(dx_norm * -std_inv, axis=0) + dvar * np.mean(-2. * x_mu, axis=0)

        dx = (dx_norm * std_inv) + (dvar * 2 * x_mu / N) + (dmu / N)
        self.weight.gradients = np.sum(dy * x_norm, axis=0)
        self.bias.gradients = np.sum(dy, axis=0)

        return dx

class BatchNorm1d(BatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError('expected 2D or 3D input (got {}D input)'.format(input.dim()))

class BatchNorm2d(BatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'.format(input.dim()))

class BatchNorm3d(BatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError('expected 5D or 3D input (got {}D input)'.format(input.dim()))

class InstanceNorm(_NormBase):
    def __init__(self, num_features: int, eps: float=1e-5, momentum: float=0.1, affine: bool=False, track_running_stats: bool=False) -> None:
        super(InstanceNorm, self).__init__(num_features, eps, momentum, affine, track_running_stats)

    def forward_cpu(self, x:  np.ndarray) ->  np.ndarray:
        mu = np.mean(x, axis=0)
        var = np.var(x, axis=0)
        x_norm = (x - mu) / np.sqrt(var + self.eps)
        y = self.weight * x_norm + self.bias

        self.running_mean.data = exp_running_avg(self.running_mean.data, mu, self.momentum)
        self.running_var.data = exp_running_avg(self.running_var.data, var, self.momentum)
        self.cache = [x, x_norm, mu, var]
        return y

    def backward_cpu(self, dy:  np.ndarray) ->  np.ndarray:
        x, x_norm, mu, var = self.cache
        N, C = x.shape
        x_mu = x - mu

        std_inv = 1. / np.sqrt(var + self.eps)
        dx_norm = dy * self.weight
        dvar = np.reduce_sum(dx_norm * x_mu, axis=0) * -.5 * std_inv ** 3
        dmu = np.reduce_sum(dx_norm * -std_inv, axis=0) + dvar * np.mean(-2. * x_mu, axis=0)

        dx = (dx_norm * std_inv) + (dvar * 2 * x_mu / N) + (dmu / N)
        self.weight.gradients = np.reduce_sum(dy * x_norm, axis=0)
        self.bias.gradients = np.reduce_sum(dy, axis=0)

        return dx

class InstanceNorm1d(InstanceNorm):
    def _check_input_dim(self, input):
        if input.dim() == 2:
            raise ValueError('InstanceNorm1d returns 0-filled tensor to 2D tensor. This is because InstanceNorm1d reshapes inputs to (1, N * C, ...) from (N, C,...) and this makes variances 0.')
        if input.dim() != 3:
            raise ValueError('expected 3D input (got {}D input)'.format(input.dim()))

class InstanceNorm2d(InstanceNorm):
    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'.format(input.dim()))

class InstanceNorm3d(InstanceNorm):
    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'.format(input.dim()))