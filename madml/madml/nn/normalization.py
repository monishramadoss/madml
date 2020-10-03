from .module import Module
import numpy as np

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
            self.weight = np.zeros((num_features))
            self.bias = np.zeros((num_features))
        #else:
        #    self.register_parameter('weight', None)
        #    self.register_parameter('bias', None)
        #if self.track_running_stats:
        #    self.register_buffer('running_mean', np.zeros(num_features))
        #    self.register_buffer('running_var', np.ones(num_features))
        #    self.register_buffer('num_batches_tracked', torch.tensor(0,
        #    dtype=torch.long))
        #else:
        #    self.register_parameter('running_mean', None)
        #    self.register_parameter('running_var', None)
        #    self.register_parameter('num_batches_tracked', None)

    def extra_repr(self):
        return '{num_features}, eps={eps}, momentum={momentum}, affine={affine}, track_running_stats={track_running_stats}'.format(**self.__dict__)

class BatchNorm(_NormBase):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super(BatchNorm, self).__init__(num_features, eps, momentum, affine, track_running_stats)

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

class _InstanceNorm(_NormBase):
    def __init__(self, num_features: int, eps: float=1e-5, momentum: float=0.1, affine: bool=False, track_running_stats: bool=False) -> None:
        super(_InstanceNorm, self).__init__(num_features, eps, momentum, affine, track_running_stats)

class InstanceNorm1d(_InstanceNorm):
    def _check_input_dim(self, input):
        if input.dim() == 2:
            raise ValueError('InstanceNorm1d returns 0-filled tensor to 2D tensor. This is because InstanceNorm1d reshapes inputs to (1, N * C, ...) from (N, C,...) and this makes variances 0.')
        if input.dim() != 3:
            raise ValueError('expected 3D input (got {}D input)'.format(input.dim()))

class InstanceNorm2d(_InstanceNorm):
    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'.format(input.dim()))

class InstanceNorm3d(_InstanceNorm):
    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'.format(input.dim()))