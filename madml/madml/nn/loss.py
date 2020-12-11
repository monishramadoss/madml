from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from typing import  Optional, Union
import madml
from madml import tensor
from .module import Module
from .activation import Softmax
from .. import regularization as reg
import numpy as np

def Regularization(model, reg_type='l2', lam=1e-3):
    reg_types = dict(l1=reg.l1_reg,  l2=reg.l2_reg)

    if reg_type not in reg_types.keys():
        raise Exception('Regularization type must be either "l1" or "l2"!')

    reg_loss = np.sum([
        reg_types[reg_type](model[k], lam)
        for k in model.keys()
        if k.startswith('W')
    ])

    return reg_loss

class _Loss(Module):
    reduction : str
    def __init__(self, size_average=None, reduce=None, reduction: str='mean', backend=None) -> None:
        super(_Loss, self).__init__(backend)
        if size_average is not None or reduce is not None:
            self.reduction = None #_Reduction.legacy_get_string(size_average, reduce)
        else:
            self.reduction = reduction
    
   
class _WeightedLoss(_Loss):
    def __init__(self, weight=None, size_average=None, reduce=None, reduction: str='mean') -> None:
        super(_WeightedLoss, self).__init__(size_average, reduce, reduction)
        self.weight = weight

class L1Loss(_Loss):
    __constants__ = ['reduction']

    def __init__(self, size_average=None, reduce=None, reduction: str='mean') -> None:
        super(L1Loss, self).__init__(size_average, reduce, reduction)

    def forward_cpu(self, logit: np.ndarray, target: np.ndarray) -> np.ndarray:
        m = logit.shape[0]
        data_loss = np.sum(np.abs(target - logit)) / m
        self.cache = [logit, target, m]
        return data_loss

    def backward_cpu(self) -> np.ndarray:
        logit, target, m = self.cache
        grad_y = np.sign(logit - target.reshape(-1, 1))
        grad_y /= m
        return grad_y

class L2Loss(_Loss):
    __constants__ = ['reduction']

    def __init__(self, size_average=None, reduce=None, reduction: str='mean') -> None:
        super(L2Loss, self).__init__(size_average, reduce, reduction)

    def forward_cpu(self, logit: np.ndarray, target: np.ndarray) -> np.ndarray:
        m = logit.shape[0]
        data_loss = 0.5 * np.sum((target - logit) ** 2) / m
        self.cache = [logit, target, m]
        return data_loss

    def backward_cpu(self) -> np.ndarray:
        logit, target, m = self.cache
        grad_y = logit - target.reshape(-1, 1)
        grad_y /= m
        return grad_y

class NLLLoss(_WeightedLoss):
    __constants__ = ['ignore_index', 'reduction']
    ignore_index : int

    def __init__(self, weight: Optional[np.ndarray]=None, size_average=None, ignore_index: int=-100,
                 reduce=None, reduction: str='mean') -> None:
        super(NLLLoss, self).__init__(weight, size_average, reduce, reduction)
        self.ignore_index = ignore_index

    def forward_cpu(self, logit: np.ndarray, target: np.ndarray) -> np.ndarray:
        return None

class MSELoss(_Loss):
    __constants__ = ['reduction']

    def __init__(self, size_average=None, reduce=None, reduction: str='mean') -> None:
        super(MSELoss, self).__init__(size_average, reduce, reduction)

    def forward_cpu(self, logit: np.ndarray, target: np.ndarray) -> np.ndarray:
        m = logit.shape[0]
        data_loss = (np.square(logit - target)).mean(axis=0)
        data_loss /= m
        self.cache = [logit, target, m]
        return data_loss
    def backward_cpu(self) -> np.ndarray:
        logit, target, m = self.cache
        grad_y = 2 * (logit - target)
        grad_y /= m
        return grad_y

class HingeLoss(_Loss):
    __constants__ = ['reduction', 'delta']
    delta : float

    def __init__(self, delta=1, size_average=None, ignore_index: int=-100, reduce=None, reduction: str='mean') -> None:
        super(HingeLoss, self).__init__(size_average, reduce, reduction)
        self.ignore_index = ignore_index
        self.delta = delta

    def forward_cpu(self, logit: np.ndarray, target: np.ndarray) -> np.ndarray:
        m = logit.shape[0]

        margin = (logit.T - logit[range(m), target]).T
        margins = margin = self.delta
        margins[margins < 0] = 0
        margins[range(m), target] = 0
        data_loss = np.sum(margins) / m
        #reg_loss = regularization(model, reg_type='l2', lam=lam)
        self.cache = [logit, target, m, margin]
        return data_loss #+ reg_loss

    def backward_cpu(self) -> np.ndarray:
        logit, target, m, margins = self.cache
        margins[range(m), target] = 0
        grad_y = (margins > 0).astype(float)
        grad_y[range(m), target] = -np.sum(grad_y, axis=1)
        grad_y /= m
        return grad_y

class BCELoss(_WeightedLoss):
    __constants__ = ['reduction']

    def __init__(self, weight: Optional[np.ndarray]=None, size_average=None, reduce=None, reduction: str='mean') -> None:
        super(BCELoss, self).__init__(weight, size_average, reduce, reduction)

    def forward_cpu(self, logit: np.ndarray, target: np.ndarray) -> np.ndarray:

        return

class CrossEntropyLoss(_WeightedLoss):
    __constants__ = ['ignore_index', 'reduction']
    ignore_index : int

    def __init__(self, weight: Optional[np.ndarray]=None, size_average=None, ignore_index: int=-100,
                 reduce=None, reduction: str='mean') -> None:
        super(CrossEntropyLoss, self).__init__(weight, size_average, reduce, reduction)
        self.ignore_index = ignore_index
        
    def forward_cpu(self, logit: np.ndarray, target: np.ndarray) -> np.ndarray:
        m = logit.shape[0]
        exps = np.exp(logit - np.max(logit))
        prob = exps / np.sum(exps, axis=0)
        log_like = -np.log(prob[range(m), target])
        data_loss = np.sum(log_like) / m
        #reg_loss = regularization(model, reg_type='l2', lam=1e-3)
        self.cache = [logit, target, prob, m]
        return np.array([data_loss])  #+ reg_loss

    def backward_cpu(self) -> np.ndarray:
        logit, target, grad_y, m = self.cache
        grad_y[range(m), target] -= 1.
        grad_y /= m
        print(grad_y.shape)
        return grad_y