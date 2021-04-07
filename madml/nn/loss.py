from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from abc import ABC
from typing import List, Optional

import numpy as np

from madml import tensor
from .module import Module

def _size(shape: List[int]) -> int:
    size = 1
    for s in shape:
        size *= s
    return size

def softmax_util_cpu(x: tensor, y: tensor) -> tensor:
    eX = np.exp((x.host_data.T - np.max(x.host_data, axis=1)).T)
    y.host_data = (eX.T / eX.sum(axis=1)).T
    return y

def l1_reg(w: tensor, lam: float=1e-3) -> float:
    return lam * np.sum(np.abs(w.host_data))

def l2_reg(w: tensor, lam: float=1e-3) -> float:
    return .5 * lam * np.sum(w.host_data * w.host_data)

class _Loss(Module):
    reduction : Optional[str]

    def __init__(self, size_average=None, reduce=None, reduction: str='mean') -> None:
        super(_Loss, self).__init__()
        if size_average is not None or reduce is not None:
            self.reduction = None  # _Reduction.legacy_get_string(size_average, reduce)
        else:
            self.reduction = reduction
        self.loss = tensor([0], [1])
        self.losses = []

    def regularize(self) -> float:
        reg_loss = 0.0
        params = self.parameters()
        for p in params:
            if self.reduction == 'mean' or self.reduction == 'l2':
                reg_loss += l2_reg(p.param)
            elif self.reduction == 'sum' or self.reduction == 'l1':
                reg_loss += l1_reg(p.param)
        return reg_loss

    def print_l(self):
        super(_Loss, self).print_l()
        print('l', self.losses[-1][0].max(), 'r', self.losses[-1][1])
        self.test()

    def test(self):
        pass

    def accuracy(self):
        raise NotImplementedError

class _WeightedLoss(_Loss):
    def __init__(self, weight=None, size_average=None, reduce=None, reduction: str='mean') -> None:
        super(_WeightedLoss, self).__init__(size_average, reduce, reduction)
        self.weight = weight

class crossentropyloss(_WeightedLoss):
    __constants__ = ['ignore_index', 'reduction']
    ignore_index : int

    def __init__(self, weight=None, size_average=None, ignore_index: int=None,
                 reduce=None, reduction: str='mean', with_logit: bool=False) -> None:
        super(crossentropyloss, self).__init__(weight, size_average, reduce, reduction)
        self.ignore_index = ignore_index
        self.with_logit = with_logit
        self.batchsize = 1

    def forward_cpu(self, logit: tensor, target: tensor) -> tensor:
        assert (len(logit.shape) != 1)
        self.batchsize = logit.shape[0]
        N = self.batchsize
        C = logit.shape[1]
        x = logit.host_data
        t = target.host_data
        if not self.with_logit:
            target = target.onehot(label_count=C)
            t = target.host_data

        max_x = np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x - max_x)
        p = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        inp = p

        # gather_weight = None
        # if self.weight is not None:
        #     gather_weight = np.take(self.weight, t, mode='clip')
        #     if self.ignore_index is not None:
        #         gather_weight = np.where(t == self.ignore_index, 0,
        #         gather_weight).astype(dtype=np.float32)
        # elif self.ignore_index is not None:
        #     gather_weight = np.where(t == self.ignore_index, 0,
        #     1).astype(dtype=np.float32)

        # if len(inp.shape) != 3:
        #     inp = inp.reshape([N, C, -1])
        #     t = t.reshape([N, C, -1])
        # D = inp.shape[2]
        #
        # neg_gather_element_input = np.zeros((N, D), dtype=np.float32)
        # for i in range(N):
        #     for d in range(D):
        #         if d != self.ignore_index:
        #             idx = int(t[i][d])
        #             neg_gather_element_input[i][d] = -inp[i][idx][d]

        loss = inp
        if self.reduction == 'mean':
            loss = np.mean(loss)
        elif self.reduction == 'sum':
            loss = np.sum(loss)
        reg = self.regularize()

        self.loss.host_data = (loss + reg) / self.batchsize
        self.cache = [logit, target, p]
        self.losses.append((loss, reg))
        return self.loss

    def backward_cpu(self) -> tensor:
        x, t, p = self.cache
        self.visited[self.loss.id] = True
        self.visited[t.id] = True
        dx = (p - t.host_data) / self.batchsize
        x.gradient.host_data = dx
        return x

    def test(self):
        x, t, p = self.cache
        # _dx = dcross_entropy(x.host_data, t.host_data)
        # assert ((x.gradient.host_data == _dx).all())

    def accuracy(self):
        x, t, p = self.cache
        tmp = np.argmax(x.host_data, axis=1) - np.argmax(t.host_data, axis=1) < 1e-2
        return 1. - np.abs(tmp.mean())

class mseloss(_Loss):
    __constants__ = ['reduction']

    def __init__(self, size_average=None, reduce=None, reduction: str='mean') -> None:
        super(mseloss, self).__init__(size_average, reduce, reduction)

    def forward_cpu(self, logit: tensor, target: tensor) -> tensor:
        if logit.shape != target.shape:
            raise ValueError("logit and target shapes must be the same got: {}, {}".format(logit.shape, target.shape))

        m = logit.shape[0]
        data_loss = 0.5 * (np.square(logit.host_data - target.host_data)).mean(axis=0)

        if self.reduction == 'mean':
            loss = np.mean(data_loss)
        elif self.reduction == 'sum':
            loss = np.sum(data_loss)

        reg = self.regularize()
        self.loss.host_data = loss + reg

        self.cache = [logit, target, m]
        self.losses.append((loss, reg))
        return self.loss

    def backward_cpu(self) -> tensor:
        x, t, m = self.cache
        grad_y = (x.host_data - t.host_data)
        x.gradient.host_data = grad_y
        return x

    def accuracy(self):
        x, t, m = self.cache
        tmp = np.argmax(x.host_data, axis=1) - np.argmax(t.host_data, axis=1) < 1e-2
        return 1. - np.abs(tmp.mean())