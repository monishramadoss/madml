from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import defaultdict
from typing import Dict, List

import numpy as np

from madml import tensor
from madml.nn import Parameter

DEBUG = False


def print_p(p: Parameter) -> None:
    print('\tdata', p.param.host_data.max(),
          '\n\tgrad', p.param.gradient.host_data.max())
    for i, x in enumerate(p.optimizer_stuff):
        print('\n\toptmizer_value:', i, x.host_data.max())


def dl1_reg(w: np.ndarray, lam: float = 1e-3, esp: float = 1e-8) -> np.ndarray:
    return w * lam / np.abs(w + esp)


def dl2_reg(w: np.ndarray, lam: float = 1e-3) -> np.ndarray:
    return w * lam


def exp_running_avg(running, new, gamma=.9):
    return gamma * running + (1. - gamma) * new


class Optimizer(object):
    _use_velocity: bool

    def __init__(self, params: List[Parameter], defaults: dict) -> None:
        self.defaults = defaults
        self.state = defaultdict(dict)
        self.params = params
        self._use_velocity = False

    def zero_grad(self) -> None:
        if DEBUG:
            print('parameter#', len(self.params))
        for t in self.params:
            t.zero_grad()

    def step(self):
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, params: List[Parameter], lr: float = 1e-3, momentum: float = 0.0, dampening: int = 0,
                 weight_decay: float = 0, nesterov: bool = False) -> None:
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening, weight_decay=weight_decay, nesterov=nesterov)
        super(SGD, self).__init__(params, defaults)

        if momentum > 0.0:
            for p in self.params:
                p.optimizer_stuff = [tensor([0.0 for _ in range(p.param.size)], p.param.shape, requires_grad=nesterov)]

    def step(self, closure=None) -> None:
        for p in self.params:
            p.param.reset_shape()
            p.param.gradient.host_data += dl2_reg(p.param.host_data, self.defaults['lr'])

            if self.defaults['momentum'] > 0.0:
                v = p.optimizer_stuff[0].host_data
                v = self.defaults['momentum'] * v - self.defaults['lr'] * p.param.gradient.host_data
                p.optimizer_stuff[0].host_data = v
                p.param.host_data += v
            elif self.defaults['nesterov'] and self.defaults['momentum'] >= 0.0:
                v = p.optimizer_stuff[0].host_data
                v = self.defaults['momentum'] * v - self.defaults['lr'] * p.param.gradient.host_data
                p.param.host_data += self.defaults['momentum'] * v - self.defaults['lr'] * p.param.gradient.host_data
            else:
                p.param.host_data -= self.defaults['lr'] * p.param.gradient.host_data

            if DEBUG:
                print_p(p)


class Adam(Optimizer):
    def __init__(self, params: List[Parameter], lr: float = 1e-3, betas: List[float] = (0.9, 0.999),
                 eps: float = 1e-8, weight_decay: float = 0.0, amsgrad: bool = False) -> None:
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)

        self.counter = 1
        super(Adam, self).__init__(params, defaults)

        for p in self.params:
            p.optimizer_stuff = [
                tensor([0.0 for _ in range(p.param.size)], p.param.shape, requires_grad=True),
                tensor([0.0 for _ in range(p.param.size)], p.param.shape, requires_grad=True)
            ]

    def step(self, closure=None) -> None:
        for p in self.params:
            p.param.reset_shape()
            p.param.gradient.host_data += dl2_reg(p.param.host_data, self.defaults['lr'])

            m = p.optimizer_stuff[0].host_data
            r = p.optimizer_stuff[1].host_data

            m = exp_running_avg(m, p.param.gradient.host_data, self.defaults['betas'][0])
            r = exp_running_avg(r, p.param.gradient.host_data ** 2, self.defaults['betas'][1])

            m_k_hat = m / (1. - self.defaults['betas'][0] ** self.counter)

            if self.defaults['amsgrad']:
                r_k_hat = p.optimizer_stuff[1].gradient.host_data
                r_k_hat = np.max(r, r_k_hat)
                p.param.host_data -= self.defaults['lr'] / (np.sqrt(r_k_hat) + self.defaults['eps']) * m

            else:
                r_k_hat = r / (1. - self.defaults['betas'][1] ** self.counter)
                p.param.host_data -= self.defaults['lr'] * m_k_hat / (np.sqrt(r_k_hat) + self.defaults['eps'])

            p.optimizer_stuff[0].host_data = m
            p.optimizer_stuff[1].host_data = r
            p.optimizer_stuff[0].gradient.host_data = m_k_hat
            p.optimizer_stuff[1].gradient.host_data = r_k_hat

            if DEBUG:
                print_p(p)
        self.counter += 1


class Adagrad(Optimizer):
    def __init__(self, params: List[Parameter], lr: float = 1e-2, lr_decay: float = 0.,
                 weight_decay: float = 0, initial_accumulator_value: int = 0, eps: float = 1e-10) -> None:

        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= lr_decay:
            raise ValueError("Invalid lr_decay value: {}".format(lr_decay))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not 0.0 <= initial_accumulator_value:
            raise ValueError("Invalid initial_accumulator_value value: {}".format(initial_accumulator_value))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))

        defaults = dict(lr=lr, lr_decay=lr_decay, eps=eps, weight_decay=weight_decay,
                        initial_accumulator_value=initial_accumulator_value)
        super(Adagrad, self).__init__(params, defaults)

        for p in self.params:
            p.optimizer_stuff = [tensor([0.0 for _ in range(p.param.size)], p.param.shape, requires_grad=False)]

    def step(self, closure=None) -> None:
        for p in self.params:
            p.param.reset_shape()
            p.param.gradient.host_data += dl2_reg(p.param.host_data, self.defaults['lr'])

            v = p.optimizer_stuff[0].host_data
            v = v + p.param.gradient.host_data

            p.param.host_data -= self.defaults['lr'] * (np.sqrt(v + self.defaults['eps'])) * p.param.gradient.host_data

            p.optimizer_stuff[0].host_data = v

            if DEBUG:
                print_p(p)


class RMSprop(Optimizer):
    def __init__(self, params: List[Parameter], lr: float = 1e-2, alpha: float = 0.99, eps: float = 1e-8,
                 weight_decay: float = 0, momentum: int = 0, centered: bool = False) -> None:

        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= momentum:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not 0.0 <= alpha:
            raise ValueError("Invalid alpha value: {}".format(alpha))

        defaults = dict(lr=lr, momentum=momentum, alpha=alpha, eps=eps, centered=centered, weight_decay=weight_decay)

        super(RMSprop, self).__init__(params, defaults)

        for x, p in self.params.items():
            p.optimizer_stuff = [tensor([0.0 for _ in range(p.param.size)], p.param.shape, requires_grad=False)]

    def step(self, closure=None) -> None:
        for k, p in self.params.items():
            p.param.reset_shape()
            p.param.gradient.host_data += dl2_reg(p.param.host_data, self.defaults['lr'])

            v = p.optimizer_stuff[0].host_data
            v = self.defaults['alpha'] * v + (1 - self.defaults['alpha']) * p.param.gradient.host_data ** 2

            p.param.host_data -= self.defaults['lr'] * (np.sqrt(v + self.defaults['eps'])) * p.param.gradient.host_data

            p.optimizer_stuff[0].host_data = v

            if DEBUG:
                print_p(p)


class Nadam(Optimizer):
    def __init__(self, params: Dict[int, Parameter], lr: float = 1e-2, lr_decay: float = 0.,
                 weight_decay: float = 0, initial_accumulator_value: int = 1, eps: float = 1e-10) -> None:

        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= lr_decay:
            raise ValueError("Invalid lr_decay value: {}".format(lr_decay))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not 0.0 <= initial_accumulator_value:
            raise ValueError("Invalid initial_accumulator_value value: {}".format(initial_accumulator_value))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))

        defaults = dict(lr=lr, lr_decay=lr_decay, eps=eps, weight_decay=weight_decay,
                        initial_accumulator_value=initial_accumulator_value)

        self.counter = initial_accumulator_value
        super(Nadam, self).__init__(params, defaults)

        for x, p in self.params.items():
            p.optimizer_stuff = [
                tensor([0.0 for _ in range(p.param.size)], p.param.shape, requires_grad=False),
                tensor([0.0 for _ in range(p.param.size)], p.param.shape, requires_grad=False)
            ]

    def step(self, closure=None) -> None:
        for k, p in self.params.items():
            p.param.reset_shape()
            p.param.gradient.host_data += dl2_reg(p.param.host_data, self.defaults['lr'])

            m = p.optimizer_stuff[0].host_data
            r = p.optimizer_stuff[1].host_data

            m = exp_running_avg(m, p.param.host_data, self.defaults['betas'][0])
            r = exp_running_avg(r, p.param.host_data ** 2, self.defaults['betas'][1])

            m_k_hat = m / (1. - self.defaults['betas'][0] ** self.counter)
            r_k_hat = r / (1. - self.defaults['betas'][1] ** self.counter)

            w = self.defaults['lr'] / (np.sqrt(r_k_hat) + self.defaults['eps'])
            w *= self.defaults['betas'][0] * m_k_hat + (1 - self.defaults['betas'][0]) / (
                    1 - self.defaults['betas'][8] ** self.counter)
            p.param.host_data -= w

            p.optimizer_stuff[0].host_data = m
            p.optimizer_stuff[1].host_data = r

            if DEBUG:
                print_p(p)
