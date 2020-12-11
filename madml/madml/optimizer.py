from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from typing import  Optional, List, Union
from collections import defaultdict
from copy import deepcopy
from itertools import chain
import numpy as np

import math
import madml
from madml.nn import Parameter


class Optimizer(object):
    _use_velocity: bool

    def __init__(self, params: List[Parameter], defaults) -> None:
        self.defaults = defaults
        self.state = defaultdict(dict)
        self.params = params
        self._use_velocity = False

    def zero_grad(self) -> None:
        for _, p in self.params.items():
            p.zero_grad(self._use_velocity)

    def step(self):
        raise NotImplementedError

class Adagrad(Optimizer):
    def __init__(self, params: List[Parameter], lr: float=1e-2, lr_decay: float=0., 
                weight_decay:float=0, initial_accumulator_value: int=0, eps: float=1e-10) -> None:

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

        defaults = dict(lr=lr, lr_decay=lr_decay, eps=eps, weight_decay=weight_decay, initial_accumulator_value=initial_accumulator_value)
        super(Adagrad, self).__init__(params, defaults)

        for param in self.params:
           param

    def share_memory(self):
        pass

    def step(self, closure=None):
        raise NotImplementedError

      

class Adam(Optimizer):
    def __init__(self, params: List[Parameter], lr: float=1e-3, betas: List[float]=(0.9, 0.999), 
                eps: float=1e-8, weight_decay: float=0.0, amsgrad: bool=False) -> None:

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
        super(Adam, self).__init__(params, defaults)

   
    def step(self, closure=None):
        raise NotImplementedError
        return 

class RMSprop(Optimizer):
    def __init__(self, params : List[Parameter], lr: float=1e-2, alpha: float=0.99, eps: float=1e-8, 
                 weight_decay: float=0, momentum: int=0, centered: bool=False) -> None:

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

    def step(self, closure=None):
        loss = None

class SGD(Optimizer):
    def __init__(self, params: List[Parameter], lr: float=1e-2, momentum: int=0.9, dampening: int=0, weight_decay: float=0, nesterov: bool=False) -> None:
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,  weight_decay=weight_decay, nesterov=nesterov)
        super(SGD, self).__init__(params, defaults)
        
    def step(self, closure=None) -> None:
        for x, p in self.params.items():
            p.velocity = self.defaults['momentum'] * p.velocity + self.defaults['lr'] * p.gradient
            p.data -= p.velocity