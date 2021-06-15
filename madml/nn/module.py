from __future__ import absolute_import
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

import numpy as np

import vknn
from madml import tensor, zeros
from ..manager import Manager

DEBUG = False

global MODULE_EXECUTOR
MODULE_EXECUTOR = ThreadPoolExecutor(max_workers=4)

insert_dict = lambda d, name, obj: (d.update({name: obj}))

def register_arg(executor, d, name, obj):
    return executor.submit(insert_dict, d, name, obj)

def run_sys(executor, kernel, *args, **kwargs):
    return executor.submit(kernel, *args, **kwargs)


manager = Manager()



class Parameter(tensor):
    optimizer_stuff: Optional[List[tensor]]
    shared_devices: bool

    def __init__(self, init_fn, shape: List[int], shared_devices: bool = False, bias: bool = False,
                 dtype=float) -> None:
        super(Parameter, self).__init__(init_fn(shape).host_data, shape, dtype=dtype)
        self.optimizer_stuff = []
        self.shared_devices = shared_devices
        self.bias = bias

    def zero_grad(self, ) -> None:
        for i in range(self.size):
            self.grad_data[i] = 0
        self.gradient.upload()

class Module(object):
    def __init__(self, device_id: int = -1):
        self.registered = False
        self.id = id(self)
        self.print_out_flag = False
        self.use_bias = False
        self._empty_gpu_tensor_obj = vknn.tensor([0.], [1])
        self.m_type = type(self).__name__
        self.name = ''
        self.parameter_cache = []
        self.device_id = device_id

        self.outputs = {}
        self.y = None

        self.w_registry = []
        self.kernel_registry = []
        self.module_registry = []
        manager.register_module(self)        
          
    def forward(self, *args, **kwargs) -> tensor:
        manager.register_input(self, args, kwargs)
        manager.register_output(self, 'y', self.y)
        return self.y

    def backward(self) -> tensor: # try and traverse the graph
        
        return self._empty_gpu_tensor_obj

    def _forward_cpu(self, *args, **kwargs) -> tensor:
        return self._empty_gpu_tensor_obj

    def _backward_cpu(self) -> tensor:
        return self._empty_gpu_tensor_obj

    def _forward_gpu(self, *args, **kwargs) -> tensor:
        return self._empty_gpu_tensor_obj

    def _backward_gpu(self) -> tensor:
        return self._empty_gpu_tensor_obj

    def __call__(self, *args, **kwargs) -> tensor:       
        return self.forward(*args, **kwargs)

    def bias_call(self):
        for i in range(self.y.shape[0]):
            self.y.host_data[i] += self.bias.host_data

    def d_bias_call(self):
        dy = self.y.gradient
        self.bias.gradient.host_data = np.sum(dy.host_data, axis=0)

    def parameters(self) -> List[Parameter]:
        parameters = self.parameter_cache
        for k, v in self.__dict__.items():
            if isinstance(v, Module):
                parameters += v.parameters()
            elif isinstance(v, list) or isinstance(v, tuple):
                for x in v:
                    if isinstance(x, Module):
                        parameters += x.parameters()
        return parameters

    def to(self, device_id: int):
        self.device_id = device_id
        for m in self.module_registry:
            m.to(device_id)
        return self

    def register_weight(self, init_fn, shape: List[int], shared_devices: bool = False):
        self.parameter_cache.append(Parameter(init_fn, shape, shared_devices, False))
        if self.parameter_cache[-1] not in self.w_registry:
            self.w_registry += tuple([self.parameter_cache[-1]])
        return self.parameter_cache[-1]

    def register_bias(self, bias, init_fn, shape: List[int], shared_devices: bool = False):
        if bias:
            self.parameter_cache.append(Parameter(init_fn, shape, shared_devices, True))
        else:
            self.parameter_cache.append(Parameter(zeros, [1], False, True))
        self.use_bias = bias
        return self.parameter_cache[-1]

    def register_output_shape(self, shape: list, name: str='y') -> tensor:
        self.outputs[name] = zeros(shape)
        return self.outputs[name]

    def register_kernel(self, kernel, *args, **kwargs):
        self.kernel_registry += [kernel(*args, **kwargs)]
        return self.kernel_registry[-1]

    def register_module(self, module, *args, **kwargs):
        self.module_registry += [module(*args, **kwargs)]
        self.module_registry[-1].to(self.device_id)
        return self.module_registry[-1]

    def print_l(self):
        print(type(self), end=': ')