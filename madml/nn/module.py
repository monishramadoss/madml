from __future__ import absolute_import
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor

from madml import tensor, zeros
import vknn

DEBUG = False

global MODULE_EXECUTOR
MODULE_EXECUTOR = ThreadPoolExecutor(max_workers=4)

insert_dict = lambda dict, name, obj : (dict.update({name: obj}))
def register_arg(executor, dict, name, obj):
    return executor.submit(insert_dict, dict, name, obj)

def run_sys(executor, kernel, *args, **kwargs):
    return executor.submit(kernel, *args, **kwargs)

class Parameter(tensor):
    optimizer_stuff : Optional[List[tensor]]
    shared_devices : bool

    def __init__(self, init_fn, shape: List[int], shared_devices: bool=False, bias: bool=False, dtype=float) -> None:
        super(Parameter, self).__init__(init_fn(shape).host_data, shape, dtype=dtype)
        self.optimizer_stuff = []
        self.shared_devices = shared_devices
        self.bias = bias

    def zero_grad(self,) -> None:
        for i in range(self.size):
            self.grad_data[i] = 0
        self.gradient.upload()

class Module(object):
    def __init__(self, device_id: int=-1):
        self.registered = False
        self.id = id(self)
        self.print_out_flag = False
        self.use_bias = False
        self._empty_gpu_tensor_obj = vknn.tensor([0.], [1])
        self.m_type = type(self).__name__
        self.previous = []
        self.next = []
        self.parameter_cache = []
        self.device_id = device_id

        self.y = None

        self.forward_args = {}        
        self.backward_args = {}

        self.forward_arg_futures = []
        self.backward_arg_futures = []

        self.input_registry = []
        self.output_registry = []
        self.weight_registry = []
        self.kernel_registry = []
        self.module_registry = []

    def forward(self, *args, **kwargs) -> tensor:
        if self.device_id == -1:
            self.y.future = MODULE_EXECUTOR.submit(self.forward_cpu, *args, **kwargs)
            #self.forward_cpu(*args, **kwargs)
        else:
            self.y.future = MODULE_EXECUTOR.submit(self.forward_gpu, *args, **kwargs)
        return self.y

    def backward(self) -> tensor:
        if self.use_bias:
            self.d_bias_call()

        if self.device_id != -1:
            dx = self.backward_gpu(**self.backward_args)
        else:
            dx = self.backward_cpu(**self.backward_args)

        self.forward_arg_futures = []
        self.backward_arg_futures = []

        for x in self.previous:
            x.backward()
        return dx

    def setup(self, *args, **kwargs) -> None:
        pass

    def forward_cpu(self, *args, **kwargs) -> tensor:
        return self._empty_gpu_tensor_obj

    def backward_cpu(self, *args, **kwargs) -> tensor:
        return self._empty_gpu_tensor_obj

    def forward_gpu(self, *args, **kwargs) -> tensor:
        return self._empty_gpu_tensor_obj

    def backward_gpu(self, *args, **kwargs) -> tensor:
        return self._empty_gpu_tensor_obj
     

    def __call__(self, *args, **kwargs) -> tensor:
        for i, x in enumerate(args):
            self.register_input(x, i)    

        self.y = self.forward(*(self.input_registry + self.weight_registry), **kwargs)
       
        if not self.registered:
            self.register_output(self.y, 0)
        self.registered = True                
        return self.y

    def bias_call(self):
        for i in range(self.y.shape[0]):
            self.y.host_data[i] += self.bias.host_data

    def d_bias_call(self):
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
        return  parameters

    def to(self, device_id: int):
        self.device_id = device_id
        for m in self.module_registry:
            m.to(device_id)
        return self

    def register_weight(self, init_fn, shape: List[int], shared_devices: bool=False):
        self.parameter_cache.append(Parameter(init_fn, shape, shared_devices, False))
        if self.parameter_cache[-1] not in self.weight_registry:
            self.weight_registry += tuple([self.parameter_cache[-1]])
        return self.parameter_cache[-1]

    def register_bias(self, bias, init_fn, shape: List[int], shared_devices: bool=False):
        if bias:
            self.parameter_cache.append(Parameter(init_fn, shape, shared_devices, True))
        else:
            self.parameter_cache.append(Parameter(zeros, [1], False, True))
        self.use_bias = bias
        return self.parameter_cache[-1]

    def register_output_shape(self, shape: list) -> tensor:
        if self.y is None:
            self.y = zeros(shape)
        return self.y

    def register_output(self, output: tensor, idx: int):
        if output not in self.output_registry:
            output.previous += [self]
            self.next = [output]                
            if idx >= len(self.output_registry):
                self.output_registry += [input]
            else:
                self.output_registry[idx] = output
            
    def register_input(self, input: tensor, idx: int):
        if input not in self.input_registry:
            input.next += [self]
            self.previous = [input]
            if idx >= len(self.input_registry):
                self.input_registry += [input]
            else: 
                self.input_registry[idx] = input
            
    def register_backward_arg(self, name: str, value: tensor):
        self.backward_arg_futures.append(register_arg(MODULE_EXECUTOR, self.backward_args, name, value))

    def register_forward_arg(self, name: str, value: tensor):
        self.forward_arg_futures.append(register_arg(MODULE_EXECUTOR, self.forward_args, name, value))


    def register_kernel(self, kernel, *args, **kwargs):
        self.kernel_registry += [kernel(*args, **kwargs)]
        return self.kernel_registry[-1]

    def register_module(self, module, *args, **kwargs):
        self.module_registry += [module(*args, **kwargs)]
        return self.module_registry[-1]

    def print_l(self):
        print(type(self), end=': ')
        for t in self.cache:
            if isinstance(t, tensor):
                print(t.shape, end=' ')
        print()