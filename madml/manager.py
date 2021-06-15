from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import inspect
from multiprocessing import Pool
from collections import namedtuple, defaultdict, OrderedDict
from concurrent.futures import ThreadPoolExecutor

global MANAGER_EXECUTOR

managed_modules = namedtuple('managed_modules', ['module', 'id', 'inputs', 'outputs', 'gradients'])
modules_ids = defaultdict(int)
mod_order = []

def get_signature(fn):
    params = inspect.signature(fn).parameters
    args = []
    kwargs = OrderedDict()
    for p in params.values():
        if p.default is p.empty:
            args.append(p.name)
        else:
            kwargs[p.name] = p.default
    return args, kwargs

class Manager(object):
    def __init__(self, num_threads=os.cpu_count() // 2):
        MANAGER_EXECUTOR = ThreadPoolExecutor(max_workers=num_threads)
        self.modules = {}

    def register_module(self, module):
        if module.name not in self.modules:
            _type = type(module).__name__
            modules_ids[_type] += 1
            _id = modules_ids[_type]
            module.name = _type +'_'+ str(_id)
            args, kwargs = get_signature(module.forward)
            inpt_args = OrderedDict([(k, None) for k in args])

            for k, v in kwargs.items():
                inpt_args[k] = v
            grad_inpt_args = OrderedDict([('d'+k, None) for k, _ in inpt_args.items()])


            self.modules[module.name] = {
                "module": module,
                "id": _id,
                "inputs": inpt_args,
                "outputs": OrderedDict(), 
                'gradients': {
                    'inputs': grad_inpt_args,
                    'outputs': OrderedDict()
                }
            }

    def register_input(self, module, args, kwargs):
        mod_order.append(module.name)
        print(args, kwargs)
        mod_dict = self.modules[module.name]
        for k, v in zip(mod_dict['input'].keys(), args):
            self.modules[module.name]['inputs'][k] = v
            self.modules[module.name]['gradients']['inputs']['d'+k] = v.gradient
        for k, v in kwargs.items():
            self.modules[module.name]['inputs'][k] = v

    def register_output(self, module, name, tensor):
        self.modules[module.name]['outputs'][name] = tensor
        self.modules[module.name]['gradients']['outputs']['d'+name] = tensor.gradient

    def forward(self):
        for name in self.modules.keys():
            inpt = self.modules[name]['module']['inputs']
            self.modules[name]['module'].forward(**inpt)

    def backward(self):
        for name in self.modules.keys():
            self.modules[name]['module'].backward()