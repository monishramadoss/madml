from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import random
from typing import List
from collections import OrderedDict
import numpy as np
import madml
import networkx as nx
import matplotlib.pyplot as plt

_tensors = dict()
_modules = OrderedDict()
_parameters = dict()
module_count = 0
graph = nx.DiGraph()
_prev_id = None

def get_modules():
    return _modules if len(_modules) != 0 else []



class Parameter(object): 
    __constants__ = ['shape']
    shape: List[int]
    _use_gpu: bool

    def __init__(self, shape: List[int], use_gpu: bool) -> None:
        self.shape = shape
        self._use_gpu = use_gpu
        _parameters[id(self)] = self
        if use_gpu:
            self.gradient = madml.zeros(shape)
            self.data = madml.zeros(shape)
            self.velocity = madml.zeros(shape)
        else:
            self.gradient = np.zeros(shape)
            self.data = np.zeros(shape)
            self.velocity = np.zeros(shape)

    def init(self, shape: List[int], data: List, gradients: List) -> None:
        self.shape = shape
        self.data = data
        self.gradient = gradients
       

    def reshape(self, shape: List[int]) -> None:
        self.shape = shape
        self.data.reshape(shape)
        self.gradient.reshape(shape)
        

    def zero_grad(self, use_velocity: bool=False) -> None:
        if self._use_gpu:
            self.gradient = madml.zeros(self.shape)
        else:
            self.gradient = np.zeros(self.shape)
        
    def update_grad(self, lam, *args) -> None:
        if self._use_velocity:
            self.weight = lam(self.weight, self.gradient, self.velocity, *args)
        else:
            self.weight = lam(self.weight, self.gradient, *args)
    
class Module(object):
    def __init__(self, backend=None):
        self.cache = []
        self.backend = backend
        self._registered = False
        self._use_gpu = False #backend == None
        self._hash = random.getrandbits(128)
        global module_count
        module_count += 1
        

      
    def _setup(self, X, Y):
        if not self._registered:
            _modules[id(self)] = self
            for y in Y:
                if isinstance(Y, tuple) or isinstance(Y, list):
                    for y in Y:
                        _tensors[id(y)] = id(self)
                else:
                    _tensors[id(Y)] = id(self)
            for x in X:
                if id(x) in _tensors.keys():
                    graph.add_edge(_tensors[id(x)], id(self), )
                else:
                    graph.add_node(id(self))
            self._registered = True

    def forward(self, *args):
        if self.backend is not None and self._use_gpu:
            out =  self.forward_gpu(*args)
        else:
            out = self.forward_cpu(*args)

        self._setup(args, out)
        _last_module = id(self)
        
        print(self, type(out), str(id(self)))
        return out

    def forward_gpu(self, *args):
        if self.backend is not None:
            return self.backend(*args)
        raise NotImplementedError("forward_gpu not implemented")

    def forward_cpu(self, *args):
        raise NotImplementedError("{} forward_cpu for layer not Implemented".format(self))

    def backward(self, weight=None, *args):
        root, t = next(reversed(_modules.items()))
        print("=== Backward call ===", root)
        G1 = nx.relabel_nodes(graph, lambda x: str(type(_modules[x])) + " " +  str(x))
        G2 = nx.relabel_nodes(graph.reverse(), lambda x: str(type(_modules[x])) + " " +  str(x))
        route = list(nx.edge_dfs(graph, source=root, orientation='reverse'))

        plt.subplot(121)       
        nx.draw(G1, pos=nx.spiral_layout(G1), with_labels=True)
        plt.subplot(122)       
        nx.draw(G2, pos=nx.spiral_layout(G2), with_labels=True)
        plt.show()

        grad_owners = {}
        grad_owners[root] = t.backward_hook(*args)
        
        for r in route:
            print(type(_modules[r[1]]), type(_modules[r[0]]))
            dy = grad_owners[r[1]]
            dx = _modules[r[0]].backward_hook(dy)
            grad_owners[r[0]] = dx

        return
        
    def backward_hook(self, *args):
        if self._use_gpu:
            return self.backward_gpu(*args)
        return self.backward_cpu(*args)

    def backward_gpu(self, *args):
        raise NotImplementedError("backward_gpu not implemented")

    def backward_cpu(self, *args):
        raise NotImplementedError("backward_cpu not implemented")

    def parameters(self):
        return _parameters

    def __call__(self, *args):
        return self.forward(*args)

    

# TODO fix branching problems