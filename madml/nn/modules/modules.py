from collections import OrderedDict, namedtuple


class Parameter:
    def __new__ (self, data=None):
        if data is None:
            pass

class Module(object):
    dump_patches = False

    def __init__ (self):
        self.training = True
        self._parameters = OrderedDict()
        self._buffers = OrderedDict()
        self._persistent_buffers_set = set()
        self._backward_hooks = OrderedDict()
        self._forward_hooks = OrderedDict()
        self._forward_pre_hooks = OrderedDict()
        self._state_dict_hooks = OrderedDict()
        self._load_state_dict_pre_hooks = OrderedDict()
        self._modules = OrderedDict()

    def forward(self, *x):
        raise NotImplementedError

    def __call__(self, x):
        pass