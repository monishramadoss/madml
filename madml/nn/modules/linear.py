import numpy as np
from .module import Module

class Linear(Module):

      def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight =np.zeros((out_features, in_features))
        if bias:
            self.bias = np.zeros(out_features)
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def forward(self, x):
        return self.weight * x  + self.bias