from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

#from typing import List, Optional, Union

import numpy as np
from .module import Module

class RNNBase(Module):
    __constants__ = ['mode', 'input_size', 'hidden_size', 'num_layers', 'bias',
                     'batch_first', 'dropout', 'bidirectional']

    mode : str
    input_size : int
    hidden_size : int
    num_layers : int
    bias : bool
    batch_first : bool
    dropout : float
    bidirectional : bool

    def __init__(self, mode: str, input_size: int, hidden_size: int,
                 num_layers: int=1, bias: bool=True, batch_first: bool=False,
                 dropout: float=0., bidirectional: bool=False) -> None:
        super(RNNBase, self).__init__()
        self.mode = mode
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = float(dropout)
        self.bidirectional = bidirectional
        num_directions = 2 if bidirectional else 1

        if mode == 'LSTM':
            gate_size = 4 * hidden_size
        elif mode == 'GRU':
            gate_size = 3 * hidden_size
        elif mode == 'RNN_TANH':
            gate_size = hidden_size
        elif mode == 'RNN_RELU':
            gate_size = hidden_size
        else:
            raise ValueError("Unrecognized RNN mode: " + mode)

        self._flat_weights_names = []
        self._all_weights = []
        for layer in range(num_layers):
            for direction in range(num_directions):
                layer_input_size = input_size if layer == 0 else hidden_size * num_directions

                w_ih = np.zeros((gate_size, layer_input_size))
                w_hh = np.zeros((gate_size, hidden_size))
                b_ih = np.zeros((gate_size))
                b_hh = np.zeros((gate_size))
                layer_params = (w_ih, w_hh, b_ih, b_hh)

                suffix = '_reverse' if direction == 1 else ''
                param_names = ['weight_ih_l{}{}', 'weight_hh_l{}{}']
                if bias:
                    param_names += ['bias_ih_l{}{}', 'bias_hh_l{}{}']
                param_names = [x.format(layer, suffix) for x in param_names]

                for name, param in zip(param_names, layer_params):
                    setattr(self, name, param)
                self._flat_weights_names.extend(param_names)
                self._all_weights.append(param_names)

class RNN(RNNBase):
    def __init__(self, *args, **kwargs):
        self.nonlinearity = kwargs.pop('nonlinearity', 'tanh')
        if self.nonlinearity == 'tanh':
            mode = 'RNN_TANH'
        elif self.nonlinearity == 'relu':
            mode = 'RNN_RELU'
        else:
            raise ValueError("Unknown nonlinearity '{}'".format(self.nonlinearity))
        super(RNN, self).__init__(mode, *args, **kwargs)

class LSTM(RNNBase):
    def __init__(self, *args, **kwargs):
        super(LSTM, self).__init__('LSTM', *args, **kwargs)

class GRU(RNNBase):
    def __init__(self, *args, **kwargs):
        super(GRU, self).__init__('GRU', *args, **kwargs)