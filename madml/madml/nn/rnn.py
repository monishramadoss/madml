from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from typing import List
from .module import Module
import madml
import numpy as np
from .activation import Sigmoid, ReLU, Softmax
from .math import tanh


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
            self.gate_size = 4
        elif mode == 'GRU':
            self.gate_size = 3
        elif mode == 'RNN_TANH':
            self.gate_size = 1
        elif mode == 'RNN_RELU':
            self.gate_size = 1
        else:
            raise ValueError("Unrecognized RNN mode: " + mode)
         
        self._flat_weights_names = []
        self._all_weights = []
        for layer in range(num_layers):
            for direction in range(num_directions):
                layer_input_size = input_size if layer == 0 else hidden_size * num_directions
                if self._use_gpu:
                    w_ih = madml.zeros((self.gate_size, hidden_size, layer_input_size))
                    w_hh = madml.zeros((self.gate_size, hidden_size, hidden_size))
                    if self.bias:
                        b_ih = madml.zeros((self.gate_size, hidden_size))
                        b_hh = madml.zeros((self.gate_size, hidden_size))
                        layer_params = (w_ih, w_hh, b_ih, b_hh)
                    else:
                        layer_params = (w_ih, w_hh)

                else:
                    w_ih = np.zeros((self.gate_size, hidden_size, layer_input_size))
                    w_hh = np.zeros((self.gate_size, hidden_size, hidden_size))
                    if self.bias:
                        b_ih = np.zeros((self.gate_size, hidden_size))
                        b_hh = np.zeros((self.gate_size, hidden_size))
                        layer_params = (w_ih, w_hh, b_ih, b_hh)
                    else:
                        layer_params = (w_ih, w_hh)

                suffix = '_reverse' if direction == 1 else ''
                param_names = ['weight_ih_l{}{}', 'weight_hh_l{}{}']
                if bias:
                    param_names += ['bias_ih_l{}{}', 'bias_hh_l{}{}']
                param_names = [x.format(layer, suffix) for x in param_names]

                for name, param in zip(param_names, layer_params):
                    setattr(self, name, param)
                self._flat_weights_names.extend(param_names)
                self._all_weights.append(param_names)

        self._flat_weights = [(lambda wn: getattr(self, wn) if hasattr(self, wn) else None)(wn) for wn in self._flat_weights_names]
        self.activation1 = Sigmoid()
        self.activation2 = tanh() if mode == 'RNN_TANH' else ReLU()
        self.activation3 = Softmax(0)

    def forward_cpu(self, x: np.ndarray, hx: np.ndarray=None, cx: np.ndarray=None) -> List[np.ndarray]:
        max_batch_size = x.shape[0] if self.batch_first else x.shape[1]
        num_directions = 2 if self.bidirectional else 1

        if hx is None:
            hx = np.zeros((self.num_layers * num_directions, max_batch_size, self.hidden_size))
        if cx is None and self.mode == 'LSTM':
            cx = np.zeros((self.num_layers * num_directions, max_batch_size, self.hidden_size))
        y = np.zeros((self.num_layers * num_directions, max_batch_size, self.hidden_size))
        #x_one_hot = np.zeros(self.input_size)
        #x_one_hot[x] = 1.
        #x_one_hot = x_one_hot.reshape(1, -1)
        #x = madml.column_stack((hx, x_one_hot))
        for layer in range(self.num_layers):
            for direction in range(num_directions):
                raise NotImplementedError("RNN NOT DONE")
                if self.bias:
                    hi = getattr(self, self._flat_weights_names[layer * num_directions * 4 + direction * 4])
                    hh = getattr(self, self._flat_weights_names[layer * num_directions * 4 + direction * 4 + 1])
                    bi = getattr(self, self._flat_weights_names[layer * num_directions * 4 + direction * 4 + 2])
                    bh = getattr(self, self._flat_weights_names[layer * num_directions * 4 + direction * 4 + 3])
                    h = hx[layer * num_directions + direction]
                    
                    if self.mode == 'LSTM':
                        c = cx[layer * num_directions + direction]                    
                        hi = self.activation1(x @ hi[0] + bi[0] + hx @ hh[0] + bh[0])
                        hf = self.activation1(x @ hi[1] + bi[1] + hx @ hh[1] + bh[1])
                        ho = self.activation1(x @ hi[2] + bi[2] + hx @ hh[2] + bh[2])
                        hg = self.activation2(x @ hi[3] + bi[3] + hx @ hh[3] + bh[3])

                        c = hf * c + hi * hg
                        h = ho * self.activation2(c)
                        y = h @ x + bi[0]

                    if self.mode == 'GRU':
                       hr = self.activation1(x @ hi[0] + bi[0] + hx @ hh[0] + bh[0])
                       hz = self.activation1(x @ hi[1] + bi[1] + hx @ hh[1] + bh[1])
                       hn = self.activation2(x @ hi[2] + bi[2] + hr * (hx @ hh[2] + bh[2]))
                       h = (1 - hz) * hn + hz * h
                       y = h @ ih + bi[0]

                    if 'RNN' in self.mode:
                        h = self.activation2(x @ hi[0] + hx @ hh[0] + bh[0])        
                        y = h @ x + bi[0]

                else:
                    ih = getattr(self, self._flat_weights_names[layer * num_directions * 2 + direction * 2])
                    hh = getattr(self, self._flat_weights_names[layer * num_directions * 2 + direction * 2 + 1])
                    h = hx[layer * num_directions + direction]

                    if self.mode == 'LSTM':
                        c = cx[layer * num_directions + direction]                    
                        hf = self.activation1(x @ hh[0])
                        hi = self.activation1(x @ hh[1])
                        ho = self.activation1(x @ hh[2])
                        hc = self.activation2(x @ hh[3])

                        c = hf * cx + hi * hc
                        h = h * self.activation2(c)
                        y = h @ x

                    if self.mode == 'GRU':
                        hr = self.activation1(x @ hi[0] + hx @ hh[0])
                        hz = self.activation1(x @ hi[1] + hx @ hh[1])
                        hn = self.activation2(x @ hi[2] + hr * (hx @ hh[2]))
                        h = (1 - hz) * hn + hz * h
                        y = h @ x
                
                    if 'RNN' in self.mode:
                        h = self.activation2(x @ hi[0] + hx @ hh[0])        
                        y = h @ x

        return [y, hx, cx]
  
    def backward_cpu(self, dy: np.ndarray) -> List[np.ndarray]:
        return dy

    def extra_repr(self) -> str:
        s = '{input_size}, {hidden_size}'
        if self.num_layers != 1:
            s += ', num_layers={num_layers}'
        if self.bias is not True:
            s += ', bias={bias}'
        if self.batch_first is not False:
            s += ', batch_first={batch_first}'
        if self.dropout != 0:
            s += ', dropout={dropout}'
        if self.bidirectional is not False:
            s += ', bidirectional={bidirectional}'
        return s.format(**self.__dict__)

class RNN(RNNBase):
    def __init__(self, input_size: int, hidden_size: int,
                 num_layers: int=1, bias: bool=True, batch_first: bool=False,
                 dropout: float=0., bidirectional: bool=False, nonlinearity: str='tanh'):
        self.nonlinearity = nonlinearity
        if self.nonlinearity == 'tanh':
            mode = 'RNN_TANH'
        elif self.nonlinearity == 'relu':
            mode = 'RNN_RELU'
        else:
            raise ValueError("Unknown nonlinearity '{}'".format(self.nonlinearity))
        super(RNN, self).__init__(mode, input_size, hidden_size, num_layers, bias, batch_first, dropout, bidirectional)

class LSTM(RNNBase):
     def __init__(self, input_size: int, hidden_size: int,
                 num_layers: int=1, bias: bool=True, batch_first: bool=False,
                 dropout: float=0., bidirectional: bool=False):
        super(LSTM, self).__init__('LSTM', input_size, hidden_size, num_layers, bias, batch_first, dropout, bidirectional)

class GRU(RNNBase):
    def __init__(self, input_size: int, hidden_size: int,
                 num_layers: int=1, bias: bool=True, batch_first: bool=False,
                 dropout: float=0., bidirectional: bool=False):
        super(GRU, self).__init__('GRU', input_size, hidden_size, num_layers, bias, batch_first, dropout, bidirectional)