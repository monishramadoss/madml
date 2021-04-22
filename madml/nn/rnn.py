from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from madml import tensor
from madml import zeros
from .module import Module, Parameter
from .activation import  relu, softmax
from .math import sigmoid, tanh, add
from .linear import linear

class rnnbase(Module):
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
        super(rnnbase, self).__init__()
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

        self.kernels = []
        self.output = []
        for layer in range(num_layers):
            directions = []
            for direction in range(num_directions):
                layer_input_size = input_size if layer == 0 else hidden_size
                gates = []
                for gate in self.gate_size:
                    w_ih = linear(layer_intput_size, hidden_size, bias=self.bias)
                    w_hh = linear(hidden_size, hidden_size, bias=self.bias)

                    a_ih_hh = add()
                    gates.append([w_ih, w_hh, a_ih_hh])
                self.output.append(linear(hidden_size, hidden_size, bias=self.bias))
                directions.append(gates)
            self.kernels.append(directions)

        RELU = lambda x: np.max(x, 0.0)

        self.activation1 = sigmoid()
        self.activation2 = np.tanh if mode == 'RNN_TANH' else RELU
        self.activation3 = softmax(0)

        def forward_cpu(self, x: tensor, hx: tensor, cx: tensor):
            max_batch_size = x.shape[0] if self.batch_first else x.shape[1]
            num_directions = 2 if self.bidirectional else 1

            if self.y is None:
                self.y = zeros([self.num_layers * num_directions, max_batch_size, self.hidden_siz])
            if hx is None:
                hx = zeros([self.num_layers * num_directions, max_batch_size, self.hidden_size])
            if cx is None and self.mode == 'LSTM':
                cx = zeros([self.num_layers * num_directions, max_batch_size, self.hidden_size])

            for layer in range(self.num_layers):
                for direction in range(num_directions):
                    gates = []
                    for gate in range(self.gates - 1):
                        tx = self.kernel[layer][direction][gate][0].forward_cpu(x)
                        th = self.kernel[layer][direction][gate][1].forward_cpu(hx[layer * num_directions + direction])
                        thx = self.kernel[layer][direction][gate][2].forward_cpu(tx, th)
                        tha = self.activation1.forward_cpu(thx)
                        gates.append(tha)

                    if self.mode == 'LSTM':
                        tx = self.kernel[layer][direction][-1][0].forward_cpu(x)
                        th = self.kernel[layer][direction][-1][1].forward_cpu(hx[layer * num_directions + direction])
                        thx = self.kernel[layer][direction][-1][2].forward_cpu(tx, th)
                        tha = self.activation2(thx.host_data)
                        c = cx.host_data[layer * num_directions + direction]
                        hi = gates[0].host_data
                        hf = gates[1].host_data
                        ho = gates[2].host_data
                        hg = tha
                        c = hf * c + hi * hg
                        h = ho * self.activation2(c)
                        hx.host_data[layer * num_direction + direction] = h
                        cx.host_data[layer * num_direction + direction] = c
                        yx = self.output[layer * num_directions + direction].forward_cpu(hx)

                    elif self.mode == 'GRU':
                        hr = gates[0].host_data
                        hz = gates[1].host_data
                        tx = self.kernel[layer][direction][-1][0].forward_cpu(x)
                        th = self.kernel[layer][direction][-1][1].forward_cpu(hx[layer * num_directions + direction])
                        th.host_data = hr * th.host_data
                        thx = self.kernel[layer][direction][-1][2].forward_cpu(tx, th)
                        tha = self.activation2(thx.host_data)
                        h = (1 - hz) * hn + hz * hx[layer * num_directions + direction]
                        hx.host_data[layer * num_directions + direction] = h                                            yx = self.output[layer * num_directions + direction].forward_cpu(hx)

                    else:
                        tx = self.kernel[layer][direction][-1][0].forward_cpu(x[0])
                        th = self.kernel[layer][direction][-1][1].forward_cpu(hx[layer * num_directions + direction])
                        thx = self.kernel[layer][direction][-1][2].forward_cpu(tx, th)
                        thx.host_data = self.activation2(thx.host_data)
                        hx.host_data[layer * num_directions + direction] = thx.host_data
                        yx = self.output[layer * num_directions + direction].forward_cpu(hx)

                    gates.clear()

        if self.mode == 'LSTM':
            return self.y, hx, cx
        return self.y, hx

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