from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from madml import tensor
from madml import zeros
from .activation import softmax
from .linear import linear
from .math import sigmoid, add
from .module import Module

class rnnbase(Module):
    __constants__ = ['mode', 'input_size', 'hidden_size', 'num_layers', 'bias',
                     'batch_first', 'dropout', 'bidirectional']

    mode: str
    input_size: int
    hidden_size: int
    num_layers: int
    bias: bool
    batch_first: bool
    dropout: float
    bidirectional: bool

    def __init__(self, mode: str, input_size: int, hidden_size: int,
                 num_layers: int = 1, bias: bool = True, batch_first: bool = False,
                 dropout: float = 0., bidirectional: bool = False) -> None:
        super(rnnbase, self).__init__()
        self.mode = mode
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = float(dropout)
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

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
            for direction in range(self.num_directions):
                layer_input_size = input_size if layer == 0 else hidden_size
                gates = []
                for gate in self.gate_size:
                    w_ih = self.register_module(linear, layer_intput_size, hidden_size, self.bias)
                    w_hh = self.register_module(linear, hidden_size, hidden_size, self.bias)
                    a_ih_hh = self.register_module(linear, add)
                    gates.append([w_ih, w_hh, a_ih_hh])
                self.output.append(self.register_module(linear, hidden_size, hidden_size, self.bias))
                directions.append(gates)
            self.kernels.append(directions)

        RELU = lambda x: np.max(x, 0.0)

        self.activation1 = sigmoid()
        self.activation2 = np.tanh if mode == 'RNN_TANH' else RELU
        self.activation3 = softmax(0)

    def forward(self, x: tensor, h: tensor=None, c: tensor=None):
        max_batch_size = x.shape[0] if self.batch_first else x.shape[1]

        self.register_output_shape([self.num_layers * self.num_directions, max_batch_size, self.hidden_siz])
        self.register_forward_arg('x', x)

        if h is None:
            h = zeros([self.num_layers * self.num_directions, max_batch_size, self.hidden_size])
        self.register_forward_arg('h', h)

        if c is None and self.mode == 'LSTM':
            c = zeros([self.num_layers * self.num_directions, max_batch_size, self.hidden_size])
        self.register_forward_arg('c', c)

        self.register_backward_arg('x', x)
        self.register_backward_arg('h', h)
        self.register_backward_arg('c', c)
        self.register_backward_arg('y', self.y)

        return super(rnnbase, self).forward(x, h, c)

    def _forward_cpu(self, x: tensor, hx: tensor, cx: tensor):
        for layer in range(self.num_layers):
            for direction in range(self.num_directions):
                gates = []
                for gate in range(self.gates - 1):
                    tx = self.kernel[layer][direction][gate][0].forward(x)
                    th = self.kernel[layer][direction][gate][1].forward(hx[layer * self.num_directions + direction])
                    thx = self.kernel[layer][direction][gate][2].forward(tx, th)
                    tha = self.activation1.forward(thx)
                    gates.append(tha)
                tx = self.kernel[layer][direction][-1][0].forward(x)
                th = self.kernel[layer][direction][-1][1].forward(hx[layer * self.num_directions + direction])

                if self.mode == 'LSTM':
                    thx = self.kernel[layer][direction][-1][2].forward(tx, th)
                    tha = self.activation2(thx.host_data)
                    c = cx.host_data[layer * self.num_directions + direction]
                    hi = gates[0].host_data
                    hf = gates[1].host_data
                    ho = gates[2].host_data
                    hg = tha
                    c = hf * c + hi * hg
                    h = ho * self.activation2(c)
                    hx.host_data[layer * num_direction + direction] = h
                    cx.host_data[layer * num_direction + direction] = c
                elif self.mode == 'GRU':
                    hr = gates[0].host_data
                    hz = gates[1].host_data
                    th.host_data = hr * th.host_data
                    thx = self.kernel[layer][direction][-1][2].forward(tx, th)
                    tha = self.activation2(thx.host_data)
                    h = (1 - hz) * hn + hz * hx[layer * self.num_directions + direction]
                    hx.host_data[layer * self.num_directions + direction] = h
                else:
                    thx = self.kernel[layer][direction][-1][2].forward(tx, th)
                    thx.host_data = self.activation2(thx.host_data)
                    hx.host_data[layer * self.num_directions + direction] = thx.host_data

                yx = self.output[layer * self.num_directions + direction].forward(hx)
                gates.clear()

        if self.mode == 'LSTM':
            return self.y, hx, cx
        return self.y, hx
    def _backward_cpu(self, x: tensor, h: tensor, c: tensor, y: tensor) -> tensor:
        dx, dh, dc, dy = x.gradient, dh.gradient, h.gradient, y.gradient
        for layer in range(self.num_layers):
            for direction in range(self.num_directions):
                gates = []
                for gate in range(self.grates -1):
                    pass

        return dx

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