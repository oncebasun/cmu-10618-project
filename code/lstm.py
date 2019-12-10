import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math
from torch.nn.utils.rnn import PackedSequence
from functional import *

# peephole connection lstm, provided by dynet

""" coupled lstm
reference: https://dynet.readthedocs.io/en/latest/builders.html?highlight=coupled#_CPPv4N5dynet18CoupledLSTMBuilderE
"""
def CoupledLSTMCell(input, hidden, w_ih, w_hh, w_pi, w_pf, w_po, b_ih=None, b_hh=None):
    hx, cx = hidden
    gates = F.linear(input, w_ih, b_ih) + F.linear(hx, w_hh, b_hh)
    ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

    ingate +=  F.linear(cx, w_pi)
    forgetgate +=  F.linear(cx, w_pf)

    ingate = torch.sigmoid(ingate)
    forgetgate = torch.sigmoid(forgetgate)

    cellgate   = torch.tanh(cellgate)
    cy = (forgetgate * cx) + (ingate * cellgate)
    outgate    += F.linear(cy, w_po)
    hy = outgate * torch.tanh(cy)
    return hy, cy


class CoupledLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2, bias=False, batch_first=False, dropout=0.0, bidirectional=False):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional

        num_directions = 2 if bidirectional else 1

        gate_size = 4 * hidden_size
        self._all_weights = []
        for layer in range(num_layers):
            for direction in range(num_directions):
                layer_input_size = input_size if layer == 0 else hidden_size * num_directions
                # for standard lstm part
                w_ih = Parameter(torch.Tensor(gate_size, layer_input_size))
                w_hh = Parameter(torch.Tensor(gate_size, hidden_size))
                b_ih = Parameter(torch.Tensor(gate_size))
                b_hh = Parameter(torch.Tensor(gate_size))
                # for peep hole connection part
                w_pi = Parameter(torch.Tensor(hidden_size, hidden_size))
                w_pf = Parameter(torch.Tensor(hidden_size, hidden_size))
                w_po = Parameter(torch.Tensor(hidden_size, hidden_size))
                layer_params = (w_ih, w_hh, w_pi, w_pf, w_po, b_ih, b_hh)
                param_names = ['weight_ih_l{}{}', 'weight_hh_l{}{}','weight_pi_l{}{}', 'weight_pf_l{}{}', 'weight_po_l{}{}']
                if bias:
                    param_names += ['bias_ih_l{}{}', 'bias_hh_l{}{}']
                suffix = '_reverse' if direction == 1 else ''
                param_names = [x.format(layer, suffix) for x in param_names]
                for name, param in zip(param_names, layer_params):
                    setattr(self, name, param)
                self._all_weights.append(param_names)
        self.reset_parameters()
        
    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    ### TODO
    ### def check_forward_args

    def forward(self, input, hx=None):
        is_packed = isinstance(input, PackedSequence)
        if is_packed:
            input, batch_sizes = input
            max_batch_size = batch_sizes[0]
            insize = input.shape[2:]
        else:
            batch_sizes = None
            max_batch_size = input.size(0) if self.batch_first else input.size(1)
            insize = input.shape[3:]

        if hx is None:
            num_directions = 2 if self.bidirectional else 1
            hx = input.new_zeros(self.num_layers * num_directions, max_batch_size, self.hidden_size,
                                 *insize, requires_grad=False)
            hx = (hx, hx)

        func = AutogradCoupledRNN(
            num_layers=self.num_layers,
            batch_first=self.batch_first,
            dropout=self.dropout,
            train=self.training,
            bidirectional=self.bidirectional,
            variable_length=batch_sizes is not None,
        )

        output, hidden = func(input, self.all_weights, hx, batch_sizes)
        if is_packed:
            output = PackedSequence(output, batch_sizes)
        return output, hidden
    
    @property
    def all_weights(self):
        return [[getattr(self, weight) for weight in weights] for weights in self._all_weights]


def AutogradCoupledRNN(num_layers=2, batch_first=False,
        dropout=0, train=True, bidirectional=False, variable_length=False):
        cell = CoupledLSTMCell
        rec_factory = variable_recurrent_factory if variable_length else Recurrent
        if bidirectional:
            layer = (rec_factory(cell), rec_factory(cell, reverse=True))
        else:
            layer = (rec_factory(cell),)
        
        func = StackedRNN(layer, num_layers, dropout=dropout, train=train)

        def forward(input, weight, hidden, batch_sizes):
            if batch_first and batch_sizes is None:
                input = input.transpose(0, 1)

            nexth, output = func(input, hidden, weight, batch_sizes)

            if batch_first and batch_sizes is None:
                output = output.transpose(0, 1)

            return output, nexth

        return forward


if __name__ == "__main__":
    lstm = CoupledLSTM(10, 15, num_layers=4, bidirectional=True)
    x = torch.Tensor(20, 40, 10)
    lstm(x)