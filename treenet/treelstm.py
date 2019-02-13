# Copyright 2018 EPFL.

import torch
from torch import nn

from .treenet import TreeNet

class TreeLSTMUnit(nn.Module):
    """Tree-LSTM unit.

    Args:
        input_size (int): Number of input features.
        memory_size (int): Number of output features.
        branch_factor (int): Maximal branching factor for input trees.

    References:
        Tai, K. S., Socher, R., & Manning, C. D. (2015).
        Improved semantic representations from tree-structured
        long short-term memory networks.
    """

    def __init__(self, input_size, memory_size, branching_factor):
        super(TreeLSTMUnit, self).__init__()

        self.input_size = input_size
        self.memory_size = memory_size
        self.branching_factor = branching_factor

        self.wi_net = nn.Linear(self.input_size, self.memory_size)
        self.wo_net = nn.Linear(self.input_size, self.memory_size)
        self.wu_net = nn.Linear(self.input_size, self.memory_size)
        self.wf_net = nn.Linear(self.input_size, self.memory_size)

        self.ui_nets = []
        self.uo_nets = []
        self.uu_nets = []
        self.uf_nets = []

        for i in range(branching_factor):
            ufs = []
            for j in range(branching_factor):
                uf = nn.Linear(self.memory_size, self.memory_size, bias=False)
                self.add_module("uf_net_{}_{}".format(i, j), uf)
                ufs.append(uf)

            ui = nn.Linear(self.memory_size, self.memory_size, bias=False)
            self.add_module("ui_net_{}".format(i), ui)
            self.ui_nets.append(ui)

            uo = nn.Linear(self.memory_size, self.memory_size, bias=False)
            self.add_module("uo_net_{}".format(i), uo)
            self.uo_nets.append(uo)

            uu = nn.Linear(self.memory_size, self.memory_size, bias=False)
            self.add_module("uu_net_{}".format(i), uu)
            self.uu_nets.append(uu)

            self.uf_nets.append(ufs)

        for p in self.parameters():
          nn.init.normal_(p)

    def forward(self, inputs, children, arities):

        i = self.wi_net(inputs)
        o = self.wo_net(inputs)
        u = self.wu_net(inputs)

        f_base = self.wf_net(inputs)
        fc_sum = inputs.new_zeros(self.memory_size)
        for k, child in enumerate(children):
            child_h, child_c = torch.chunk(child, 2, dim=1)
            i.add_(self.ui_nets[k](child_h))
            o.add_(self.uo_nets[k](child_h))
            u.add_(self.uu_nets[k](child_h))

            f = f_base
            for l, other_child in enumerate(children):
                other_child_h, _ = torch.chunk(other_child, 2, dim=1)
                f = f.add(self.uf_nets[k][l](other_child_h))
            fc_sum.add(torch.sigmoid(f) * child_c)

        c = torch.sigmoid(i) * torch.tanh(u) + fc_sum
        h = torch.sigmoid(o) * torch.tanh(c)
        return torch.cat([h, c], dim=1)

class TreeLSTM(TreeNet):
    """Tree-LSTM network.

    Args:
        input_size (int): Number of input features.
        memory_size (int): Number of output features.
        branch_factor (int): Maximal branching factor for input trees.

    References:
        Tai, K. S., Socher, R., & Manning, C. D. (2015).
        Improved semantic representations from tree-structured
        long short-term memory networks.
    """

    def __init__(self, input_size, memory_size, branching_factor):
        unit = TreeLSTMUnit(input_size, memory_size, branching_factor)
        super(TreeLSTM, self).__init__(memory_size * 2, branching_factor, unit)

    def forward(self, *args, **kwargs):
        hc = super(TreeLSTM, self).forward(*args, **kwargs)
        h, _ = torch.chunk(hc, 2, dim=1)
        return h


