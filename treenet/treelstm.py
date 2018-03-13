# Copyright 2018 EPFL.

import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

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

        for _ in range(branching_factor):
            ufs = []
            for _ in range(branching_factor):
                ufs.append(nn.Linear(self.memory_size, self.memory_size,
                                     bias=False))
            self.ui_nets.append(nn.Linear(self.memory_size, self.memory_size,
                                          bias=False))
            self.uo_nets.append(nn.Linear(self.memory_size, self.memory_size,
                                          bias=False))
            self.uu_nets.append(nn.Linear(self.memory_size, self.memory_size,
                                          bias=False))
            self.uf_nets.append(ufs)

        for p in self.parameters():
          nn.init.normal(p)

    def forward(self, inputs, children, arities):

        i = self.wi_net(inputs)
        o = self.wo_net(inputs)
        u = self.wu_net(inputs)

        f_base = self.wf_net(inputs)
        fc_sum = Variable(inputs.data.new(self.memory_size).fill_(0))
        for k, child in enumerate(children):
            child_h, child_c = torch.chunk(child, 2, dim=1)
            i.add_(self.ui_nets[k](child_h))
            o.add_(self.uo_nets[k](child_h))
            u.add_(self.uu_nets[k](child_h))

            f = f_base
            for l, other_child in enumerate(children):
                other_child_h, _ = torch.chunk(child, 2, dim=1)
                f.add_(self.uf_nets[k][l](other_child_h))
            fc_sum.add(F.sigmoid(f) * child_c)

        c = F.sigmoid(i) * F.tanh(u) + fc_sum
        h = F.sigmoid(o) * F.tanh(c)
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


