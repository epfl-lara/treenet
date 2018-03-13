# Copyright 2018 EPFL.

import unittest

import torch
from torch import nn

from treenet.encoder import TreeEncoder
from treenet.treenet import TreeNet

class InterpreterUnit(nn.Module):

    def forward(self, arities, inputs, children):
        n = arities.size(0)
        outputs = []
        for i in range(n):
            x = inputs[i]
            if x.data[0] > 0:
                outputs.append(children[0][i] + children[1][i])
            elif x.data[1] > 0:
                outputs.append(children[0][i] * children[1][i])
            elif x.data[2] > 0:
                if arities[i] == 1:
                    outputs.append(-children[0][i])
                else:
                    outputs.append(children[0][i] - children[1][i])
            else:
                outputs.append(x[3:])
        return torch.stack(outputs, dim=0)

class TestTreeNetEncoderUnit(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestTreeNetEncoderUnit, self).__init__(*args, **kwargs)

        def value(tree):
            out = torch.IntTensor(4)
            out.fill_(0)

            op = tree[0]
            if op == '+':
                out[0] = 1
            elif op == '*':
                out[1] = 1
            elif op == '-':
                out[2] = 1
            else:
                out[3] = op

            return out

        def children(tree):
            return tree[1]

        self.encoder = TreeEncoder(value, children)

    def test_batch(self):

        # 42  ==>  42
        tree1 = (42, [])

        # (3 * -4) + (10 - 9)  ==>  -11
        tree2 = ('+', [('*', [(3, []), ('-', [(4, [])])]),
                       ('-', [(10, []),
                              (9, [])])])

        # -(17 - (3 + 4)) * 2  ==>  -20
        tree3 = ('*', [('-', [('-', [(17, []),
                                     ('+', [(3, []),
                                            (4, [])])])]),
                       (2, [])])

        net = TreeNet(1, 2, InterpreterUnit())

        inputs, arities = self.encoder.encode_batch([tree1, tree2, tree3])
        inputs = torch.autograd.Variable(inputs)
        outputs = net(inputs, arities)

        self.assertEqual(outputs.size(), torch.Size([3, 1]))
        self.assertTrue(type(outputs) is torch.autograd.Variable)
        self.assertTrue(type(outputs.data) is torch.IntTensor)

        self.assertEqual(outputs.data[0, 0], 42)
        self.assertEqual(outputs.data[1, 0], -11)
        self.assertEqual(outputs.data[2, 0], -20)


class LinearSumUnit(nn.Module):

    def __init__(self, input_dim, memory_dim):
        super(LinearSumUnit, self).__init__()
        self.fc1 = nn.Linear(input_dim, memory_dim)
        self.fc2 = nn.Linear(memory_dim, memory_dim)

    def forward(self, arities, inputs, children):
        sum_children = torch.sum(torch.stack(children, dim=0), dim=0)
        return self.fc1(inputs) + self.fc2(sum_children)

class TestTreeNetEncoderUnit(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestTreeNetEncoderUnit, self).__init__(*args, **kwargs)

        self.encoder = TreeEncoder(lambda x: (x[0],), lambda x: x[1])

    def test_single(self):
        net = TreeNet(3, 4, LinearSumUnit(1, 3))
        tree = (12, [(1, []),
                     (14, [(17, []), (29, [])]),
                     (12, []),
                     (70, [])])

        inputs, arities = self.encoder.encode_batch([tree])
        inputs = torch.autograd.Variable(inputs)
        outputs = net(inputs, arities)

        self.assertEqual(outputs.size(), torch.Size([1, 3]))

        target = torch.autograd.Variable(torch.Tensor([[155, 12, 72]]))
        loss = nn.functional.mse_loss(outputs, target)
        net.zero_grad()
        self.assertTrue(net.unit.fc1.bias.grad is None)
        loss.backward()
        self.assertTrue(net.unit.fc1.bias.grad is not None)
