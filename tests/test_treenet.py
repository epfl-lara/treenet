# Copyright 2018 EPFL.

import unittest

import torch
from torch import nn
from torch.autograd import Variable

from treenet.encoder import TreeEncoder
from treenet.treenet import TreeNet

class InterpreterUnit(nn.Module):

    def forward(self, inputs, children, arities):
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

class TestTreeNetInterpreterUnit(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestTreeNetInterpreterUnit, self).__init__(*args, **kwargs)

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
        inputs = Variable(inputs)
        outputs = net(inputs, arities)

        self.assertEqual(outputs.size(), torch.Size([3, 1]))
        self.assertTrue(type(outputs) is Variable)
        self.assertTrue(type(outputs.data) is torch.IntTensor)

        self.assertEqual(outputs.data[0, 0], 42)
        self.assertEqual(outputs.data[1, 0], -11)
        self.assertEqual(outputs.data[2, 0], -20)


class LinearSumUnit(nn.Module):

    def __init__(self, input_dim, memory_dim):
        super(LinearSumUnit, self).__init__()
        self.fc1 = nn.Linear(input_dim, memory_dim)
        self.fc2 = nn.Linear(memory_dim, memory_dim)

    def forward(self, inputs, children, arities):
        sum_children = torch.sum(torch.stack(children, dim=0), dim=0)
        return self.fc1(inputs) + self.fc2(sum_children)

class TestTreeNetLinearSumUnit(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestTreeNetLinearSumUnit, self).__init__(*args, **kwargs)

        self.encoder = TreeEncoder(lambda x: (x[0],), lambda x: x[1])

    def test_single(self):
        net = TreeNet(3, 4, LinearSumUnit(1, 3))
        tree = (12, [(1, []),
                     (14, [(17, []), (29, [])]),
                     (12, []),
                     (70, [])])

        inputs, arities = self.encoder.encode_batch([tree])
        inputs = Variable(inputs)
        outputs = net(inputs, arities)

        self.assertEqual(outputs.size(), torch.Size([1, 3]))

        target = Variable(torch.Tensor([[155, 12, 72]]))
        loss = nn.functional.mse_loss(outputs, target)
        net.zero_grad()
        self.assertTrue(net.unit.fc1.bias.grad is None)
        loss.backward()
        self.assertTrue(net.unit.fc1.bias.grad is not None)

class BasicTreeNet(nn.Module):

    def __init__(self, output_size, branching_factor, unit):
        super(BasicTreeNet, self).__init__()
        self.output_size = output_size
        self.branching_factor = branching_factor
        self.unit = unit

    def forward(self, tree):
        arity = len(tree[1])
        children_results = [self.forward(child) for child in tree[1]]
        children_results_padding = [Variable(torch.zeros(1, self.output_size))\
            for _ in range(self.branching_factor - arity)]
        children_results.extend(children_results_padding)
        arities = torch.LongTensor([arity])
        inputs = torch.autograd.Variable(tree[0].unsqueeze(0))
        return self.unit(inputs, children_results, arities)

class TestEquivalenceTreeNetBasic(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestEquivalenceTreeNetBasic, self).__init__(*args, **kwargs)
        self.encoder = TreeEncoder(lambda x: x[0], lambda x: x[1])

    def test_equivalence(self):
        def t(elems):
            return torch.Tensor(elems)
        tree = (t([1, 2, 3]), [(t([4, 5, 6]), []),
                               (t([7, 8, 9]), [(t([10, 11, 12]), [])]),
                               (t([13, 14, 15]), [(t([16, 17, 18]), []),
                                                  (t([19, 20, 21]),
                                                   [(t([22, 23, 24]), [])])])])

        unit = LinearSumUnit(3, 5)

        net = TreeNet(5, 3, unit)
        inputs, arities = self.encoder.encode_batch([tree])
        inputs = Variable(inputs)
        res_net = net(inputs, arities)

        basic = BasicTreeNet(5, 3, unit)
        res_basic = basic(tree)


        # Test forward direction.
        self.assertEqual(res_net.size(), res_basic.size())
        for i in range(res_net.size(1)):
            self.assertAlmostEqual(res_net.data[0, i], res_basic.data[0, i])


        # Test backward direction.
        target = Variable(torch.randn(1, 5))

        # Getting gradients after using net.
        unit.zero_grad()
        loss_net = nn.functional.mse_loss(res_net, target)
        loss_net.backward()
        grads_net = [p.grad.data.clone() for p in unit.parameters()]

        # Getting gradients after using basic.
        unit.zero_grad()
        loss_basic = nn.functional.mse_loss(res_basic, target)
        loss_basic.backward()
        grads_basic = [p.grad.data.clone() for p in unit.parameters()]

        # Checking that gradients are within a small epsilon of each other.
        epsilon = 0.0001
        for grad_net, grad_basic in zip(grads_net, grads_basic):
            self.assertTrue(((grad_net - grad_basic).abs() < epsilon).all())


if __name__ == '__main__':
    unittest.main()