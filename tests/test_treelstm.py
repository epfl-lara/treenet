# Copyright 2018 EPFL.

import unittest

import torch

from treenet.encoder import TreeEncoder
from treenet.treelstm import TreeLSTM, TreeLSTMUnit

class TestTreeLSTM(unittest.TestCase):

    def test_batch(self):
        net = TreeLSTM(3, 7, 2)
        encoder = TreeEncoder(lambda x: x[0], lambda x: x[1])
        tree1 = ((1, 2, 3), [((4, 5, 6), []),
                             ((7, 8, 9), [])])
        tree2 = ((11, 12, 13), [((14, 15, 16), [((17, 18, 19), [])]),
                                ((20, 21, 22), [])])
        tree3 = ((21, 22, 23), [((24, 25, 26), [])])
        tree4 = ((31, 32, 33), [])

        inputs, arities = encoder.encode_batch([tree1, tree2, tree3, tree4])
        inputs = torch.autograd.Variable(inputs)

        result = net.forward(inputs, arities)

        self.assertEqual(result.size(), torch.Size([4, 7]))

if __name__ == '__main__':
    unittest.main()