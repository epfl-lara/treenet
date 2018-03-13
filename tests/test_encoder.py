# Copyright 2018 EPFL.

import unittest

import torch

from treenet.encoder import TreeEncoder

class TestListEncoder(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestListEncoder, self).__init__(*args, **kwargs)
        self.encoder = TreeEncoder(
            lambda x: [x[0], x[0] * 2],
            lambda x: x[1])

    def test_encode_single_node(self):
        tree = (42, [])
        values, arities = self.encoder.encode(tree)

        self.assertEqual(values.size(), torch.Size((1, 2)))
        self.assertEqual(values[0, 0], 42)
        self.assertEqual(values[0, 1], 84)

        self.assertEqual(arities.size(), torch.Size((1,)))
        self.assertEqual(arities[0], 0)

    def test_encode_tree(self):
        tree = (42, [(17, []),
                     (12, [(10, [(53, []),
                                 (15, [(8, [])])])]),
                     (28, [])])
        values, arities = self.encoder.encode(tree)
        self.assertEqual(values.size(), torch.Size((8, 2)))
        self.assertEqual([list(value) for value in values], [
            [28, 56],
            [8, 16],
            [15, 30],
            [53, 106],
            [10, 20],
            [12, 24],
            [17, 34],
            [42, 84]])

        self.assertEqual(arities.size(), torch.Size((8,)))
        self.assertEqual(list(arities), [0, 0, 1, 0, 2, 1, 0, 3])

    def test_encode_batch(self):
        tree1 = (11, [(81, []),
                      (9, [])])

        tree2 = (42, [(17, []),
                      (12, [(10, [(53, []),
                                  (15, [(8, [])])])]),
                      (28, [])])

        tree3 = (18, [(19, []),
                      (14, []),
                      (11, [(79, [])]),
                      (99, []),
                      (70, [])])

        values, arities = self.encoder.encode_batch([tree1, tree2, tree3])

        self.assertEqual(values.size(), torch.Size((8, 3, 2)))

        self.assertEqual([list(value) for value in values[:, 0, :]], [
            [9, 18],
            [81, 162],
            [11, 22],
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0]])

        self.assertEqual([list(value) for value in values[:, 1, :]], [
            [28, 56],
            [8, 16],
            [15, 30],
            [53, 106],
            [10, 20],
            [12, 24],
            [17, 34],
            [42, 84]])

        self.assertEqual([list(value) for value in values[:, 2, :]], [
            [70, 140],
            [99, 198],
            [79, 158],
            [11, 22],
            [14, 28],
            [19, 38],
            [18, 36],
            [0, 0]])

        self.assertEqual(arities.size(), torch.Size((8, 3)))

        self.assertEqual(list(arities[:, 0]), [0, 0, 2, -1, -1, -1, -1, -1])
        self.assertEqual(list(arities[:, 1]), [0, 0, 1, 0, 2, 1, 0, 3])
        self.assertEqual(list(arities[:, 2]), [0, 0, 0, 1, 0, 0, 5, -1])

    def test_encode_batch_batch_first(self):
        tree1 = (11, [(81, []),
                      (9, [])])

        tree2 = (42, [(17, []),
                      (12, [(10, [(53, []),
                                  (15, [(8, [])])])]),
                      (28, [])])

        tree3 = (18, [(19, []),
                      (14, []),
                      (11, [(79, [])]),
                      (99, []),
                      (70, [])])

        values, arities = self.encoder.encode_batch([tree1, tree2, tree3],
                                                    batch_first=True)

        self.assertEqual(values.size(), torch.Size((3, 8, 2)))

        self.assertEqual([list(value) for value in values[0]], [
            [9, 18],
            [81, 162],
            [11, 22],
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0]])

        self.assertEqual([list(value) for value in values[1]], [
            [28, 56],
            [8, 16],
            [15, 30],
            [53, 106],
            [10, 20],
            [12, 24],
            [17, 34],
            [42, 84]])

        self.assertEqual([list(value) for value in values[2]], [
            [70, 140],
            [99, 198],
            [79, 158],
            [11, 22],
            [14, 28],
            [19, 38],
            [18, 36],
            [0, 0]])

        self.assertEqual(arities.size(), torch.Size((3, 8)))

        self.assertEqual(list(arities[0]), [0, 0, 2, -1, -1, -1, -1, -1])
        self.assertEqual(list(arities[1]), [0, 0, 1, 0, 2, 1, 0, 3])
        self.assertEqual(list(arities[2]), [0, 0, 0, 1, 0, 0, 5, -1])

class TestTensorEncoder(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestTensorEncoder, self).__init__(*args, **kwargs)
        self.encoder = TreeEncoder(
            lambda x: torch.Tensor([x[0], x[0] * 2, x[0] + 1]),
            lambda x: x[1])

    def test_encode_tree(self):
        tree = (42, [(17, []),
                     (12, [(10, [(53, []),
                                 (15, [(8, [])])])]),
                     (28, [])])
        values, arities = self.encoder.encode(tree)
        self.assertEqual(values.size(), torch.Size((8, 3)))
        self.assertEqual([list(value) for value in values], [
            [28, 56, 29],
            [8, 16, 9],
            [15, 30, 16],
            [53, 106, 54],
            [10, 20, 11],
            [12, 24, 13],
            [17, 34, 18],
            [42, 84, 43]])

class TestTupleEncoder(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestTupleEncoder, self).__init__(*args, **kwargs)
        self.encoder = TreeEncoder(
            lambda x: (x[0], x[0] * 2, x[0] + 1),
            lambda x: x[1])

    def test_encode_tree(self):
        tree = (42, [(17, []),
                     (12, [(10, [(53, []),
                                 (15, [(8, [])])])]),
                     (28, [])])
        values, arities = self.encoder.encode(tree)
        self.assertEqual(values.size(), torch.Size((8, 3)))
        self.assertEqual([list(value) for value in values], [
            [28, 56, 29],
            [8, 16, 9],
            [15, 30, 16],
            [53, 106, 54],
            [10, 20, 11],
            [12, 24, 13],
            [17, 34, 18],
            [42, 84, 43]])

class TestByteTensorEncoder(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestByteTensorEncoder, self).__init__(*args, **kwargs)
        self.encoder = TreeEncoder(
            lambda x: torch.ByteTensor([x[0], x[0] * 2, x[0] + 1]),
            lambda x: x[1])

    def test_encode_tree(self):
        tree = (42, [(17, []),
                     (12, [(10, [(53, []),
                                 (15, [(8, [])])])]),
                     (28, [])])
        values, arities = self.encoder.encode(tree)
        self.assertEqual(values.size(), torch.Size((8, 3)))
        self.assertEqual([list(value) for value in values], [
            [28, 56, 29],
            [8, 16, 9],
            [15, 30, 16],
            [53, 106, 54],
            [10, 20, 11],
            [12, 24, 13],
            [17, 34, 18],
            [42, 84, 43]])
        self.assertTrue(type(values) is torch.ByteTensor)
        self.assertTrue(type(arities) is torch.LongTensor)

    def test_encode_batch(self):
        tree1 = (42, [(17, []),
                      (12, [(10, [(53, []),
                                  (15, [(8, [])])])]),
                      (28, [])])
        tree2 = (12, [(17, [(8, [])]),
                      (16, [(19, [(61, []),
                                  (10, [])])]),
                      (56, [])])
        values, arities = self.encoder.encode_batch([tree1, tree2])
        self.assertTrue(type(values) is torch.ByteTensor)
        self.assertTrue(type(arities) is torch.LongTensor)

if __name__ == '__main__':
    unittest.main()