# Copyright 2018 EPFL.

import torch

class TreeEncoder(object):
    """Encoder for objects with tree structure.

    Args:
        value_fn: The function used to extract features from nodes.
            Should either return one-dimensional tensors,
            lists, or tuples.
        children_fn: The function used to get the children of nodes.
    """

    def __init__(self, value_fn, children_fn):
        super(TreeEncoder, self).__init__()
        self.value_fn = value_fn
        self.children_fn = children_fn


    def encode(self, tree):
        """Encodes a tree-like object.

        Args:
            tree: A tree-like object.

        Returns:
            (Tensor, LongTensor): A pair of two tensors,
            one for the values of the tree and one for the arities.

            The returned tensor of values is of shape `(N, features_size)`,
            where `N` is the number of elements in the tree and `features_size`
            is the number of features.

            The returns tensor of arities is of shape `(N,)`,
            where `N` is the number of elements in the tree.

            The values and arities appear in right-first post-order.
        """

        children = self.children_fn(tree)
        n = len(children)
        value = self.value_fn(tree)
        if type(value) is list or type(value) is tuple:
            value = torch.Tensor(value)

        if children:
            lower_values, lower_arities = \
                zip(* [self.encode(c) for c in reversed(children)])

            lower_values = list(lower_values)
            lower_arities = list(lower_arities)
        else:
            lower_values = []
            lower_arities = []
        lower_values.append(value.unsqueeze(0))
        lower_arities.append(torch.LongTensor([len(children)]))
        return torch.cat(lower_values), torch.cat(lower_arities)


    def encode_batch(self, trees, batch_first=False, ignore_value=None):
        """Encodes a sequence of tree-like objects.

        Args:
            trees: A sequence of tree-like objects.
            batch_first: If ``True``, the values are returned with the
                batch dimension first. Otherwise, the temporal dimension is
                returned first.
                Default: ``False``
            ignore_value: The features used to pad the tensor of features.
                Can either be a one dimensional Tensor, a list or a tuple.
                Default: Zeros.

        Returns:
            (Tensor, LongTensor): A pair of two tensors,
            one for the values of the trees and one for the arities.

            The returned tensor of values is of shape
            `(N, batch_size, features_size)`, where
            `N` is the largest number of elements in the trees,
            `batch_size` is the number of trees, and
            `features_size` is the number of features.

            The returns tensor of arities is of shape `(N, batch_size)`, where
            `N` is the largest number of elements in the trees, and
            `batch_size` is the number of trees.

            The values are padded by `ignore_value` (by default zeros), and
            the arities are padded by ``-1``.

            The values and arities appear in post-order.
        """

        if type(ignore_value) is list or type(ignore_value) is tuple:
            ignore_value = torch.FloatTensor(ignore_value)

        batch_dim = 0 if batch_first else 1
        all_values = []
        all_arities = []
        max_size = 0
        for tree in trees:
            values, arities = self.encode(tree)
            all_values.append(values)
            all_arities.append(arities)
            max_size = max(max_size, values.size(0))

        def pad_values(tensor):
            dims = list(tensor.size())
            dims[0] = max_size - dims[0]
            if ignore_value is not None:
                padding = ignore_value.unsqueeze(0).expand(dims)
            else:
                padding = tensor.new(* dims).fill_(0)
            return torch.cat([tensor, padding])

        def pad_arities(tensor):
            pad_size = max_size - tensor.size(0)
            padding = tensor.new(pad_size).fill_(-1)
            return torch.cat([tensor, padding])

        all_values = [pad_values(v) for v in all_values]
        all_arities = [pad_arities(a) for a in all_arities]

        all_values = torch.stack(all_values, dim=batch_dim)
        all_arities = torch.stack(all_arities, dim=batch_dim)

        return all_values, all_arities