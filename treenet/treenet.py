# Copyright 2018 EPFL.

import torch
from torch import nn
from torch.autograd import Variable

class TreeNet(nn.Module):
    """Class for recursive neural networks with n-ary tree structure.

    The class supports batch processing of tree-like objects with
    bounded branching factor.
    The class is intended as a base class for recursive neural networks.

    Given a `unit` network for processing single nodes (see note below),
    the TreeNet class returns a network capable of processing (properly
    encoded) trees.

    Note:
        The `unit` network specifies what should be done for each node of
        the input trees. It receives as input three parameters:
            - inputs: A Variable containing the input features of
                the current nodes. Of shape `(batch_size, input_size)`.
            - children: A list, of size `branching_factor`, of Variables
                containing the output features of the children of the
                current nodes.
                Each Variable has the shape `(batch_size, output_size)`.
                If a node has less arity than the `branching_factor`,
                the features corresponding to children absent from the
                node are guaranteed to have magnitude zero.
            - arities: A LongTensor containing the arities of the nodes.
                Of shape `(batch_size,)`.
        The `unit` network should return the output features for the current
        nodes, which should be of shape `(batch_size, output_size)`.

    Args:
        output_size (int): Number of features output by the `unit` network.
        branching_factor (int): Largest branching factor of input trees.
        unit (torch.nn.Module): Network used for processing nodes.

    See Also:
        See the `treenet.encoder` module for how to encode trees and batches
        of trees.

    References:
        Bowman, S. R., Gauthier, J., Rastogi, A., Gupta, R.,
        Manning, C. D., & Potts, C. (2016).
        A Fast Unified Model for Parsing and Sentence Understanding.
    """


    def __init__(self, output_size, branching_factor=2, unit=None):
        super(TreeNet, self).__init__()
        self.output_size = output_size
        self.branching_factor = branching_factor
        if unit is not None:
            self.unit = unit


    def forward(self, inputs, arities, batch_first=False):
        """Feed the network with encoded tree-like objects.

        Args:
            inputs (Variable): The features.
                Should be of shape `(time, batch_size, input_size)`.
            arities (LongTensor): The arities of the nodes.
                Should be of shape `(time, batch_size)`.
            batch_first (bool): If ``True``, then `inputs` and `arities`
                are expected to have the batch dimension first.

        Note:
            Inputs and arities of nodes are expected to appear in
            right-first post-order. See the `treenet.encoder`
            module for building a suitable encoder.

        Returns:
            Variable: The output features,
                of shape `(batch_size, output_size)`.
        """

        if batch_first:
            inputs = inputs.permute(1, 0, 2)
            arities = arities.permute(1, 0)

        # Time size.
        T = inputs.size(0)

        # Batch size.
        B = inputs.size(1)

        # 0, 1 .. B - 1. Used for indexing.
        k = arities.new(range(B))

        # Memory will contain the state of every node.
        memory = Variable(inputs.data.new(T, B, self.output_size))
        memory.fill_(0)

        # The stack maintains pointers to the memory for unmerged subtrees.
        # It contains extra entries, to avoid out of bounds accesses.
        stack = arities.new(B, T + self.branching_factor)
        stack.fill_(0)

        # Points to the head of the stack.
        stack_pointer = arities.new(B)
        # Starts at the given index in order to avoid out of bounds reads.
        stack_pointer.fill_(self.branching_factor - 1)

        for t in range(T):
            arity = arities[t]
            current = inputs[t]

            entries = []
            for i in range(self.branching_factor):
                entry = memory[stack[k, stack_pointer - i], k]
                mask = entry.data.new(B)
                mask.copy_(arity > i)
                mask = mask.unsqueeze(1).expand(entry.size())
                mask = Variable(mask)
                entries.append(entry * mask)

            # Obtain the state for the node.
            new_entry = self.unit(current, entries, arity)

            # If multiple entries are returned, each entry must be
            # appropriately masked.
            if type(new_entry) is list or type(new_entry) is tuple:
                for i, entry in enumerate(new_entry):
                    factors = entry.data.new(B)
                    factors.copy_(arity == i)
                    factors = factors.unsqueeze(1).expand(entry.size())
                    factors = Variable(factors)
                    memory[t] = memory[t] + (entry * factors)
            else:
                memory[t] = new_entry

            # Update the stack pointer.
            stack_pointer.add_(-torch.abs(arity) + 1)

            # Ensure that the top of the stack is untouched if the arity is the
            # special value -1.
            ignore = (arity == -1).long()
            stack[k, stack_pointer] *= ignore
            stack[k, stack_pointer] += t * ((ignore + 1) % 2)

        # Return the content of the memory location
        # pointed by the top of the stack.
        return memory[stack[k, stack_pointer], k]

