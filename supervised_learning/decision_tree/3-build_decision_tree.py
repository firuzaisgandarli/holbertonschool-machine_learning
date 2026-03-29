#!/usr/bin/env python3
"""Decision Tree module with Node, Leaf, and Decision_Tree classes"""


class Node:
    """Class that represents an internal node of a decision tree"""

    def __init__(self, feature=None, threshold=None, left_child=None,
                 right_child=None, depth=0, is_root=False):
        """Initialize a Node"""
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.depth = depth
        self.is_root = is_root
        self.is_leaf = False

    def get_leaves_below(self):
        """Returns all leaves under this node"""
        leaves = []
        if self.left_child:
            leaves.extend(self.left_child.get_leaves_below())
        if self.right_child:
            leaves.extend(self.right_child.get_leaves_below())
        return leaves


class Leaf(Node):
    """Class that represents a leaf of a decision tree"""

    def __init__(self, value, depth=0):
        """Initialize a Leaf"""
        super().__init__(depth=depth)
        self.value = value
        self.is_leaf = True

    def __str__(self):
        """String representation of a Leaf"""
        return "-> leaf [value={}]".format(self.value)

    def get_leaves_below(self):
        """Returns itself as a leaf"""
        return [self]


class Decision_Tree:
    """Class that represents a decision tree"""

    def __init__(self, root=None):
        """Initialize the Decision Tree"""
        self.root = root

    def get_leaves(self):
        """Returns all leaves of the tree"""
        return self.root.get_leaves_below()
