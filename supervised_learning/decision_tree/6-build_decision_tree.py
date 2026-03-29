#!/usr/bin/env python3
"""Decision Tree implementation with optimized predict method"""

import numpy as np


class Node:
    """Class representing an internal node of the decision tree"""

    def __init__(self, feature=None, threshold=None,
                 left_child=None, right_child=None,
                 depth=0, is_root=False):
        """Initializes the node"""
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.depth = depth
        self.is_root = is_root
        self.is_leaf = False
        self.lower = {}
        self.upper = {}

    def get_leaves_below(self):
        """Returns all leaves below this node"""
        leaves = []
        if self.left_child:
            leaves.extend(self.left_child.get_leaves_below())
        if self.right_child:
            leaves.extend(self.right_child.get_leaves_below())
        return leaves

    def pred(self, x):
        """Recursively computes prediction for one sample"""
        if x[self.feature] > self.threshold:
            return self.left_child.pred(x)
        return self.right_child.pred(x)


class Leaf(Node):
    """Class representing a leaf node"""

    def __init__(self, value, depth=0):
        """Initializes the leaf"""
        super().__init__(depth=depth)
        self.value = value
        self.is_leaf = True
        self.indicator = None

    def __str__(self):
        """Returns string representation of the leaf"""
        return "-> leaf [value={}]".format(self.value)

    def get_leaves_below(self):
        """Returns a list containing only this leaf"""
        return [self]

    def pred(self, x):
        """Returns the stored value for prediction"""
        return self.value

    def update_indicator(self):
        """Creates an indicator function for this leaf"""
        lower = self.lower
        upper = self.upper

        def indicator(x):
            for i in set(list(lower.keys()) + list(upper.keys())):
                if x[i] <= lower.get(i, -np.inf):
                    return 0
                if x[i] > upper.get(i, np.inf):
                    return 0
            return 1

        self.indicator = indicator


class Decision_Tree:
    """Class representing a decision tree"""

    def __init__(self, root=None):
        """Initializes the decision tree"""
        self.root = root
        self.predict = None

    def get_leaves(self):
        """Returns all leaves of the tree"""
        return self.root.get_leaves_below()

    def pred(self, x):
        """Computes prediction for one sample using recursion"""
        return self.root.pred(x)

    def update_bounds(self):
        """Updates lower and upper bounds for all nodes"""

        self.root.lower = {}
        self.root.upper = {}

        def recurse(node):
            if node.is_leaf:
                return

            feat = node.feature
            thr = node.threshold

            node.left_child.lower = node.lower.copy()
            node.left_child.upper = node.upper.copy()
            node.left_child.lower[feat] = thr

            node.right_child.lower = node.lower.copy()
            node.right_child.upper = node.upper.copy()
            node.right_child.upper[feat] = thr

            recurse(node.left_child)
            recurse(node.right_child)

        recurse(self.root)

    def update_predict(self):
        """Creates a fast vectorized predict function"""

        self.update_bounds()
        leaves = self.get_leaves()

        for leaf in leaves:
            leaf.update_indicator()

        self.predict = lambda A: np.array([
            sum(leaf.indicator(x) * leaf.value for leaf in leaves)
            for x in A
        ])
