#!/usr/bin/env python3
"""Decision Tree implementation with predict function"""

import numpy as np


class Node:
    """Class representing an internal node"""

    def __init__(self, feature=None, threshold=None, left_child=None,
                 right_child=None, depth=0, is_root=False):
        """Initialize node"""
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
        """Return all leaves under node"""
        leaves = []
        if self.left_child:
            leaves.extend(self.left_child.get_leaves_below())
        if self.right_child:
            leaves.extend(self.right_child.get_leaves_below())
        return leaves

    def pred(self, x):
        """Recursive prediction"""
        if x[self.feature] > self.threshold:
            return self.left_child.pred(x)
        return self.right_child.pred(x)


class Leaf(Node):
    """Class representing a leaf"""

    def __init__(self, value, depth=0):
        """Initialize leaf"""
        super().__init__(depth=depth)
        self.value = value
        self.is_leaf = True
        self.indicator = None

    def __str__(self):
        """String representation"""
        return "-> leaf [value={}]".format(self.value)

    def get_leaves_below(self):
        """Return itself"""
        return [self]

    def pred(self, x):
        """Return value"""
        return self.value

    def update_indicator(self):
        """Create indicator function"""
        lower = self.lower
        upper = self.upper

        def indicator(x):
            for i in lower:
                if x[i] <= lower[i] or x[i] > upper[i]:
                    return 0
            return 1

        self.indicator = indicator


class Decision_Tree:
    """Decision Tree class"""

    def __init__(self, root=None):
        """Initialize tree"""
        self.root = root
        self.predict = None

    def get_leaves(self):
        """Return all leaves"""
        return self.root.get_leaves_below()

    def pred(self, x):
        """Recursive prediction"""
        return self.root.pred(x)

    def update_bounds(self):
        """Update bounds for all nodes"""

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
        """Create efficient predict function"""
        self.update_bounds()
        leaves = self.get_leaves()

        for leaf in leaves:
            leaf.update_indicator()

        self.predict = lambda A: np.array(
            [sum(leaf.indicator(x) * leaf.value for leaf in leaves)
             for x in A]
        )
