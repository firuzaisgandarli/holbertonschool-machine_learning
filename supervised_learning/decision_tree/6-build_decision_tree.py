#!/usr/bin/env python3
"""Decision Tree implementation with safe predict"""

import numpy as np


class Node:
    """Internal node"""

    def __init__(self, feature=None, threshold=None,
                 left_child=None, right_child=None,
                 depth=0, is_root=False):
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
        """Get all leaves"""
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
    """Leaf node"""

    def __init__(self, value, depth=0):
        super().__init__(depth=depth)
        self.value = value
        self.is_leaf = True
        self.indicator = None

    def __str__(self):
        return "-> leaf [value={}]".format(self.value)

    def get_leaves_below(self):
        return [self]

    def pred(self, x):
        return self.value

    def update_indicator(self):
        """Safe indicator (NO KeyError)"""
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
    """Decision Tree"""

    def __init__(self, root=None):
        self.root = root
        self.predict = None

    def get_leaves(self):
        return self.root.get_leaves_below()

    def pred(self, x):
        return self.root.pred(x)

    def update_bounds(self):
        """Update bounds safely"""

        # VERY IMPORTANT
        self.root.lower = {}
        self.root.upper = {}

        def recurse(node):
            if node.is_leaf:
                return

            feat = node.feature
            thr = node.threshold

            # Left child
            node.left_child.lower = node.lower.copy()
            node.left_child.upper = node.upper.copy()
            node.left_child.lower[feat] = thr

            # Right child
            node.right_child.lower = node.lower.copy()
            node.right_child.upper = node.upper.copy()
            node.right_child.upper[feat] = thr

            recurse(node.left_child)
            recurse(node.right_child)

        recurse(self.root)

    def update_predict(self):
        """Efficient predict"""

        self.update_bounds()
        leaves = self.get_leaves()

        for leaf in leaves:
            leaf.update_indicator()

        self.predict = lambda A: np.array([
            sum(leaf.indicator(x) * leaf.value for leaf in leaves)
            for x in A
        ])
