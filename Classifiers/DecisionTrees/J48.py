# coding=utf-8
"""ID3(Quinlan) decision tree algorithm Python implementation."""

import numpy as np
from operator import itemgetter


class Node:
    def __init__(self, is_leaf=True, test=None, leaf_decision=None):
        self.is_leaf = is_leaf
        self.test = test
        self.leaf_decision = leaf_decision


def ID3(X_train, y_train, Attributes):
    """Recursivly construct decision tree."""
    if len(np.unique(y_train)) == 1:
        return Node(leaf_decision=y_train[0])
    else:
        if not len(Attributes):
            a, b = np.unique(y_train, return_counts=True)
            selection = list(zip(a.to_list(), b.to_list()))
            selection.sort(key=itemgetter(1), reverse=True)
            return Node(leaf_decision=selection[0][0])
        else:
            pass
