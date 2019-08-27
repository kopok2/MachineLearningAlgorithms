# coding=utf-8
"""ID3(Quinlan) decision tree algorithm Python implementation."""

import math
from operator import itemgetter
import numpy as np


def entropy(distribution):
    if not sum(distribution):
        return 1
    prob1 = distribution[0] / sum(distribution)
    prob2 = distribution[1] / sum(distribution)
    if not prob1 or not prob2:
        return 0
    else:
        return -prob1 * math.log2(prob1) - prob2 * math.log2(prob2)


def information_gain(y, y_neg, y_pos):
    s_distr = (np.sum(y), len(y) - np.sum(y))
    neg_distr = (np.sum(y_neg), len(y_neg) - np.sum(y_neg))
    pos_distr = (np.sum(y_pos), len(y_pos) - np.sum(y_pos))
    return entropy(s_distr) - (len(y_neg) / len(y)) * entropy(neg_distr) - (len(y_pos) / len(y)) * entropy(pos_distr)


def best_atr(X_train, y_train, Attributes):
    results = []
    for atr in Attributes:
        y = y_train
        y_pos = y_train[X_train[:, atr] == True]
        y_neg = y_train[X_train[:, atr] == False]
        results.append((atr, information_gain(y, y_neg, y_pos)))
    results.sort(key=itemgetter(1), reverse=True)
    print(results)
    return results[0][0]


class Node:
    def __init__(self, is_leaf=True, test=None, leaf_decision=None):
        self.is_leaf = is_leaf
        self.test = test
        self.leaf_decision = leaf_decision
        self.true_decision = None
        self.false_decision = None


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
            atr = best_atr(X_train, y_train, Attributes)
            root = Node(is_leaf=False, test=atr)
            natr = Attributes[::]
            natr.remove(atr)
            if not len(X_train[X_train[:, atr] == True]):
                a, b = np.unique(y_train, return_counts=True)
                selection = list(zip(a.to_list(), b.to_list()))
                selection.sort(key=itemgetter(1), reverse=True)
                root.true_decision = selection[0][0]
            else:
                root.true_decision = ID3(X_train[X_train[:, atr] == True], y_train[X_train[:, atr] == True], natr)
            if not len(X_train[X_train[:, atr] == False]):
                a, b = np.unique(y_train, return_counts=True)
                selection = list(zip(a.to_list(), b.to_list()))
                selection.sort(key=itemgetter(1), reverse=True)
                root.false_decision = selection[0][0]
            else:
                root.false_decision = ID3(X_train[X_train[:, atr] == False], y_train[X_train[:, atr] == False], natr)
        return root


if __name__ == "__main__":
    X_train = np.array([[True, True, False], [True, False, False], [False, True, False], [False, True, False]])
    y_train = np.array([True, True, False, False])
    model = ID3(X_train, y_train, [0, 1, 2])
    to_visit = [model]
    while to_visit:
        visiting = to_visit.pop(0)
        if visiting:
            print(visiting, visiting.is_leaf, visiting.leaf_decision, visiting.test)
            to_visit.append(visiting.true_decision)
            to_visit.append(visiting.false_decision)
