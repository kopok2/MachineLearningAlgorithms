# coding=utf-8
"""Module implements State-of-the-Art classification and regression machine learning algorithm -
Accelerated Gradient Boosting.

Research source:
    link: https://arxiv.org/pdf/1803.02042.pdf

Abstract:
Gradient tree boosting is a prediction algorithm that sequentially produces a model in the form of
linear combinations of decision trees, by solving an infinite-dimensional optimization problem.
We combine gradient boosting and Nesterov’s accelerated descent to design a new algorithm,
which we callAGB(for Accelerated Gradient Boosting).
Substantial numerical evidence is provided on both synthetic and real-life data sets
to assess the excellent performance of the method in a large variety of prediction problems.
It is empirically shown that AGBis much less sensitive to the shrinkage parameter
and outputs predictors that are considerably more sparse in the number of trees,
while retaining the exceptional performance of gradient boosting.
"""
import sys
import math
import operator
from operator import itemgetter

#sys.setrecursionlimit(100000)


class DecisionNode:
    """Decision node which tests at given threshold and returns subtree decision recursively."""
    def __init__(self, test_feature, test_threshold):
        self.test_feature = test_feature
        self.test_threshold = test_threshold
        self.left = None
        self.right = None

    def decide(self, features):
        """Test features at feature given by node and return subtree decision.

        Args:
            features (list): sample features.

        Returns:
            decided regression value for given features.
        """
        if features[self.test_feature] < self.test_threshold:
            result = self.left.decide(features)
        else:
            result = self.right.decide(features)
        return result


class WeakRegressionTree:
    """Iterative weak regression tree with d + 1 leaf nodes for regression (d - data dimensionality)."""
    def __init__(self):
        self.root = LeafNode(0)

    def fit(self, x, y, tested_feature):
        """Fit regression tree stump to given features x and target y.

        Args:
            x (list of lists): data features.
            y (list): data target value.
            tested_feature (int): tested feature index.
        """
        x.sort(key=itemgetter(tested_feature))
        loss = -1
        test_threshold = 0
        left_regression = 0
        right_regression = 0
        for test_boundary in range(1, len(x)):
            left_part = y[:test_boundary]
            right_part = y[test_boundary:]
            lr = sum(left_part) / len(left_part)
            rr = sum(right_part) / len(right_part)
            regression = [lr] * len(left_part) + [rr] * len(right_part)
            mean_squared_error = sum([(a - b) ** 2 for a, b in zip(regression, y)])
            if mean_squared_error < loss or loss == -1:
                loss = mean_squared_error
                test_threshold = (x[test_boundary - 1][tested_feature] + x[test_boundary][tested_feature]) / 2
                left_regression = lr
                right_regression = rr

        self.root = DecisionNode(tested_feature, test_threshold)
        self.root.left = LeafNode(left_regression)
        self.root.right = LeafNode(right_regression)

    def predict(self, x):
        """Generate regression for given x.

        Args:
            x (list of lists): given data.

        Returns:
            list of regression predictions for x.
        """
        try:
            if len(x[0]):
                result = list(map(self.root.decide, x))
            else:
                result = self.root.decide(x)
        except TypeError:
            result = self.root.decide(x)
        return result


class LeafNode:
    """Leaf node which returns regression value."""
    def __init__(self, decision):
        self.decision = decision

    def decide(self, features):
        """Return node's regression decision.

        Args:
            features (list): sample features.

        Returns:
            decided regression value for given features.
        """
        return self.decision


def f_factory(epoch, shrinkage, learner, g):
    """Generate f inplace function."""
    return lambda vec: g[epoch](vec) + shrinkage * learner.predict(vec)


def g_factory(epoch, gamma_param, f):
    """Generate g inplace function."""
    return lambda vec: (1 - gamma_param) * f[epoch + 1](vec) + gamma_param * f[epoch](vec)


class AGBRegressor:
    """Accelerated Gradient Boosting regressor."""
    def __init__(self):
        self.predict = lambda: 0

    def fit(self, x, y, shrinkage=0.9, epochs=20):
        """Fit additive model to given data.

        Args:
            x (list of lists): data.
            y (list): target values.
            shrinkage (double): learning bound.
            epochs (int): learning iterations.
        """
        lambda_param = [0]
        gamma_param = 0
        start_function = sum(y) / len(y)
        g = [lambda vec: start_function]
        f = [lambda vec: start_function]
        tested_feature = 0
        for epoch in range(epochs):
            print("Training on epoch {0}".format(epoch))
            print(epoch, f, g)
            z = list(map(lambda a: operator.sub(*a), zip(y, map(g[-1], x))))
            learner = WeakRegressionTree()
            learner.fit(x, z, tested_feature)
            f.append(f_factory(epoch, shrinkage, learner, g))
            print(f, epoch, epoch + 1)
            g.append(g_factory(epoch, gamma_param, f))
            lambda_param.append((1 + math.sqrt(1 + 4 * (lambda_param[-1] ** 2))) / 2)
            gamma_param = (1 - lambda_param[-2]) / lambda_param[-1]
            tested_feature += 1
            if tested_feature >= len(x[0]):
                tested_feature = 0
        self.predict = f[-1]


if __name__ == '__main__':
    x = [[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]]
    y = [1, 2, 3, 4]
    agb = AGBRegressor()
    agb.fit(x, y, epochs=30)
    print(list(map(agb.predict, x)), y)
