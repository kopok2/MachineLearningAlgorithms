# coding=utf-8
"""SGD.

Stochastic Gradient Descent generalized linear model fit.
"""

import csv
import numpy as np
from sklearn import linear_model
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle


def normalize_matrix(X):
    """Normalize matrix to contain values in range 0-1.

    Args:
        X: variable numpy matrix.

    Returns:
        normalized numpy matrix.
    """
    matrix_min = np.min(X, axis=0)
    matrix_max = np.max(X, axis=0)
    matrix_value_range = matrix_max - matrix_min
    normalized = 1 - ((matrix_max - X) / matrix_value_range)
    return normalized


def load_dataset(filename):
    """Load dataset from file."""
    with open(filename, "r") as in_file:
        lines = csv.reader(in_file)
        dataset = list(lines)
        for i in range(len(dataset)):
            dataset[i] = [float(x) for x in dataset[i]]
        return np.array(dataset)


if __name__ == "__main__":
    print("Loading dataset...")
    dataset = load_dataset("TestDataset.csv")
    print("Preparing data...")
    print("Normalizing variable matrix...")
    X = normalize_matrix(dataset[:, :-1])
    Y = dataset[:, -1]
    X, Y = shuffle(X, Y)
    X_train = X[X.shape[0] // 2:]
    X_test = X[:X.shape[0] // 2]
    y_train = Y[X.shape[0] // 2:]
    y_test = Y[:X.shape[0] // 2]

    print("Fitting model...")
    model = linear_model.SGDClassifier()
    model.fit(X_train, y_train)

    print("Evaluating model...")
    print("Test data:")
    print(confusion_matrix(y_test, model.predict(X_test)))
    print("Training data:")
    print(confusion_matrix(y_train, model.predict(X_train)))
