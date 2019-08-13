# encoding-utf-8
"""Logistic Regression Python implementation."""
import PyQt5
import csv
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


def sigmoid(beta, X):
    """Logistic function.

    Args:
        beta: coefficient numpy vector.
        X: variable numpy matrix.

    Returns:
        numpy vector of sigmoid function results.
                     1
    g(B, X) = ---------------
               1 + e^(B^t X)
    """
    return 1 / (1 + np.exp(-np.dot(X, beta.T)))


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


def logistic_gradient(beta, X, Y):
    """Logistic gradient function.

    Args:
        beta: coefficient numpy vector.
        X: variable numpy matrix.
        Y: target value numpy vector.

    Returns:
        Logistic gradient.
    """
    a = sigmoid(beta, X) - Y.reshape(X.shape[0], -1)
    return np.dot(a.T, X)


def cost_function(beta, X, Y):
    """Model cost function.

    Args:
        beta: coefficient numpy vector.
        X: variable numpy matrix.
        Y: target value numpy vector.

    Returns:
        Model cost.
    """
    sig_value = sigmoid(beta, X)
    Y = np.squeeze(Y)
    s1 = Y * np.log(sig_value)
    s2 = (1 - Y) * np.log(1 - sig_value)
    return np.mean(-s1 - s2)


def gradient_descent(beta, X, Y, learning_rate=0.01, converge_change=0.001):
    """Gradient descent cost function optimizer.

    Args:
        beta: coefficient numpy vector.
        X: variable numpy matrix.
        Y: target value numpy vector.
        learning_rate: gradient descent step.
        converge_change: minimal cost change to continue training.

    Returns:
        Model coefficients, iterations number.
    """
    cost = cost_function(beta, X, Y)
    change_cost = 1
    iterations = 1
    while change_cost > converge_change:
        old_cost = cost
        beta = beta - (learning_rate * logistic_gradient(beta, X, Y))
        cost = cost_function(beta, X, Y)
        change_cost = old_cost - cost
        iterations += 1

    return beta, iterations


def predict_values(beta, X):
    """Predict values for given variable matrix.

    Args:
       beta: coefficient numpy vector.
        X: variable numpy matrix.
    """
    regression = sigmoid(beta, X)
    prediction = np.where(regression >= 0.5, 1, 0)
    return np.squeeze(prediction)


def plot_regression(beta, X, Y):
    """Plot decision boundary and original classes."""
    mpl.use("QT4Agg")
    x_0 = X[np.where(Y == 0)]
    x_1 = X[np.where(Y == 1)]

    plt.scatter([x_0[:, 1]], [x_0[:, 2]], c="g", label="Y = 0")
    plt.scatter([x_1[:, 1]], [x_1[:, 2]], c="r", label="Y = 1")

    x1 = np.arange(0, 1, 0.1)
    x2 = -(beta[0, 0] + beta[0, 1] * x1) / beta[0, 2]
    plt.plot(x1, x2, c="k", label="Decision boundary")

    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.legend()
    plt.show()


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
    print("Forming valid regression variable matrix...")
    X = np.hstack((np.matrix(np.ones(X.shape[0])).T, X))
    Y = dataset[:, -1]
    beta = np.matrix(np.zeros(X.shape[1]))

    print("Fitting model...")
    print("Performing gradient descent...")
    beta, iterations = gradient_descent(beta, X, Y)
    print("Estimated regression coefficients:", beta)
    print("Gradient descent iterations: {0}".format(iterations))

    predictions = predict_values(beta, X)
    print("Correctly predicted values: {0}".format(np.sum(Y == predictions)))

    plot_regression(beta, X, Y)
