# coding=utf-8
"""ScikitLearn KNN Iris dataset classification."""

from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import numpy as np


if __name__ == "__main__":
    print("Loading and shuffling data...")
    iris = datasets.load_iris()
    iris_X = iris.data
    iris_y = iris.target
    np.random.seed(0)
    indices = np.random.permutation(len(iris_X))
    iris_X_train = iris_X[indices[:-10]]
    iris_y_train = iris_y[indices[:-10]]
    iris_X_test = iris_X[indices[-10:]]
    iris_y_test = iris_y[indices[-10:]]
    print("Creating model...")
    knn = KNeighborsClassifier()
    knn.fit(iris_X_train, iris_y_train)
    predicted = knn.predict(iris_X_test)
    print(metrics.classification_report(iris_y_test, predicted))
    print(metrics.confusion_matrix(iris_y_test, predicted))
