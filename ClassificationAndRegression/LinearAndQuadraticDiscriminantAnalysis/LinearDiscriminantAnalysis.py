# coding=utf-8
"""Linear Discriminant Analysis."""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import discriminant_analysis, datasets, tree, svm, linear_model
from sklearn.metrics import confusion_matrix, classification_report


if __name__ == "__main__":
    n_samples = 100_000
    n_features = 20
    n_averages = 50

    print("Generating data...")
    X, y = datasets.make_blobs(n_samples, 1, centers=[[-2], [2]])
    X = np.hstack([X, np.random.randn(n_samples, n_features - 1)])
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    print("Fitting models...")
    da = discriminant_analysis.LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    da.fit(X_train, y_train)

    daws = discriminant_analysis.LinearDiscriminantAnalysis(solver="lsqr")
    daws.fit(X_train, y_train)

    tr = tree.DecisionTreeClassifier()
    tr.fit(X_train, y_train)

    sv = svm.SVC(gamma="scale")
    sv.fit(X_train, y_train)

    pr = linear_model.Perceptron()
    pr.fit(X_train, y_train)

    for model in [da, daws, tr, sv, pr]:
        print(model)
        print("Test data:")
        print(confusion_matrix(y_test, model.predict(X_test)))
        print("Training data:")
        print(confusion_matrix(y_train, model.predict(X_train)))
        print("\n" + "#" * 128 + "\n")
        print(classification_report(y_test, model.predict(X_test)))
        print("\n")
