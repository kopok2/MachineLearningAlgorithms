# coding=utf-8
"""Guassian Process Classifier applied on the Iris dataset."""

from sklearn import datasets, model_selection, gaussian_process, metrics


if __name__ == "__main__":
    print("Loading data...")
    X, y = datasets.load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y)

    print("Fitting model...")
    gpc = gaussian_process.GaussianProcessClassifier(kernel=gaussian_process.kernels.RBF([1.0]))
    gpc.fit(X_train, y_train)

    print("Evaluating model...")
    print(metrics.classification_report(y_test, gpc.predict(X_test)))
    print(metrics.confusion_matrix(y_test, gpc.predict(X_test)))
