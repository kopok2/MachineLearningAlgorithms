# coding=utf-8
"""Comparison of decision tree classifier and random forest classifier."""

from sklearn import datasets, model_selection, metrics, tree, ensemble

if __name__ == "__main__":
    print("Loading data...")
    X, y = datasets.load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y)

    print("Fitting classifiers...")
    t = tree.DecisionTreeClassifier()
    t.fit(X_train, y_train)

    e = ensemble.RandomForestClassifier(n_estimators=50)
    e.fit(X_train, y_train)

    ex = ensemble.ExtraTreesClassifier(n_estimators=50)
    ex.fit(X_train, y_train)

    print("Evaluating classifiers...")

    print("#" * 128)
    print("Decision tree:")
    print("Test:")
    print(metrics.classification_report(y_test, t.predict(X_test)))
    print(metrics.confusion_matrix(y_test, t.predict(X_test)))
    print("Training:")
    print(metrics.classification_report(y_train, t.predict(X_train)))
    print(metrics.confusion_matrix(y_train, t.predict(X_train)))

    print("#" * 128)
    print("Random forest:")
    print("Test:")
    print(metrics.classification_report(y_test, e.predict(X_test)))
    print(metrics.confusion_matrix(y_test, e.predict(X_test)))
    print("Training:")
    print(metrics.classification_report(y_train, e.predict(X_train)))
    print(metrics.confusion_matrix(y_train, e.predict(X_train)))

    print("#" * 128)
    print("Extreme random forest:")
    print("Test:")
    print(metrics.classification_report(y_test, ex.predict(X_test)))
    print(metrics.confusion_matrix(y_test, ex.predict(X_test)))
    print("Training:")
    print(metrics.classification_report(y_train, ex.predict(X_train)))
    print(metrics.confusion_matrix(y_train, ex.predict(X_train)))
