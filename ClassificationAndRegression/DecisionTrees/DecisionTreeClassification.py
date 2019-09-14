# coding=utf-8
"""Decision tree classification."""

from sklearn import datasets, model_selection, metrics, tree, ensemble

if __name__ == "__main__":
    print("Loading data...")
    X, y = datasets.load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y)

    print("Fitting classifiers...")
    t = tree.DecisionTreeClassifier()
    t.fit(X_train, y_train)

    e = tree.ExtraTreeClassifier()
    e.fit(X_train, y_train)

    et = ensemble.ExtraTreesClassifier(n_estimators=50)
    et.fit(X_train, y_train)

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
    print("Extra decision tree:")
    print("Decision tree:")
    print("Test:")
    print(metrics.classification_report(y_test, e.predict(X_test)))
    print(metrics.confusion_matrix(y_test, e.predict(X_test)))
    print("Training:")
    print(metrics.classification_report(y_train, e.predict(X_train)))
    print(metrics.confusion_matrix(y_train, e.predict(X_train)))

    print("#" * 128)
    print("Extra decision tree ensemble:")
    print("Decision tree:")
    print("Test:")
    print(metrics.classification_report(y_test, et.predict(X_test)))
    print(metrics.confusion_matrix(y_test, et.predict(X_test)))
    print("Training:")
    print(metrics.classification_report(y_train, et.predict(X_train)))
    print(metrics.confusion_matrix(y_train, et.predict(X_train)))
