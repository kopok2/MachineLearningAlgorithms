# coding=utf-8
"""AdaBoost multiclass classification."""

from sklearn import datasets, model_selection, metrics, ensemble

if __name__ == "__main__":
    print("Loading data...")
    X, y = datasets.load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y)

    print("Fitting classifiers...")
    t = ensemble.AdaBoostClassifier(n_estimators=200)
    t.fit(X_train, y_train)

    e = ensemble.BaggingClassifier(ensemble.AdaBoostClassifier(n_estimators=200),
                                   n_estimators=35, max_features=0.5, max_samples=0.5)
    e.fit(X_train, y_train)

    print("Evaluating classifiers...")

    print("#" * 128)
    print("AdaBoost:")
    print("Test:")
    print(metrics.classification_report(y_test, t.predict(X_test)))
    print(metrics.confusion_matrix(y_test, t.predict(X_test)))
    print("Training:")
    print(metrics.classification_report(y_train, t.predict(X_train)))
    print(metrics.confusion_matrix(y_train, t.predict(X_train)))

    print("#" * 128)
    print("AdaBoost in ensemble:")
    print("Test:")
    print(metrics.classification_report(y_test, e.predict(X_test)))
    print(metrics.confusion_matrix(y_test, e.predict(X_test)))
    print("Training:")
    print(metrics.classification_report(y_train, e.predict(X_train)))
    print(metrics.confusion_matrix(y_train, e.predict(X_train)))
