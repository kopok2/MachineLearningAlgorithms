# coding=utf-8
"""Comparision of various Gradient Tree Boosting classifiers."""

import numpy as np
import lightgbm as lgb
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn import datasets, model_selection, metrics, ensemble

if __name__ == "__main__":
    print("Loading data...")
    X, y = datasets.load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y)

    print("Fitting classifiers...")
    t = ensemble.GradientBoostingClassifier()
    t.fit(X_train, y_train)

    e = HistGradientBoostingClassifier()
    e.fit(X_train, y_train)

    train_data = lgb.Dataset(X_train, label=y_train)
    lr = 0.2
    param = {'objective': 'multiclass', 'num_class': 3, "learning_rate": lr,
             'boosting': 'dart',
             'top_k': 2300, 'tree_learner': 'voting'}
    num_round = 1000

    lgb_model = lgb.train(param, train_data, num_round, valid_sets=[lgb.Dataset(X_test, y_test)],
                          early_stopping_rounds=10)

    print("Evaluating classifiers...")

    print("#" * 128)
    print("Gradient Boosting Classifier:")
    print("Test:")
    print(metrics.classification_report(y_test, t.predict(X_test)))
    print(metrics.confusion_matrix(y_test, t.predict(X_test)))
    print("Training:")
    print(metrics.classification_report(y_train, t.predict(X_train)))
    print(metrics.confusion_matrix(y_train, t.predict(X_train)))

    print("#" * 128)
    print("Hist Gradient Boosting Classifier:")
    print("Test:")
    print(metrics.classification_report(y_test, e.predict(X_test)))
    print(metrics.confusion_matrix(y_test, e.predict(X_test)))
    print("Training:")
    print(metrics.classification_report(y_train, e.predict(X_train)))
    print(metrics.confusion_matrix(y_train, e.predict(X_train)))

    print("#" * 128)
    print("LightGBM Classifier:")
    p = lgb_model.predict(X_test)
    predictions = []

    for x in p:
        predictions.append(np.argmax(x))
    print("Test:")
    print(metrics.classification_report(y_test, predictions))
    print(metrics.confusion_matrix(y_test, predictions))
    p = lgb_model.predict(X_train)
    predictions = []

    for x in p:
        predictions.append(np.argmax(x))
    print("Training:")
    print(metrics.classification_report(y_train, predictions))
    print(metrics.confusion_matrix(y_train, predictions))
