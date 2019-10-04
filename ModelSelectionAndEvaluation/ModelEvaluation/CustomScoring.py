# coding=utf-8
"""Scikit-Learn custom scoring scheme."""

from CustomScoringFunction import custom_scoring_function
from sklearn.metrics import confusion_matrix, make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    custom_scorer = make_scorer(custom_scoring_function)
    model = SVC()
    res = cross_val_score(model,
                          X_train,
                          y_train,
                          scoring=custom_scorer,
                          cv=5,
                          n_jobs=-1)
    print(res)
    model.fit(X_train, y_train)
    print(confusion_matrix(y_test, model.predict(X_test)))
