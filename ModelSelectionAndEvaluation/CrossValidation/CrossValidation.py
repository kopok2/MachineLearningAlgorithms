# coding=utf-8
"""Cross Validation."""

from sklearn import model_selection, datasets, svm, metrics

if __name__ == '__main__':
    X, y = datasets.load_iris(return_X_y=True)
    model = svm.SVC(gamma="scale")
    scores = model_selection.cross_val_score(model, X, y, cv=5, scoring="f1_macro")
    print(scores)
    print(scores.mean(), scores.std())
    cv = model_selection.ShuffleSplit(n_splits=5, test_size=0.35)
    scores = model_selection.cross_val_score(model, X, y, cv=cv, scoring="f1_macro")
    print(scores)
    print(scores.mean(), scores.std())
