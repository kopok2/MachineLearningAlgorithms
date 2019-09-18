# coding=utf-8
"""Feature selection based on feature importance in pretrained model."""

from sklearn import ensemble, feature_selection, datasets


if __name__ == "__main__":
    data = datasets.load_iris()
    X, y = data.data, data.target
    print(X, X.shape)

    clf = ensemble.ExtraTreesClassifier(n_estimators=100)
    clf.fit(X, y)
    selector = feature_selection.SelectFromModel(clf, prefit=True)
    X_t = selector.transform(X)
    print(X_t, X_t.shape)
