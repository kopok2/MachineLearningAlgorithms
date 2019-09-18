# coding=utf-8
"""Univariate Feature Selection."""

from sklearn import datasets, feature_selection


if __name__ == "__main__":
    data = datasets.make_blobs(30000, 20, center_box=(10, 20))
    X, y = data
    print(X)
    print(X.shape)

    X_t = feature_selection.SelectKBest(feature_selection.chi2, k=5).fit_transform(X, y)
    print(X_t)
    print(X_t.shape)
