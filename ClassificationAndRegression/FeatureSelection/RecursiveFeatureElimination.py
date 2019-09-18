# coding=utf-8
"""Recursive Feature Elimination with Support Vector Machines Classifier."""

import matplotlib.pyplot as plt
from sklearn import datasets, svm, feature_selection, model_selection


if __name__ == "__main__":
    print("Generating data...")
    X, y = datasets.make_classification(3000, 20, n_informative=5, n_redundant=3, n_repeated=2, n_classes=3,
                                        n_clusters_per_class=1, random_state=37)

    print("Performing recursive feature selection with crossvalidation.")
    svc = svm.SVC(kernel="linear")
    cvrfs = feature_selection.RFECV(svc, cv=model_selection.StratifiedKFold(2), scoring="accuracy")
    cvrfs.fit(X, y)
    print("Optimal selected features: {0}".format(cvrfs.n_features_))

    print("Plotting results...")
    plt.plot(range(1, len(cvrfs.grid_scores_) + 1), cvrfs.grid_scores_)
    plt.show()
