# coding=utf-8
"""SVM Pipeline Scaling and Feature Selection classification."""

import numpy as np
from sklearn import svm, datasets, feature_selection, model_selection, pipeline, preprocessing


if __name__ == "__main__":
    X, y = datasets.load_iris(return_X_y=True)
    X = np.hstack((X, 2 * np.random.random((X.shape[0], 36))))
    model = pipeline.Pipeline([("anova", feature_selection.SelectPercentile(feature_selection.chi2)),
                              ("scaling", preprocessing.StandardScaler()),
                              ("svc", svm.SVC(gamma="auto"))])
    percentiles = (x for x in range(1, 101))
    for percentile in percentiles:
        model.set_params(anova__percentile=percentile)
        print(percentile, model_selection.cross_val_score(model, X, y, cv=5).mean())
