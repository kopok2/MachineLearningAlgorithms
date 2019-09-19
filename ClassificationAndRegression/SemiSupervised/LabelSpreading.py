# coding=utf-8
"""
Label Spreading

Semi-supervised Machine Learning algorithm.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import semi_supervised, datasets


if __name__ == '__main__':
    print("Generating data...")
    samples = 300
    X, y = datasets.make_circles(n_samples=samples)
    o = 1
    i = 0
    labels = np.full(samples, -1.0)
    labels[0] = o
    labels[-1] = i

    print("Spreading labels...")
    ls = semi_supervised.label_propagation.LabelSpreading(kernel="knn", alpha=0.8)
    ls.fit(X, y)

    print("Plotting propagation...")
    out_l_ = ls.transduction_
    plt.figure(figsize=(8.5, 4))
    plt.subplot(1, 2, 1)
    plt.scatter(X[labels == o, 0], X[labels == o, 1], color='navy',
                marker='s', lw=0, label="outer labeled", s=10)
    plt.scatter(X[labels == i, 0], X[labels == i, 1], color='c',
                marker='s', lw=0, label='inner labeled', s=10)
    plt.scatter(X[labels == -1, 0], X[labels == -1, 1], color='darkorange',
                marker='.', label='unlabeled')
    plt.legend(scatterpoints=1, shadow=False, loc='upper right')
    plt.title("Raw data (2 classes=outer and inner)")

    plt.subplot(1, 2, 2)
    output_label_array = np.asarray(out_l_)
    outer_numbers = np.where(output_label_array == o)[0]
    inner_numbers = np.where(output_label_array == i)[0]
    plt.scatter(X[outer_numbers, 0], X[outer_numbers, 1], color='navy',
                marker='s', lw=0, s=10, label="outer learned")
    plt.scatter(X[inner_numbers, 0], X[inner_numbers, 1], color='c',
                marker='s', lw=0, s=10, label="inner learned")
    plt.legend(scatterpoints=1, shadow=False, loc='upper right')
    plt.title("Labels learned with Label Spreading (KNN)")

    plt.subplots_adjust(left=0.07, bottom=0.07, right=0.93, top=0.92)
    plt.show()

    print("Generating data...")
    samples = 3000
    X, y = datasets.make_blobs(n_samples=samples)
    o = 1
    i = 0
    labels = np.full(samples, -1.0)
    labels[0] = o
    labels[-1] = i

    print("Spreading labels...")
    ls = semi_supervised.label_propagation.LabelSpreading(kernel="knn", alpha=0.8)
    ls.fit(X, y)

    print("Plotting propagation...")
    out_l_ = ls.transduction_
    plt.figure(figsize=(8.5, 4))
    plt.subplot(1, 2, 1)
    plt.scatter(X[labels == o, 0], X[labels == o, 1], color='navy',
                marker='s', lw=0, label="outer labeled", s=10)
    plt.scatter(X[labels == i, 0], X[labels == i, 1], color='c',
                marker='s', lw=0, label='inner labeled', s=10)
    plt.scatter(X[labels == -1, 0], X[labels == -1, 1], color='darkorange',
                marker='.', label='unlabeled')
    plt.legend(scatterpoints=1, shadow=False, loc='upper right')
    plt.title("Raw data (2 classes=outer and inner)")

    plt.subplot(1, 2, 2)
    output_label_array = np.asarray(out_l_)
    outer_numbers = np.where(output_label_array == o)[0]
    inner_numbers = np.where(output_label_array == i)[0]
    plt.scatter(X[outer_numbers, 0], X[outer_numbers, 1], color='navy',
                marker='s', lw=0, s=10, label="outer learned")
    plt.scatter(X[inner_numbers, 0], X[inner_numbers, 1], color='c',
                marker='s', lw=0, s=10, label="inner learned")
    plt.legend(scatterpoints=1, shadow=False, loc='upper right')
    plt.title("Labels learned with Label Spreading (KNN)")

    plt.subplots_adjust(left=0.07, bottom=0.07, right=0.93, top=0.92)
    plt.show()
