# coding=utf-8
"""MNIST digits classification using support vector classification from ScikitLearn."""

from joblib import dump, load
from sklearn import svm
from sklearn import datasets
from sklearn import metrics
import matplotlib.pyplot as plt


if __name__ == "__main__":
    print("Loading MNIST dataset...")
    digits = datasets.load_digits()
    n_samples = len(digits.images)
    print("Creating SVM model...")
    clf = svm.SVC(gamma=0.001, C=100.0)
    print("Fitting model...")
    clf.fit(digits.data[:n_samples // 2], digits.target[:n_samples // 2])
    print("Validating model...")
    expected = digits.target[n_samples // 2:]
    predicted = clf.predict(digits.data[n_samples // 2:])
    print("Report for SVM classifier:\n {0}".format(metrics.classification_report(expected, predicted)))
    print("Confusion matrix for SVM classifier:\n {0}".format(metrics.confusion_matrix(expected, predicted)))
    images_predictions = list(zip(digits.images[n_samples // 2:], predicted))
    for index, (image, prediction) in enumerate(images_predictions[:4]):
        plt.subplot(2, 4, index + 5)
        plt.axis("off")
        plt.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
        plt.title("Prediction {0}".format(prediction))
    plt.show()
    print("Saving model...")
    dump(clf, "svc_mnist.joblib")
    print("Loading model...")
    clf2 = load("svc_mnist.joblib")
    predicted = clf2.predict(digits.data[n_samples // 2:])
    print("Report for SVM classifier:\n {0}".format(metrics.classification_report(expected, predicted)))
    print("Confusion matrix for SVM classifier:\n {0}".format(metrics.confusion_matrix(expected, predicted)))
