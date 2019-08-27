# coding=utf-8
"""Python implementation of k-nearest neighbors classifier.

Data consists of feature vectors and class.

Euclidean distance is used.
"""
import pandas as pd
import matplotlib.pyplot as plt


class KNearestNeighboursModel:
    def __init__(self, X, y):
        """Create classification model with variables matrix X and class vector y"""
        self.X = X
        self.y = y
        self.k = 5

    def predict(self, x, training=False):
        """Predict class for given feature vector x."""
        distances = abs(self.X - x).sum(axis=1)
        distances = distances.sort_values()
        if training:
            distances = distances[distances != 0]
        neighbours = distances.index[:self.k]
        predictions = self.y[self.y.index.isin(neighbours)]
        votes = predictions[predictions.columns[0]].value_counts()
        votes = votes.sort_values(ascending=False)
        return votes.index[0]

    def fit(self):
        precision = []
        for k in range(1, len(self.y)):
            self.k = k
            good = 0
            wrong = 0
            for index, row in self.X.iterrows():
                actual = self.predict(row, True)
                expected = self.y.at[index, "variety"]
                if actual == expected:
                    good += 1
                else:
                    wrong += 1
            precision.append(good / (good + wrong))
            print("Training for k: {0}. Achieved precision: {1}".format(k, precision[-1]))
        self.k = precision.index(max(precision)) + 1
        print("Selected k: {0}. With precision: {1}".format(self.k, precision[self.k - 1]))
        plt.plot(precision)
        plt.show()


if __name__ == "__main__":
    iris_data = pd.read_csv("iris.csv")
    X = iris_data.iloc[:, 0:4]
    y = iris_data.iloc[:, 4:]
    print(X, y)
    KNNM = KNearestNeighboursModel(X, y)
    print(KNNM.predict([1, 2, 3, 4]))
    KNNM.fit()
