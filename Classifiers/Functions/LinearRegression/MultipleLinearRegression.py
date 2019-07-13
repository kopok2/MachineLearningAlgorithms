"""Mulitple linear regression implementation.

Author: Karol Oleszek 2019

Prediction formula:
Y = a + b1 * x1 + b2 * x2 + ... + bn * xn

Observations:
Y = [y1, y2, ..., yn]^t

Explanation variables:
     __               __
    | 1 x11 x12 ... x1k \
    | 1 x21 x22 ... x2k |
x = | .  .   .       .  |
    | .  .   .       .  |
    | 1 xn1 xn2 ... xnk |
    L_                __|

Model parameters:
B = [b1, b2, ..., bn]^t

Model errors:
E = [e1, e2, ..., en]^t

Model estimation:
Y^ = XB^
Y - Y^ = E

Least square estimator of parameters:
B^ = (X^t X)^-1 X^t y
"""

import numpy as np


class MultipleLinearRegressionModel:
    def __init__(self, x, y):
        self.X = np.hstack((np.ones(len(x)).reshape(len(x), 1), np.array(x)))
        self.Y = np.array(y)
        self.B = np.zeros(len(x)).reshape(len(x), 1)
        self.estimations = np.zeros(len(x)).reshape(len(x), 1)
        self.errors = np.zeros(len(x)).reshape(len(x), 1)

    def fit(self):
        print("Fitting model...")
        self.B = np.dot(np.dot(np.linalg.inv(np.dot(self.X.transpose(), self.X)), self.X.transpose()), self.Y)
        print("Estimated model parameters:")
        print(self.B)
        print("Assessing model...")
        self.estimations = np.dot(self.X, self.B)
        self.errors = self.Y - self.estimations
        print("Real values:")
        print(self.Y)
        print("Estimations:")
        print(self.estimations)
        print("Model errors:")
        print(self.errors)
        ssr = sum((self.estimations - np.mean(self.Y)) ** 2)
        ssres = sum((self.Y - self.estimations) ** 2)
        sst = sum((self.Y - np.mean(self.Y)) ** 2)
        F = (ssr / len(self.B)) / (ssres / (len(self.X) - len(self.B) - 1))
        print("F: {0}".format(F))
        rradj = 1 - ((ssr / (len(self.X) - len(self.B) - 1)) / (sst / (len(self.X) - 1)))
        print("R^2 adjusted: {0}".format(rradj))


    def predict(self, x):
        return np.dot(self.B[1:], np.array(x)) + self.B[0]


if __name__ == "__main__":
    mlrm = MultipleLinearRegressionModel([[1, 7], [2, 3], [3, 4], [3, 12], [12, 234]], [1, 2, 3, 3, 12])
    mlrm.fit()
    print("Making predictions:")
    print("x, prediction, error")
    for x in range(12):
        print(x, mlrm.predict([x, 0]), x - mlrm.predict([x, 0]))
