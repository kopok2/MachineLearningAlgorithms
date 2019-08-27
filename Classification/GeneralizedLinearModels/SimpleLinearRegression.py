"""Simple linear regression implementation.

Author: Karol Oleszek 2019

Model symbols:
x     - explanatory variable
y     - response variable
y^    - predicted response
b0,b1 - estimated regression coefficients
e     - estimation error
y_    - mean y
x_    - mean x
n     - observations count
s     - residual standard error
mse   - mean square error

Model:
y^ = b0 + b1 * x
e = y - y^

Regression:
b0 = y_ - b1 * x_

     nE xy - ExEy
b1 = ---------------
     nEx^2 - (Ex)^2

"""

import math
import numpy as np


class SimpleLinearRegressionModel:
    """Simple linear regression model.
       Features model fitting and predictor.
    """
    def __init__(self, x, y):
        """Initialize model.
        Args:
            x - explanatory variable vector.
            y - response variable vector.
        """
        self.x = np.array(x)
        self.y = np.array(y)
        self.n = len(x)
        self.b0 = 0
        self.b1 = 0
        self.s = 0
        self.mse = 0
        if len(x) != len(y):
            print("Length of response vector doesn't match length of explanatory variable vector")

    def fit(self):
        """Fit model."""
        print("Fitting model...")
        self.b1 = (self.n * sum(self.x * self.y) - sum(self.x) * sum(self.y)) / \
                  (self.n * sum(self.x * self.x) - sum(self.x) ** 2)
        self.b0 = self.y.mean() - self.b1 * self.x.mean()
        print("Model fitted with coefficients: b0 = {0}, b1 = {1}".format(self.b0, self.b1))
        print("Calculating model errors...")
        pred = np.vectorize(self.predict)
        self.mse = (sum(self.y - pred(self.x)) ** 2) / (self.n - 2)
        self.s = math.sqrt(self.mse)
        print("Mean square error: {0}".format(self.mse))
        print("Residual standard error: {0}".format(self.s))

    def predict(self, x_var):
        return self.b0 + x_var * self.b1


if __name__ == "__main__":
    print("Simple linear regression demo.")
    reg_mod = SimpleLinearRegressionModel([1, 2, 332], [3.12, 2.324, 1.2134])
    reg_mod.fit()
    print("Trying predictions:")
    print(1, reg_mod.predict(1))
    print(2, reg_mod.predict(2))
    print(332, reg_mod.predict(332))
