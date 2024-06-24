from src.LinearRegression.LinearRegressionModel import LinearRegressionModel
import matplotlib.pyplot as plt
import numpy as np
from get_data import get_data
from get_scores import get_regression_error


X_train, X_test, y_train, y_test = get_data()
model = LinearRegressionModel(0.5, 3000)
model.fit(X_train, y_train)
preds = [model.predict(x) for x in X_test]
error = get_regression_error(preds, y_test)
print("Error of this model is: " + str(error))

each_sample_error = [pred - y for pred, y in zip(preds, y_test)]
plt.plot(np.arange(0, len(X_test), 1), each_sample_error)
plt.title("Linear Regression error per sample")
plt.show()
