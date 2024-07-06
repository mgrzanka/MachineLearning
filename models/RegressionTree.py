from src.RegressionTree.RegressionTreeModel import RegressionTree
import matplotlib.pyplot as plt
import numpy as np
from get_data import get_data
from get_scores import get_regression_error


X_train, X_test, y_train, y_test = get_data()
model = RegressionTree(7, 7)
model.fit(X_train, y_train)
preds = model.predict(X_test)
error = get_regression_error(preds, y_test)
print("Error of this model is: " + str(error))

# Plot
each_sample_error = [pred - y for pred, y in zip(preds, y_test)]
plt.plot(np.arange(0, len(X_test), 1), each_sample_error)
plt.title("Linear Regression error per sample")
plt.show()


# Prunning
print("Test pruning with validation set...:")
X_validate = X_test[:int(0.5*len(X_test))]
y_validate = y_test[:int(0.5*len(y_test))]
X_test = X_test[len(X_validate):]
y_test = y_test[len(X_validate):]

model = RegressionTree(max_samples=10, max_depth=10)
model.fit(X_train, y_train)
preds = model.predict(X_test)
error = get_regression_error(preds, y_test)
print("Error of this model is (before prunning): " + str(error))

model.prune(X_validate, y_validate)
preds_after_prunning = model.predict(X_test)
error_after_prunning = get_regression_error(preds_after_prunning, y_test)
print("Error of this model after pruning is (after prunning): " + str(error_after_prunning))
