from src.ExtremeGradientBoost.XGBoostModel import XGboost
from get_scores import get_regression_error, get_classification_accurancy
from get_data import get_data
from sklearn.datasets import fetch_california_housing, load_breast_cancer, make_moons
from sklearn.model_selection import train_test_split


# Regression
X_train, X_test, y_train, y_test = get_data()
model = XGboost(10, 0.9, 1, 1, 3, 0.5)
model.fit(X_train, y_train)
preds = model.predict(X_test)
error = get_regression_error(preds, y_test)
print("XGBoost with regression error: " + str(error))


# Classification
model = XGboost(100, 0.9, 1, 1, 4, 0.5)
data = make_moons()
X, y = data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)
preds = model.predict(X_test)
accurancy = get_classification_accurancy(preds, y_test)
print("XGBoost with classification accurancy: " + str(accurancy))
