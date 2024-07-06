from src.GradientBoosting.GradientBoostingModel import GBoost
from get_scores import get_regression_error, get_classification_accurancy
from get_data import get_data
from sklearn.datasets import fetch_california_housing, load_breast_cancer, make_moons
from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = get_data()

model = GBoost(10, 0.1, 5, 10)
model.fit(X_train, y_train)
preds = model.predict(X_test)
error = get_regression_error(preds, y_test)
print("Regression problem with GBoost: " + str(error) + " error")


# data = load_breast_cancer()
# X, y = data.data, data.target
X, y = make_moons()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = GBoost(100, 0.1, 5, 10)
model.fit(X_train, y_train)
preds = model.predict(X_test)
accurancy = get_classification_accurancy(preds, y_test)
print("Classification problem with GBoost: " + str(accurancy) + " accurancy")
