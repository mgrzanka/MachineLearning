from src.LinearRegression.LinearRegressionModel import LinearRegressionModel
from GaltonFamiliesPreprocessing import get_regression_data
from get_scores import get_regression_error


X_train, X_test, y_train, y_test = get_regression_data()
model = LinearRegressionModel(0.5, 1000)
model.fit(X_train, y_train)
preds = model.predict(X_test)
error = get_regression_error(preds, y_test)
print("Error of linear regression: " + str(error))
