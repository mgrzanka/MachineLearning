from src.LogisticRegression.LogisticRegressionModel import LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from get_scores import get_classification_accurancy


data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression(0.5, 300)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
accurancy = get_classification_accurancy(predictions, y_test)
print(accurancy)
