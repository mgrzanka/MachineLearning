from src.KNearestNeighbours.KNearestNeighboursModel import KNearestNeighbours
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from get_scores import get_classification_accurancy


data = load_breast_cancer()
X = data.data
y = data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = KNearestNeighbours(4)
model.fit(X_train, y_train)
preds = [model.predict(x) for x in X_test]
accurancy = get_classification_accurancy(preds, y_test)
print(accurancy)
