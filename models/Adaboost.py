from src.Adaboost.AdaboostModel import Adaboost
from sklearn.datasets import load_breast_cancer, make_moons, load_iris
from sklearn.model_selection import train_test_split
from get_scores import get_classification_accurancy


data = make_moons(noise=0.2)
X, y = data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = Adaboost(100)
model.fit(X_train, y_train)
preds = [model.predict(x) for x in X_test]
accurancy = get_classification_accurancy(preds, y_test)
print(accurancy)
