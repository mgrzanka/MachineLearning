from src.RandomForest.Tree import DecisionTree
from src.RandomForest.RandomForestModel import RandomForest
from get_scores import get_classification_accurancy
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
n_samples, n_features = X_train.shape
n = int(n_features**0.5)
num_trees = 50
max_samples = 10
max_depth = 4


forest = RandomForest(num_trees, n, max_samples, max_depth)
forest.fit(X_train, y_train)
preds = [forest.predict(x) for x in X_test]
accurancy = get_classification_accurancy(preds, y_test)
print(f"Accurancy of Random Forest wit {num_trees} trees: {accurancy}")


model = DecisionTree(n_features, max_samples, max_depth)
model.fit(X_train, y_train)
preds = [model.predict(x) for x in X_test]
accurancy = get_classification_accurancy(preds, y_test)
print(f"Accurancy of single Decision Tree: {accurancy}")
