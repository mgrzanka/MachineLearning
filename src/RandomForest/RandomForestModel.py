from .Tree import DecisionTree
import numpy as np


class RandomForest:
    def __init__(self, number_of_trees, number_of_features, max_samples, max_depth):
        self.number_of_trees = number_of_trees
        self.trees = np.array([DecisionTree(number_of_features, max_samples, max_depth) for _ in range(number_of_trees)])

    def bootstrap(self, X: np.ndarray, y: np.ndarray):
        n_samples = X.shape[0]
        indexes = np.random.choice(n_samples, size=(n_samples,), replace=True)
        return X[indexes], y[indexes]

    def fit(self, X: np.ndarray, y: np.ndarray):
        for indx, tree in enumerate(self.trees):
            data_x, data_y = self.bootstrap(X, y)
            tree.fit(data_x, data_y)
            print(f"{indx+1}/{self.number_of_trees} tree created")

    def predict(self, x: np.ndarray):
        predictions = [model.predict(x) for model in self.trees]
        y, y_counts = np.unique(predictions, return_counts=True)
        return y[np.argmax(y_counts)]
