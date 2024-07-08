import numpy as np


class KNearestNeighbours:
    def __init__(self, k):
        self.k = k
        self.data = None
        self.mean_X = self.std_X = None

    def calculate_distance(self, training_sample: np.ndarray, to_predict: np.ndarray):
        return (np.sum((training_sample-to_predict)**2))**0.5

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.data = list(zip(self.scale_data(X), y))

    def predict(self, x: np.ndarray):
        x = (x-self.mean_X)/self.std_X
        sorted_data = sorted(self.data, key=lambda data: self.calculate_distance(data[0], x))
        y, y_counts = np.unique(list(map(lambda row: row[1], sorted_data))[:self.k], return_counts=True)
        return y[np.argmax(y_counts)]

    def scale_data(self, X: np.ndarray):
        self.mean_X = np.mean(X, axis=0)
        self.std_X = np.std(X, axis=0)
        return (X-self.mean_X)/self.std_X
