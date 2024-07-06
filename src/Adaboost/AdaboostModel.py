from .Stump import Stump
import numpy as np


class Adaboost:
    def __init__(self, stumps_number: int):
        self.stumps_number = stumps_number
        self.stumps = np.array([Stump() for _ in range(stumps_number)])
    
    def update_dataset(self, weights: np.ndarray, X: np.ndarray, y: np.ndarray):
        indexes = np.arange(stop=len(y), step=1)
        picked_indexes = np.random.choice(indexes, p=weights, replace=True, size=len(y))
        return X[picked_indexes], y[picked_indexes]

    def calculate_weights(self, stump: Stump, current_weights: np.ndarray, X: np.ndarray, y: np.ndarray):
        new_weights = np.zeros(current_weights.shape)
        predictions = [stump.predict(x) for x in X]
        incorrect = predictions != y
        total_error = np.dot(current_weights, incorrect)
        stump.amount_of_say = 0.5 * np.log((1-total_error+1e-8)/(total_error+1e-8))
        
        index = 0
        for prediction, target, weight in zip(predictions, y, current_weights):
            if prediction != target:
                new_weight = weight * np.exp(stump.amount_of_say)
            else:
                new_weight = weight * np.exp(-stump.amount_of_say)
            new_weights[index] = new_weight
            index += 1
        
        return new_weights/np.sum(new_weights)

    def fit(self, X: np.ndarray, y: np.ndarray):
        weights = np.full(len(y), 1/len(y))
        new_X = X
        new_y = y
        for stump_index, stump in enumerate(self.stumps):
            stump.train(new_X, new_y)
            print(f"{stump_index+1}/{self.stumps_number} stump trained")
            
            iteration = 0
            weights = self.calculate_weights(stump, weights, X, y)
            new_X, new_y = self.update_dataset(weights, X, y)
            while len(np.unique(new_y)) == 1:
                if iteration < 10:
                    new_X, new_y = self.update_dataset(weights, X, y)
                else:
                    return

    def predict(self, x: np.ndarray):
        predictions = {}
        for stump in self.stumps:
            pred = stump.predict(x)
            predictions[pred] = predictions.get(pred, 0) + stump.amount_of_say
        return max(predictions, key=predictions.get)
