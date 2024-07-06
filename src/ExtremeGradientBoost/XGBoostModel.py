import numpy as np
from .XGboostTree import Node, XGboostTree, XGboostTreeRegression, XGboostTreeClassification


class XGboost:
    def __init__(self, trees_number: int, learning_rate: float, regularization_param: int, prune_constant: int, max_depth: int,
                 min_cover: float):
        self.trees_number = trees_number
        self.learning_rate = learning_rate
        # constans for trees
        self.lambd = regularization_param
        self.gamma = prune_constant
        self.max_depth = max_depth
        self.min_cover = min_cover
        self.trees = []

    def create_tree(self, X: np.ndarray, y: np.ndarray, previous_predictions=None, first=False):
        if not self.classification:
            tree = XGboostTreeRegression(self.lambd, self.gamma, self.max_depth)
            if first:
                tree.root = Node(value=0.5, is_leaf=True)
            else:
                tree.fit(X, y, previous_predictions)
                tree.prune()
        else:
            tree = XGboostTreeClassification(self.lambd, self.gamma, self.max_depth, self.min_cover)
            if first:
                tree.root = Node(value=0.0, is_leaf=True)
            else:
                tree.fit(X, y, previous_predictions)
                tree.prune()
        self.trees.append(tree)

    def get_pseudo_residuals(self, X: np.ndarray, y: np.ndarray):
        prediction = self.predict(X, final=False)
        return y - prediction, prediction

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.classification = True if len(np.unique(y)) < 5 else False

        self.create_tree(X, y, first=True)
        pseudo_residuals, prediction = self.get_pseudo_residuals(X, y)
        print(f"1/{self.trees_number} XTree trained")

        for tree_index in range(self.trees_number-1):
            self.create_tree(X, pseudo_residuals, prediction)
            pseudo_residuals, prediction = self.get_pseudo_residuals(X, y)
            print(f"{tree_index+2}/{self.trees_number} XTree trained")

        self.trees = np.array(self.trees)

    def predict(self, X: np.ndarray, final=True):
        total_prediction = self.trees[0].predict(X)
        for tree in self.trees[1:]:
            total_prediction += self.learning_rate * tree.predict(X)
        if self.classification:
            probability = np.exp(total_prediction)/(1+np.exp(total_prediction))
            return probability if not final else probability > 0.5
        else:
            return total_prediction
