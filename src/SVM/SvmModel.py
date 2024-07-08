import numpy as np
from cvxopt import matrix, solvers


class PrimalSVM:
    def __init__(self, learning_rate, iterations, lambda_param):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.lambda_param = lambda_param
        self.min_X = self.max_X = None
        self.weights = self.bias = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        n_samples, n_features = X.shape
        Y = np.where(y <= 0, -1, 1)
        X = self.scale_data(X)

        self.weights = np.zeros(shape=(n_features,))
        self.bias = 0

        for t in range(self.iterations):
            for x, y in zip(X, Y):
                condition = y*(np.dot(x, self.weights) + self.bias) >= 1
                if condition:
                    self.weights -= self.learning_rate * (2*self.lambda_param*self.weights)
                else:
                    self.weights -= self.learning_rate * (2*self.lambda_param*self.weights - np.dot(x, y))
                    self.bias -= self.learning_rate * y * -1
            print(f"{t}/{self.iterations} training iteration ended")

    def predict(self, x: np.ndarray):
        x = (x - self.min_X)/(self.max_X-self.min_X+1e-8)
        return np.sign(np.dot(x, self.weights) + self.bias)

    def scale_data(self, X: np.ndarray):
        self.min_X = np.min(X, axis=0)
        self.max_X = np.max(X, axis=0)
        return (X-self.min_X)/(self.max_X-self.min_X+1e-8)


class SVM:
    def __init__(self, r=1, d=2, C=1, kernel="linear"):
        self.support_vectors_alpha = self.bias = None
        self.support_vectors_x = self.support_vectors_y = None
        self.min_X = self.max_X = None
        self.kernel = kernel
        self.C = C
        self.r = r
        self.d = d

    def lineal_kernel(self, X1: np.ndarray, X2: np.ndarray):
        return np.dot(X1, X2.T)

    def polynomial_kernel(self, X1: np.ndarray, X2: np.ndarray):
        return np.power(np.dot(X1, X2.T)+self.r, self.d)

    def gauss_kernel(self, X1: np.ndarray, X2: np.ndarray):
        pass

    def kernel_function(self, X1: np.ndarray, X2: np.ndarray):
        if self.kernel == "linear":
            return self.lineal_kernel(X1, X2)
        elif self.kernel == "polynomial":
            return self.polynomial_kernel(X1, X2)
        elif self.kernel == "gauss":
            return self.gauss_kernel(X1, X2)
        else:
            raise ValueError("Wrong kernel name. Use either linear, polynomial or gauss")

    def QP_optimization(self, X: np.ndarray, Y: np.ndarray):
        n_samples, n_features = X.shape

        K = self.kernel_function(X, X)
        P = matrix(np.outer(Y, Y) * K)
        q = matrix(np.ones(shape=(n_samples, 1)) * -1.0)

        G_std = np.eye(n_samples) * -1.0
        h_std = np.zeros(shape=(n_samples, 1))
        G_slack = np.eye(n_samples)
        h_slack = np.full(shape=(n_samples, 1), fill_value=self.C)
        G = matrix(np.vstack((G_std, G_slack)))
        h = matrix(np.vstack((h_std, h_slack)))

        A = matrix(Y.astype(np.float64), (1, n_samples))
        b = matrix(0.0)

        solution = solvers.qp(P, q, G, h, A, b)
        best_alpha = np.array(solution['x']).flatten()
        return best_alpha

    def get_support_vectors(self, X: np.ndarray, Y: np.ndarray, alpha: np.ndarray):
        support_vectors = (alpha > 1e-5) & (alpha < self.C)
        return X[support_vectors], Y[support_vectors], alpha[support_vectors]

    def get_bias(self, X: np.ndarray, Y: np.ndarray, alpha: np.ndarray):
        sum_b = 0
        count = 0
        for x, y in zip(self.support_vectors_x, self.support_vectors_y):
            temp = np.sum(self.support_vectors_alpha * self.support_vectors_y *
                          self.kernel_function(x, self.support_vectors_x))
            sum_b += y - temp
            count += 1
        return sum_b/count

    def fit(self, X: np.ndarray, y: np.ndarray):
        Y = np.where(y <= 0, -1, 1)
        X = self.scale_data(X)

        alpha = self.QP_optimization(X, Y)
        self.support_vectors_x, self.support_vectors_y, self.support_vectors_alpha = self.get_support_vectors(X, Y, alpha)
        self.bias = self.get_bias(X, Y, alpha)

    def predict(self, x: np.ndarray):
        x = (x-self.min_X)/(self.max_X-self.min_X+1e-8)
        prediction_value = np.sum(self.support_vectors_alpha*self.support_vectors_y*self.kernel_function(x, self.support_vectors_x)) + self.bias
        return np.sign(prediction_value)

    def scale_data(self, X: np.ndarray):
        self.min_X = np.min(X, axis=0)
        self.max_X = np.max(X, axis=0)
        return (X-self.min_X)/(self.max_X-self.min_X+1e-8)
