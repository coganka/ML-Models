import numpy as np

class SimpleSVR:
    def __init__(self, epsilon=0.1, C=1.0, learning_rate=0.01, num_iterations=1000):
        self.epsilon = epsilon
        self.C = C
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.w = None
        self.b = None

    def fit(self, X, y):
        num_samples, num_features = X.shape

        self.w = np.zeros(num_features)
        self.b = 0

        for _ in range(self.num_iterations):
            for idx, x_i in enumerate(X):
                condition = abs(np.dot(x_i, self.w) + self.b - y[idx]) <= self.epsilon

                if condition:
                    self.w -= self.learning_rate * (2 * self.C * self.w)
                else:
                    error = np.dot(x_i, self.w) + self.b - y[idx]
                    self.w -= self.learning_rate * (2 * self.C * self.w - error * x_i)
                    self.b -= self.learning_rate * error

    def predict(self, X):
        return np.dot(X, self.w) + self.b