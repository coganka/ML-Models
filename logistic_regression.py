import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.bias = 0

        for _ in range(self.num_iterations):
            model = np.dot(X, self.weights) + self.bias
            predicted = self.sigmoid(model)

            dw = (1 / num_samples) * np.dot(X.T, (predicted - y))
            db = (1 / num_samples) * np.sum(predicted - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        model = np.dot(X, self.weights) + self.bias
        predicted_probabilities = self.sigmoid(model)
        predicted_labels = [1 if prob >= 0.5 else 0 for prob in predicted_probabilities]
        return predicted_labels
