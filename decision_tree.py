import numpy as np
import matplotlib.pyplot as plt

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value


class DecisionTreeRegressor:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.root = None

    def fit(self, X, y):
        self.root = self._grow_tree(X, y, depth=0)

    def _grow_tree(self, X, y, depth):
        num_samples, num_features = X.shape

        if depth == self.max_depth or num_samples <= 1:
            value = np.mean(y)
            return Node(value=value)

        best_feature, best_threshold = None, None
        best_variance_reduction = -float('inf')

        for feature_idx in range(num_features):
            thresholds = sorted(set(X[:, feature_idx]))

            for threshold in thresholds:
                left_indices = (X[:, feature_idx] <= threshold)
                right_indices = ~left_indices

                left_var = np.var(y[left_indices])
                right_var = np.var(y[right_indices])

                weighted_var = (left_var * sum(left_indices) + right_var * sum(right_indices)) / num_samples
                variance_reduction = np.var(y) - weighted_var

                if variance_reduction > best_variance_reduction:
                    best_variance_reduction = variance_reduction
                    best_feature = feature_idx
                    best_threshold = threshold

        left_indices = (X[:, best_feature] <= best_threshold)
        right_indices = ~left_indices

        left_subtree = self._grow_tree(X[left_indices], y[left_indices], depth + 1)
        right_subtree = self._grow_tree(X[right_indices], y[right_indices], depth + 1)

        return Node(feature=best_feature, threshold=best_threshold, left=left_subtree, right=right_subtree)

    def predict_single(self, sample):
        node = self.root
        while node.left:
            if sample[node.feature] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value


np.random.seed(42)
X_train = np.random.rand(1000, 3)  
y_train = np.sin(X_train[:, 0]) + np.cos(X_train[:, 1]) + np.tan(X_train[:, 2])

X_test = np.random.rand(200, 3)  
y_test = np.sin(X_test[:, 0]) + np.cos(X_test[:, 1]) + np.tan(X_test[:, 2])


max_depth = 5  
tree_regressor = DecisionTreeRegressor(max_depth=max_depth)
tree_regressor.fit(X_train, y_train)

predictions = [tree_regressor.predict_single(sample) for sample in X_test]

mse = np.mean((predictions - y_test) ** 2)
print(f"Mean Squared Error on Test Data: {mse}")

plt.figure(figsize=(8, 6))
plt.scatter(y_test, predictions, color='blue', label='Predicted vs. True')
plt.plot(y_test, y_test, color='red', linestyle='--', label='Perfect Prediction')

plt.title('Decision Tree Regressor: Predicted vs. True Values')
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.legend()
plt.grid(True)
plt.show()


