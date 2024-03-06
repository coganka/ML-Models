import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

np.random.seed(42)

X_train = np.random.rand(100, 2) * 10 
y_train = np.random.randint(0, 2, size=100)  

X_test = np.random.rand(50, 2) * 10 
y_test = np.random.randint(0, 2, size=50)  

class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        predictions = []
        for sample in X_test:
            distances = [np.linalg.norm(sample - x) for x in self.X_train]
            k_indices = np.argsort(distances)[:self.k]
            k_nearest_labels = [self.y_train[i] for i in k_indices]
            most_common = max(set(k_nearest_labels), key=k_nearest_labels.count)
            predictions.append(most_common)
        return predictions

knn = KNN(k=3)

knn.fit(X_train, y_train)

predictions = knn.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy of KNN on Test Data: {accuracy:.4f}")

plt.figure(figsize=(8, 6))

plt.scatter(X_test[y_test == 0][:, 0], X_test[y_test == 0][:, 1], color='blue', label='Class 0 (True)')
plt.scatter(X_test[y_test == 1][:, 0], X_test[y_test == 1][:, 1], color='red', label='Class 1 (True)')

predictions = np.array(predictions)

plt.scatter(X_test[predictions == 0][:, 0], X_test[predictions == 0][:, 1], marker='x', color='cyan', label='Class 0 (Predicted)')
plt.scatter(X_test[predictions == 1][:, 0], X_test[predictions == 1][:, 1], marker='x', color='orange', label='Class 1 (Predicted)')

plt.title('KNN: True vs. Predicted Classes')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True)
plt.show()