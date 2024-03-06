import numpy as np
import matplotlib.pyplot as plt

class SimpleLinearRegression:
    def __init__(self):
        self.slope = None
        self.intercept = None

    def fit(self, X, y):
        mean_x = sum(X) / len(X)
        mean_y = sum(y) / len(y)
        
        numerator = sum((X[i] - mean_x) * (y[i] - mean_y) for i in range(len(X)))
        denominator = sum((X[i] - mean_x) ** 2 for i in range(len(X)))
        
        self.slope = numerator / denominator
        self.intercept = mean_y - self.slope * mean_x

    def predict(self, x):
        if self.slope is None or self.intercept is None:
            raise Exception("Model not fitted yet. call fit() first.")
        return self.slope * x + self.intercept

np.random.seed(42)

X_train = np.random.rand(100, 1) * 10  
y_train = 3 * X_train + 5 + np.random.randn(100, 1) * 2  

X_test = np.random.rand(50, 1) * 10 
y_test = 3 * X_test + 5 + np.random.randn(50, 1) * 2 

simple_lr = SimpleLinearRegression()

simple_lr.fit(X_train.squeeze(), y_train.squeeze())

predictions_test = np.array([simple_lr.predict(x) for x in X_test.squeeze()])

mse_test = np.mean((predictions_test - y_test.squeeze())**2)
print(f"Mean Squared Error on Test Data (Custom Model): {mse_test}")

plt.figure(figsize=(8, 6))
plt.scatter(X_test.squeeze(), y_test.squeeze(), color='blue', label='True Values')
plt.plot(X_test.squeeze(), predictions_test, color='red', label='Predicted Values')

plt.title('Simple Linear Regression: True vs. Predicted Values')
plt.xlabel('X Test')
plt.ylabel('Y Test')
plt.legend()
plt.grid(True)
plt.show()