import numpy as np
import matplotlib.pyplot as plt

class KMeans:
    def __init__(self, n_clusters=3, max_iters=100):
        self.n_clusters = n_clusters
        self.max_iters = max_iters

    def fit(self, X):
        self.centroids = X[np.random.choice(X.shape[0], self.n_clusters, replace=False)]
        
        for _ in range(self.max_iters):
            clusters = self._assign_clusters(X)
            
            new_centroids = []
            for i in range(self.n_clusters):
                cluster_mean = np.mean(X[clusters == i], axis=0)
                new_centroids.append(cluster_mean)
            self.centroids = np.array(new_centroids)
    
    def _assign_clusters(self, X):
        distances = np.linalg.norm(X[:, None] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)
    
    def predict(self, X):
        distances = np.linalg.norm(X[:, None] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)

np.random.seed(42)
X = np.random.rand(100, 2) * 10  

kmeans = KMeans(n_clusters=3, max_iters=100)

kmeans.fit(X)

predictions = kmeans.predict(X)

print(predictions)

plt.figure(figsize=(8, 6))

for i in range(kmeans.n_clusters):
    cluster_points = X[predictions == i]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {i}')

plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], marker='x', color='black', label='Centroids')

plt.title('K-Means Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True)
plt.show()