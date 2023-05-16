import numpy as np
import matplotlib.pyplot as plt

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1-x2)**2))
     

class KMeans:
    def __init__(self, k=5, max_iters=100, plot_steps=False):
        self.k = k
        self.max_iters = max_iters
        self.plot_steps = plot_steps
        # A list of indices for each cluster
        self.clusters = [[] for _ in range(k)]
        # The centers (mean vector) of each cluster
        self.centroids = []


    # No fit method (no y) because K-means clustering is an unsupervised learning technique
    def predict(self, X):
        self.X = X
        self.n_samples, self.n_features = X.shape
        random_samples_indices = np.random.choice(self.n_samples, self.k, replace=False)
        self.centroids = [self.X[index] for index in random_samples_indices]
        for _ in range(self.max_iters):
            self.clusters = self._create_clusters(self.centroids)
            if self.plot_steps:
                self.plot()
            old_centroids = self.centroids
            self.centroids = self._get_centroids(self.clusters)
            if self._is_converged(self.centroids, old_centroids):
                break
            if self.plot_steps:
                self.plot()
        return self._get_cluster_labels(self.clusters)
    

    def _get_cluster_labels(self, clusters):
        # each sample will get the label of the cluster it was assigned to
        labels = np.empty(self.n_samples)
        for cluster_index, cluster in enumerate(clusters):
            for sample_index in cluster:
                labels[sample_index] = cluster_index
        return labels


    def _create_clusters(self, centroids):
        # assign each sample to the closest centroid
        clusters = [[] for _ in range(self.k)]
        for index, sample in enumerate(self.X):
            centroid_index = self._closest_centroid(sample, centroids)
            clusters[centroid_index].append(index)
        return clusters
    

    def _closest_centroid(self, sample, centroids):
        distances = [euclidean_distance(sample, centroid) for centroid in centroids]
        closest_centroid_index = np.argmin(distances)
        return closest_centroid_index
    

    def _get_centroids(self, clusters):
        # calculate centroids (the mean value of each cluster)
        centroids = np.zeros((self.k, self.n_features))
        for cluster_index, cluster in enumerate(clusters):
            cluster_mean = np.mean(self.X[cluster], axis=0)
            centroids[cluster_index] = cluster_mean
        return centroids
    

    def _is_converged(self, centroids_old, centroids):
        distances = [euclidean_distance(centroids_old[i], centroids[i]) for i in range(self.k)]
        return sum(distances) == 0
    

    def plot(self):
        fig, ax = plt.subplots(figsize=(12, 8))
        for i, index in enumerate(self.clusters):
            point = self.X[index].T
            ax.scatter(*point)
        for centroid in self.centroids:
            ax.scatter(*centroid, marker='x', color='black', linewidth=2)
        plt.show()
