import numpy as np
from collections import Counter

def euclidean_distance(x1, x2):
    distance = np.sqrt(np.sum((x1-x2)**2))
    return distance

class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return predictions
        pass

    def _predict(self, x):
        distances = [euclidean_distance(x, x_t) for x_t in self.X_train]

        knn_indices = np.argsort(distances)[:self.k]
        knn_labels = [self.y_train[index] for index in knn_indices]
        
        y_pred = Counter(knn_labels).most_common(1)[0][0]
        return y_pred