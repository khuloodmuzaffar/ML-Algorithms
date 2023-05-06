import numpy as np

class NaiveBayes:
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self._classes = np.unique(y)
        n_classes = len(self._classes)

        # Calculate the mean, variance, and prior for each class
        # Prior is the frequency of a particular class in X
        self._mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self._var = np.zeros((n_classes, n_features), dtype=np.float64)
        self._prior = np.zeros(n_classes, dtype=np.float64)

        for index, cls in enumerate(self._classes):
            X_c = X[y == cls]
            self._mean[index, :] = X_c.mean(axis=0)
            self._var[index, :] = X_c.var(axis=0)
            self._prior[index] = X_c.shape[0] / float(n_samples)


    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)
    

    def _predict(self, x):
        posteriors = []

        # Calculate the posterior probability for each class
        for index, cls in enumerate(self._classes):
            log_prior = np.log(self._prior[index])
            posterior = np.sum(np.log(self._probability_density_func(index, x)))
            posterior += log_prior
            posteriors.append(posterior)

        # Return class with the highest posterior
        return self._classes[np.argmax(posteriors)]
    

    def _probability_density_func(self, index, x):
        mean = self._mean[index]
        variance = self._var[index]

         # Calculate P(x|y) using the gaussian distribution
        p = 1 / np.sqrt(2 * np.pi * variance)
        p = p * np.exp(-((x - mean) ** 2)/(2 * variance))
        return p

