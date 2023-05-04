import numpy as np

def sigmoid(y):
    return (1 / (1 + np.exp(-y))) 
   
def linear_prediction(X, weights, bias):
    return (np.dot(X, weights) + bias)

class LogisticRegression:
    def __init__(self, lr=0.001, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None


    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            lin_y_pred  = linear_prediction(X, self.weights, self.bias)
            y_pred = sigmoid(lin_y_pred)

            dw = (2/n_samples) * (np.dot(X.T, (y_pred - y)))
            db = (2/n_samples) * (np.sum(y_pred - y))

            self.weights -= self.lr * dw
            self.bias -= self.lr * db


    def predict(self, X):
        lin_y_pred  = linear_prediction(X, self.weights, self.bias)
        y_pred = sigmoid(lin_y_pred)

        class_pred = [0 if y <= 0.5 else 1 for y in y_pred]
        return class_pred