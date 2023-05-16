from main import KMeans
from sklearn.datasets import make_blobs
import numpy as np


np.random.seed(42)
X, y = make_blobs(centers=3, n_samples=500, n_features=2, shuffle=True, random_state=40)
clusters = len(np.unique(y))

model = KMeans(k = clusters, max_iters=150, plot_steps=True)
y_pred = model.predict(X)
model.plot()