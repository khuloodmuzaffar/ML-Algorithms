from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import datasets
from main import NaiveBayes
import numpy as np

def accuracy(y, y_pred):
    accuracy = np.sum(y==y_pred) / len(y)
    return accuracy

X, y = datasets.make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=123)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

nb_model = NaiveBayes()
nb_model.fit(X_train, y_train)
y_pred = nb_model.predict(X_test)

acc = accuracy(y_test, y_pred)
print(acc)


