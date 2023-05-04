import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from main import LogisticRegression
from main import binary_cross_entropy

b_cancer = datasets.load_breast_cancer()
X, y = b_cancer.data, b_cancer.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

log_reg_model = LogisticRegression()
log_reg_model.fit(X_train, y_train)
y_pred = log_reg_model.predict(X_test) 

def accuracy(y, y_pred):
    return np.sum(y == y_pred) / len(y)

print(accuracy(y_test, y_pred))