import numpy as np
from sklearn.datasets import make_circles
from Models import Models

models_manager = Models()

data = make_circles(noise=0.2, factor=0.5, random_state=1)

X = np.array(data[0])
y = np.array(data[1])

X = X[:, np.newaxis, 1]

X_train = X[:-20]
X_test = X[-20:]
y_train = y[:-20]
y_test = y[-20:]

models_manager.linear(X_train, X_test, y_train, y_test)

models_manager.polynomial(X_train, y_train)

models_manager.ridge(X_train, X_test, y_train, y_test)