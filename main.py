import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_circles
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

data = make_circles(noise=0.2, factor=0.5, random_state=1)

X = np.array(data[0])
y = np.array(data[1])

X = X[:, np.newaxis, 1]

print(X)

X_train = X[:-20]
X_test = X[-20:]
y_train = y[:-20]
y_test = y[-20:]

model = LinearRegression().fit(X_train, y_train)
y_pred = model.predict(X_test)


print(model.intercept_)
print(model.coef_)

plt.scatter(X_test, y_test,  color='black')
plt.plot(X_test, y_pred, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()