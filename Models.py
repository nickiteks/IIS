from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures


class Models:

    def linear(self, X_train, X_test, y_train, y_test):
        model = LinearRegression().fit(X_train, y_train)
        y_pred = model.predict(X_test)

        print('___Линейная регрессия___')
        print(model.intercept_)
        print(model.coef_)
        print('')

        plt.scatter(X_test, y_test, color='black')
        plt.plot(X_test, y_pred, color='blue', linewidth=3)
        plt.show()

    def polynomial(self, X, y):
        poly_reg = PolynomialFeatures(degree=3)
        X_poly = poly_reg.fit_transform(X)
        pol_reg = LinearRegression()
        pol_reg.fit(X_poly, y)

        print('___Полиномиальная  регрессия___')
        print(pol_reg.intercept_)
        print(pol_reg.coef_)
        print('')

        plt.scatter(X, y, color='red')
        plt.plot(X, pol_reg.predict(poly_reg.fit_transform(X)), color='blue')
        plt.title('Truth or Bluff (Linear Regression)')
        plt.xlabel('Position level')
        plt.ylabel('Salary')
        plt.show()

    def ridge(self, X_train, X_test, y_train, y_test):
        clf = Ridge(alpha=1.0)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        print('___Гребневая полиномиальная регрессия___')
        print(clf.intercept_)
        print(clf.coef_)

        plt.scatter(X_test, y_test, color='black')
        plt.plot(X_test, y_pred, color='blue', linewidth=3)
        plt.show()
