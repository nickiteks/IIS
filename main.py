import numpy as np
import pandas as p
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
import eli5

data = p.read_csv('titanic.csv')

''' --Keys--
0 - PassengerId
1 - Survived
2 - Pclass
3 - Name
4 - Sex
5 - Age
6 - SibSp
7 - Ticket
8 - Fare
9 - Cabin
10 - Embarked
'''
keys = data.keys()

# удаление шумов
data[keys[4]] = data[keys[4]].factorize()[0]
data[keys[5]].update(data[keys[5]].replace(np.nan, data[keys[5]].median()))

# отделение нужных данных
X = data.drop([keys[0], keys[1], keys[2], keys[3], keys[6], keys[7], keys[8], keys[9], keys[10], keys[11]], axis=1)
y = data[keys[1]]

# выборка данных для обучения и тестирования
X_train = X[:-200]
X_test = X[-200:]
y_train = y[:-200]
y_test = y[-200:]

# обучение
clf = tree.DecisionTreeClassifier(max_depth=5, random_state=21)
clf.fit(X_train, y_train)
print(clf.score(X_train, y_train))

# тестирование
rfc = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=21)
rfc.fit(X_train, y_train)
print(rfc.score(X_test, y_test))

# отпределение веса
print(eli5.explain_weights_sklearn(clf, feature_names=X_train.columns.values))
