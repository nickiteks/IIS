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


def word_to_number(word):
    result = 1
    for letter in word:
        result += ord(letter) + result % 10
    return result


# удаление шумов и преобразование данных
def data_preparation():
    for i in range(len(data[keys[3]])):
        data[keys[3]].update(data[keys[3]].replace(data[keys[3]][i], word_to_number(data[keys[3]][i])))
    data[keys[4]] = data[keys[4]].factorize()[0]
    data[keys[5]].update(data[keys[5]].replace(np.nan, data[keys[5]].median()))


# тестирование
def test_tree():
    rfc = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=21)
    rfc.fit(X_train, y_train)
    print(rfc.score(X_test, y_test))


# обучение
def learn_tree():
    clf = tree.DecisionTreeClassifier(max_depth=5, random_state=21)
    clf.fit(X_train, y_train)
    print(clf.score(X_train, y_train))
    return clf


data_preparation()

# отделение нужных данных
X = data.drop([keys[0], keys[1], keys[2], keys[6], keys[7], keys[8], keys[9], keys[10], keys[11]], axis=1)
y = data[keys[1]]
print(X.head())
# выборка данных для обучения и тестирования
X_train = X[:-200]
X_test = X[-200:]
y_train = y[:-200]
y_test = y[-200:]

clf = learn_tree()
test_tree()

# отпределение веса
print(eli5.explain_weights_sklearn(clf, feature_names=X_train.columns.values))
