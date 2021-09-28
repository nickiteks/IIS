import numpy as np
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier


class Manager:

    def __init__(self, data):
        self.data = data
        self.keys = data.keys()

    def word_to_number(self, word):
        result = 1
        for letter in word:
            result += ord(letter) + result % 10
        return result

    # удаление шумов и преобразование данных
    def data_preparation(self):

        for i in range(len(self.data[self.keys[3]])):
            self.data[self.keys[3]].update(
                self.data[self.keys[3]].replace(
                    self.data[self.keys[3]][i], self.word_to_number(self.data[self.keys[3]][i])))

        self.data[self.keys[4]] = self.data[self.keys[4]].factorize()[0]
        self.data[self.keys[5]].update(self.data[self.keys[5]].replace(np.nan, self.data[self.keys[5]].median()))
        return self.data

    # тестирование
    def test_tree(self, X_train, y_train, X_test, y_test):
        rfc = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=21)
        rfc.fit(X_train, y_train)
        print(rfc.score(X_test, y_test))

    # обучение
    def learn_tree(self, X_train, y_train):
        clf = tree.DecisionTreeClassifier(max_depth=5, random_state=21)
        clf.fit(X_train, y_train)
        print(clf.score(X_train, y_train))
        return clf
