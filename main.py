import numpy as np
import pandas as p
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt
import graphviz
from sklearn.metrics import classification_report, confusion_matrix

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
data[keys[5]].update(data[keys[5]].replace(np.nan, 0))
data[keys[4]] = data[keys[4]].factorize()[0]

X = data.drop([keys[0], keys[1], keys[2], keys[3], keys[6], keys[7], keys[8], keys[9], keys[10], keys[11]], axis=1)
y = data[keys[1]]

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, y)

dot_data = tree.export_graphviz(clf, out_file='file')
graph = graphviz.Source(dot_data)

tree.plot_tree(clf)
plt.show()
