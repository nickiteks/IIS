import pandas as p

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


def count_survived():
    survived = 0

    for i in data[keys[1]]:
        if i == 1:
            survived = survived + 1

    return survived


result = round(count_survived() / len(data[keys[1]]) * 100, 2)

print(result)
