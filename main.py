import pandas as pd
import random as r
import names
from matplotlib import pyplot as plt

list_eyes_color = ['Красные', 'Фиолетовые', 'Черные']

data = pd.DataFrame({'Длина носа': [r.randint(1, 20) for i in range(100)],
                     'Размер ноги': [r.randint(1, 30) for i in range(100)],
                     'Имя': [names.get_full_name() for i in range(100)],
                     'Пиво буиш?': [r.choice([True, False]) for i in range(100)],
                     'Цвет глаз': [r.choice(list_eyes_color) for i in range(100)]})

data.to_csv('file.csv')

a = data['Длина носа'].min()
b = data['Длина носа'].max()
c = data['Длина носа'].mean()
d = data['Длина носа'].median()

print(a, b, c, d)


def min_cl():
    min = data['Длина носа'][0]
    for i in range(len(data['Длина носа'])):
        if data['Длина носа'][i] < min:
            min = data['Длина носа'][i]
    return min


print(min_cl())

fig, ax = plt.subplots()
ax.set_title('Длина носа')
ax.set_xlabel('Люди')
ax.set_ylabel('Длина')
plt.plot(data['Длина носа'])
plt.plot(data['Размер ноги'])
plt.grid('on')
plt.show()