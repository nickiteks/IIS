import pandas as p

data = p.read_csv('spotify_dataset.csv')

''' --Keys--
 0   Index                      1556 non-null   int64 
 1   Highest Charting Position  1556 non-null   int64 
 2   Number of Times Charted    1556 non-null   int64 
 3   Week of Highest Charting   1556 non-null   object
 4   Song Name                  1556 non-null   object
 5   Streams                    1556 non-null   object
 6   Artist                     1556 non-null   object
 7   Artist Followers           1556 non-null   object
 8   Song ID                    1556 non-null   object
 9   Genre                      1556 non-null   object
 10  Release Date               1556 non-null   object
 11  Weeks Charted              1556 non-null   object
 12  Popularity                 1556 non-null   object
 13  Danceability               1556 non-null   object
 14  Energy                     1556 non-null   object
 15  Loudness                   1556 non-null   object
 16  Speechiness                1556 non-null   object
 17  Acousticness               1556 non-null   object
 18  Liveness                   1556 non-null   object
 19  Tempo                      1556 non-null   object
 20  Duration (ms)              1556 non-null   object
 21  Valence                    1556 non-null   object
 22  Chord                      1556 non-null   object
'''
keys = data.keys()


def count_sum_streams():
    sum = 0
    for i in data[keys[5]]:
        number = i.replace(',', '')
        sum += int(number)

    return sum


def average_followers():
    sum = 0
    for i in range(len(data[keys[7]])):
        if data[keys[7]][i] == ' ':
            sum += 1
        else:
            sum += int(data[keys[7]][i])
    return round(sum/len(data[keys[7]]),2)


print(count_sum_streams())
print(average_followers())