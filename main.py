import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Даныне фильмов
metadata = pd.read_csv('movies_metadata.csv', low_memory=False)

# Формула для расчета рейтинга WR = ((v/(v+m)) * R )+( (m/(v+m)) * C )
# v = vote_count
# R = vote_average
# m - Установитьь процентаж

C = metadata['vote_average'].mean()
m = metadata['vote_count'].quantile(0.90)


def weighted_rating(x, m=m, C=C):
    v = x['vote_count']
    R = x['vote_average']
    # Calculation based on the IMDB formula
    return ((v / (v + m)) * R) + ((m / (v + m)) * C)


# Новый датаФрейм для того что бы не испортить стартоввый
q_movies = metadata.copy().loc[metadata['vote_count'] >= m]
q_movies['score'] = q_movies.apply(weighted_rating, axis=1)
q_movies = q_movies.sort_values('score', ascending=False)

tfidf = TfidfVectorizer(stop_words='english')

metadata['overview'] = metadata['overview'].fillna('')

tfidf_matrix = tfidf.fit_transform(metadata['overview'])
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

indices = pd.Series(metadata.index, index=metadata['title']).drop_duplicates()


def get_recommendations(title, cosine_sim=cosine_sim):
    idx = indices[title]

    sim_scores = list(enumerate(cosine_sim[idx]))

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    sim_scores = sim_scores[1:11]

    movie_indices = [i[0] for i in sim_scores]

    return metadata['title'].iloc[movie_indices]


print(get_recommendations('Spider-Man'))
