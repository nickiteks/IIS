from operator import itemgetter

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


def save_file(model, filename):
    joblib.dump(model, filename)


def load_file(filename):
    return joblib.load(filename)


# Данные фильмов
metadata = pd.read_csv('movies_metadata.csv', low_memory=False)

tfidf = TfidfVectorizer(stop_words='english')

metadata['overview'] = metadata['overview'].fillna('')

tfidf_matrix = tfidf.fit_transform(metadata['overview'])

cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

save_file(cosine_sim,'model.pkl')

indices = metadata['title'].drop_duplicates().reset_index().set_index('title')['index']


# высчитивание похожих фильмов на основе описания фильмов
def get_recommendations(title, cosine_sim=cosine_sim):
    idx = indices[title]

    sim_scores = list(enumerate(cosine_sim[idx]))

    sim_scores = sorted(sim_scores, key=itemgetter(1), reverse=True)

    sim_scores = sim_scores[1:11]

    movie_indices = [i[0] for i in sim_scores]

    return metadata['title'].iloc[movie_indices]


print(get_recommendations('The Avengers'))
