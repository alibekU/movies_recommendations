import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import time
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from global_parameters import *

# assign values from global_parameters.py to local variables
number_movies_returned  = global_number_movies_returned
min_movies_rated_by_user = global_min_movies_rated_by_user
min_users_rated_movie = global_min_users_rated_movie
global_database_filepath = global_database_filepath
table_name = global_recs_table_name

def preprocess_movies(movies_df):
    movies_df.loc[movies_df['movie_title'] == 'Gisaengchung (2019)', 'movie_title'] = 'Parasite (Gisaengchung) (2019)'
    movies_df.loc[movies_df['movie_title'] == 'Vaiana (2016)', 'movie_title'] = 'Moana (Vaiana) (2016)'
    pattern_amp = ['&amp;', '&']
    movies_df.loc[:, 'movie_title'] = movies_df['movie_title'].str.replace('|'.join(pattern_amp), 'and')
    return movies_df 

movies = pd.read_csv('data/movies.dat', sep='::', names=['movie_id', 'movie_title', 'genra'], header=None, engine='python')
movies.dropna(inplace=True)
movies.drop_duplicates(subset=['movie_id'], inplace=True)


movies.index = movies['movie_id']
movies = movies.drop(columns=['movie_id'])

movies = preprocess_movies(movies)

ratings = pd.read_csv('data/ratings.dat', sep='::', names=['user_id', 'movie_id', 'rating', 'rating_timestamp'], header=None, engine='python')
ratings_new = ratings.drop(columns=['rating_timestamp'])

ratings_new['count_user'] = ratings_new.groupby(['user_id'])['user_id'].transform('count')
ratings_user = ratings_new.query('count_user >= @min_movies_rated_by_user')
ratings_user = ratings_user.drop(columns=['count_user'])

ratings_user['count_movie'] = ratings_user.groupby(['movie_id'])['movie_id'].transform('count')
ratings_movie = ratings_user.query('count_movie >= @min_users_rated_movie')

user_by_movie = ratings_movie.groupby(['user_id', 'movie_id'])['rating'].max().unstack()
print('The size of user by movie matrix is ', user_by_movie.shape)

movies_used = movies.loc[user_by_movie.columns, 'movie_title']
movie_genres_df = movies.loc[user_by_movie.columns, 'genra']

user_by_movie_matrix = user_by_movie.to_numpy()

def create_genra_list(movie_genres_df):
    genres_set = set()
    for i in range(movie_genres_df.shape[0]):
        genres_list = movie_genres_df.iloc[i].split('|')
        genres_set.update(genres_list)
    return list(genres_set)

def create_genres_matrix(movie_genres_df, genres):
    numb_movies = movie_genres_df.shape[0]
    numb_genres = len(genres)
    genres_matrix = np.zeros(shape=(numb_movies, numb_genres))
    for i in range(numb_movies):
        genres_row = set(movie_genres_df.iloc[i].split('|'))
        for j,genre in enumerate(genres):
            if genre in genres_row:
                genres_matrix[i,j] = 1
    return genres_matrix


def get_all_scores(user_by_movie_matrix):
    temp_matrix = user_by_movie_matrix.copy()
    temp_matrix[np.isnan(temp_matrix)] = 0.0
    scores = np.dot(temp_matrix.T, temp_matrix)
    np.fill_diagonal(scores, -1)
    return scores


###########################################
# ratings
current_time1 = time.time()

scores = get_all_scores(user_by_movie_matrix)

current_time2 = time.time()


###########################################
# genres
genres = create_genra_list(movie_genres_df)
genres_matrix = create_genres_matrix(movie_genres_df, genres)
genre_similarities = cosine_similarity(genres_matrix)


###########################################
# combine genres and ratings
scores = np.multiply(scores, genre_similarities)


###########################################
# get the top movies
negative_n = -1*number_movies_returned
closest_movies = np.argpartition(scores, negative_n, axis=1)[:, negative_n:]

current_time3 = time.time()

time_scores = current_time2 - current_time1
time_total = current_time3 - current_time1
print('Total time: ', time_total)
print('Score calculation took: ', time_scores)


###########################################
# save to db
all_closest_movies_df = pd.DataFrame()
all_closest_movies_df['movie_title'] = movies_used.values
for i in range(number_movies_returned):
    all_closest_movies_df['closest_movie_{}'.format(i+1)] = closest_movies[:,-1*i]

engine = create_engine('sqlite:///' + global_database_filepath)
all_closest_movies_df.to_sql(table_name, engine, index=False, if_exists='replace')
engine.dispose()





