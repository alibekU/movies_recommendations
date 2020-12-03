import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import time
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from scipy.spatial import distance

number_movies_returned  = 10
min_movies_rated_by_user = 2
min_users_rated_movie = 100
liked_score_min = 7
database_filename = 'movie_recommendations_1600_multipl_genres.db'
table_name = 'Closest_movies'


movies = pd.read_csv('data/movies.dat', sep='::', names=['movie_id', 'movie_title', 'genra'], header=None, engine='python')
movies.dropna(inplace=True)
movies.drop_duplicates(subset=['movie_id'], inplace=True)


movies.index = movies['movie_id']
movies = movies.drop(columns=['movie_id'])

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

'''
def get_users(movie_index, user_by_movie_matrix):
    column = user_by_movie_matrix[:,movie_index]
    users = np.where(np.isnan(column) == False)[0] 
    return users

def get_common_users(movie_index1, movie_index2, user_by_movie_matrix):
    users1 = get_users(movie_index1, user_by_movie_matrix)
    users2 = get_users(movie_index2, user_by_movie_matrix)
    common_users = np.intersect1d(users1, users2, assume_unique=True)
    return common_users

def compute_score(movie_index1, movie_index2, user_by_movie_matrix):
    common_users = get_common_users(movie_index1, movie_index2, user_by_movie_matrix)
    movie1_ratings = user_by_movie_matrix[common_users, movie_index1]
    movie2_ratings = user_by_movie_matrix[common_users, movie_index2]

    score = 1- distance.cosine(movie1_ratings,movie2_ratings)

    return score

def get_all_scores(user_by_movie_matrix):
    movies_number = user_by_movie_matrix.shape[1]

    user_by_movie_matrix[user_by_movie_matrix < liked_score_min] = 0
    user_by_movie_matrix[user_by_movie_matrix >= liked_score_min] = 1

    scores = np.zeros(shape=(movies_number, movies_number))
    for index1 in range(movies_number):
        for index2 in range(index1+1, movies_number):
            scores[index1, index2] = compute_score(index1, index2, user_by_movie_matrix)
    return scores
'''

def get_all_scores(user_by_movie_matrix):
    #movies_number = user_by_movie.shape[1]
    #scores = np.zeros(shape=(movies_number, movies_number))
    '''
    for index1 in range(movies_number):
        diffs = np.subtract(user_by_movie_matrix, np.vstack(user_by_movie_matrix[:, index1]))
        diffs[np.isnan(diffs)] = 0.0
        #scores[index1] = np.linalg.norm(diffs, axis=0)
        #scores[index1] = np.nanmean(np.absolute(diffs), axis=0)
    #np.fill_diagonal(scores, -1)
    #scores[scores==0] = 1
    #scores = 1 / scores
    '''
    temp_matrix = user_by_movie_matrix.copy()
    temp_matrix[np.isnan(temp_matrix)] = 0.0
    scores = np.dot(temp_matrix.T, temp_matrix)
    np.fill_diagonal(scores, -1)
    return scores


###########################################
# genres

genres = create_genra_list(movie_genres_df)
genres_matrix = create_genres_matrix(movie_genres_df, genres)
genre_similarities = cosine_similarity(genres_matrix)


###########################################
# ratings

current_time1 = time.time()

scores = get_all_scores(user_by_movie_matrix)

current_time2 = time.time()


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

engine = create_engine('sqlite:///' + database_filename)
all_closest_movies_df.to_sql(table_name, engine, index=False, if_exists='replace')
engine.dispose()





