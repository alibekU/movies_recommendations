'''
    create_recommendations.py - script to identify similar movies based on user ratings and genres, and store them in a database.
    The location of files with movie data and ratings are set by default in the process_data function
'''

# import libraries
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import time
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from global_parameters import *
from scipy.spatial import distance

# assign values from global_parameters.py to local variables
# number_movies_returned - how many most similar movies to store for each movie and output when requested
number_movies_returned  = global_number_movies_returned
# min_movies_rated_by_user - how many movies at least a user should rated to keep the user for later processing 
min_movies_rated_by_user = global_min_movies_rated_by_user
# min_users_rated_movie - how many users at least should rate a movie to keep the movie for later processing
min_users_rated_movie = global_min_users_rated_movie
# min_common_raters - minimum number of common reviewers between two movies to be considered as similar
min_common_raters = global_min_common_raters
# global_database_filepath - name of the sqlalchemy database file where recommendations will be stored 
global_database_filepath = global_database_filepath
# table_name - name of the table with recommendations in the above database 
table_name = global_recs_table_name

def process_movie_names(movies_df):
    '''
        process_movie_names - function to correct some of the names of the movies that 
        can cause inconvenience for the user when searching for it in the web app.

        Input:
        - movies_df - Pandas Series with movie titles
        Output:
        - movies_df - Pandas Series with corrected movie titles
    '''
    movies_df.loc[movies_df == 'Gisaengchung (2019)'] = 'Parasite (Gisaengchung) (2019)'
    movies_df.loc[movies_df == 'Vaiana (2016)'] = 'Moana (Vaiana) (2016)'
    pattern_amp = ['&amp;', '&'] 
    movies_df = movies_df.str.replace('|'.join(pattern_amp), 'and')
    movies_df = movies_df.str.replace('é', 'e')
    return movies_df 

def process_data(movies_file='data/movies.dat', ratings_file='data/ratings.dat'):
    '''
        process_data - function to correct to filter data and create objects for use 
        in recommendations generation

        Input:
        - movies_file - str, path to the file with data on movies, default='data/movies.dat'
        - ratings_file - str, path to the file with data on movie ratings, default='data/ratings.dat'
        Output:
        - user_by_movie_matrix - numpy matrix with movie ratings where each row represents a user and each column a movie
        - movies_used - Pandas Series with movies that are used for recommendations creating
        - movie_genres_df - Pandas DataFrame with genres of movies used for recommendations creating
    '''
    movies = pd.read_csv(movies_file, sep='::', names=['movie_id', 'movie_title', 'genra'], header=None, engine='python')
    movies.dropna(inplace=True)
    movies.drop_duplicates(subset=['movie_id'], inplace=True)
    movies.index = movies['movie_id']
    movies = movies.drop(columns=['movie_id'])

    ratings = pd.read_csv(ratings_file, sep='::', names=['user_id', 'movie_id', 'rating', 'rating_timestamp'], header=None, engine='python')
    ratings_new = ratings.drop(columns=['rating_timestamp'])
    ratings_new['count_user'] = ratings_new.groupby(['user_id'])['user_id'].transform('count')
    ratings_user = ratings_new.query('count_user >= @min_movies_rated_by_user')
    ratings_user = ratings_user.drop(columns=['count_user'])
    ratings_user['count_movie'] = ratings_user.groupby(['movie_id'])['movie_id'].transform('count')
    ratings_movie = ratings_user.query('count_movie >= @min_users_rated_movie')

    user_by_movie = ratings_movie.groupby(['user_id', 'movie_id'])['rating'].max().unstack()
    print('The size of user by movie matrix is ', user_by_movie.shape)

    movies_used = movies.loc[user_by_movie.columns, 'movie_title']
    movies_used = process_movie_names(movies_used)
    movie_genres_df = movies.loc[user_by_movie.columns, 'genra']
    user_by_movie_matrix = user_by_movie.to_numpy()
    
    return user_by_movie_matrix, movies_used, movie_genres_df

def create_genra_list(movie_genres_df):
    '''
        create_genra_list - function to create a list with all possible genres

        Input:
        - movie_genres_df - Pandas DataFrame with genres of movies used for recommendations creating
        Output:
        - genres_list - list with all the genres
    '''
    genres_set = set()
    for i in range(movie_genres_df.shape[0]):
        genres_list = movie_genres_df.iloc[i].split('|')
        genres_set.update(genres_list)
    genres_list = list(genres_set)
    return genres_list

def create_genres_matrix(movie_genres_df, genres):
    '''
        create_genres_matrix - function to create a 0-1 matrix which shows what genres are applicable for each movie

        Input:
        - movie_genres_df - Pandas DataFrame with genres of movies used for recommendations creating
        - genres - list of all possible genres in 'movie_genres_df'
        Output:
        - genres_matrix - numpy matrix of zeroes and ones, where 1 means that a genre applicable to a movie, and where each row is a movie and 
        each column is a genres
    '''
    numb_movies = movie_genres_df.shape[0]
    numb_genres = len(genres)
    genres_matrix = np.zeros(shape=(numb_movies, numb_genres))
    for i in range(numb_movies):
        genres_row = set(movie_genres_df.iloc[i].split('|'))
        for j,genre in enumerate(genres):
            if genre in genres_row:
                genres_matrix[i,j] = 1
    return genres_matrix


def get_common_raters(user_by_movie_matrix):
    '''
        get_common_raters - function to calculate number of common raters (users who rated both movies) between each pair of movies based on user ratings 
        using linear kernel to speed up the process (like cosine similarity but without norming - sum of multilications of corresponding coordinates).

        Input:
        - user_by_movie_matrix - numpy matrix with movie ratings where each row represents a user and each column a movie
        Output:
        - common_raters - numpy matrix of raters where each row and column represent a movie, so score[i,j] stores number of users who scored both movies number i and number j.
    '''
    # create a copy of matrix since we will be modifying it
    temp_matrix = user_by_movie_matrix.copy()
    # fill all non nans with ones to compute number of common elements when multiplying vectors
    temp_matrix[~np.isnan(temp_matrix)] = 1
    # fill nans with zeroes to have correct matrix multiplication
    temp_matrix[np.isnan(temp_matrix)] = 0
    # compute linear kernel (like cosine similarity but without norming) - measure of similarity between movies
    # the first term is transposed as user_by_movie_matrix has movies as columns, but we need first term to row-oriented
    common_raters = np.dot(temp_matrix.T, temp_matrix)
    # replace zeroes and movies that have too few common reviewers with -1 to avoid dividing by zero and give the movies with few common raters a low score
    common_raters[common_raters<min_common_raters] = -1
    common_raters[common_raters>=min_common_raters] = 1
    return common_raters


def compute_rating_score(user_by_movie_matrix):
    '''
        compute_rating_score - function to calculate similarity score between each pair of movies based on user ratings 
        using average linear kernel (like cosine similarity but without norming - sum of multilications of corresponding coordinates, divided by the number of  corresponding coordinates).
        This works well for:
        1. The higher the scores given by a user to a 2 different movies on average, the more the user liked both movies, and the higher will be the score. 

        Input:
        - user_by_movie_matrix - numpy matrix with movie ratings where each row represents a user and each column a movie
        Output:
        - scores - numpy matrix of scores where each row and column represent a movie, so score[i,j] stores similarity score based on ratings
        between movies number i and number j.
    '''
    # matrix with the number of common reviewers between each pair of movies
    common_raters= get_common_raters(user_by_movie_matrix)
    # create a copy of matrix since we will be modifying it
    temp_matrix = user_by_movie_matrix.copy()
    # fill nans with zeroes to have correct matrix multiplication
    temp_matrix[np.isnan(temp_matrix)] = 0.0

    # compute linear kernel (like cosine similarity but without norming) - measure of similarity between movies
    # the first term is transposed as user_by_movie_matrix has movies as columns, but we need first term to row-oriented
    scores_kernel = np.dot(temp_matrix.T, temp_matrix)
    # divide the scores by number of common reviewers to get a more fair average score
    scores = scores_kernel * (1/common_raters)

    # set diagonals with 0 so that movies is not declared to be most similar to itself in the web app
    np.fill_diagonal(scores, 0)

    return scores


def main():
    ###########################################
    # get the data

    user_by_movie_matrix, movies_used, movie_genres_df = process_data()


    ###########################################
    # calculate ratings-based similarity score

    # time the performance
    current_time1 = time.time()
    # get similarity scores
    scores = compute_rating_score(user_by_movie_matrix)
    current_time2 = time.time()


    ###########################################
    # calculate genres-based similarity score 

    genres = create_genra_list(movie_genres_df)
    genres_matrix = create_genres_matrix(movie_genres_df, genres)
    # use cosine similarity as the number of coordinates is constant and they are 0 and 1, 
    # not integer numbers with quantity meaning something
    genre_similarities = cosine_similarity(genres_matrix)


    ###########################################
    # combine genres and ratings

    # multiply position by position, not as matrices to apply genre scores sqrt (to lessen its effect) as weights
    scores = np.multiply(scores, np.sqrt(genre_similarities))


    ###########################################
    # get the top movies

    # will need last 'number_movies_returned' movies when sorted ascending
    negative_n = -1*number_movies_returned
    # use partial aggregation to get top n movies without full sort
    closest_movies = np.argpartition(scores, negative_n, axis=1)[:, negative_n:]
    # time performance
    current_time3 = time.time()


    ###########################################
    # output performance

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


if __name__ == "__main__":
    main()



