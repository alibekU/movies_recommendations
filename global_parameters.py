'''
    global_parameters.py - python script with global variables set to proper values for later use both in the 
    recommendations generating (create_recommendations.py) and in the web app (run.py)
'''

# how many most similar movies to store for each movie and output when requested
global_number_movies_returned  = 12
# name of the sqlalchemy database file where recommendations will be stored
global_database_filepath = 'recommendation_database/movie_recs_6000_kernel_min5_common_genres_15.db'
# name of the table with recommendations in the above database
global_recs_table_name = 'Closest_movies'
# how many movies at least a user should rated to keep the user for later processing
global_min_movies_rated_by_user = 2
# how many users at least should rate a movie to keep the movie for later processing
global_min_users_rated_movie = 15
# minimum number of common reviewers between two movies to be considered as similar
global_min_common_raters = 5