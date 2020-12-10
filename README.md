# movies_recommendations
A web app that recommends movies similar to those a user provides as an input.

# Table of contents
- [Purpose](#purpose)
- [Web Application](#web-application)
- [Data](#data)
- [Installation](#installation)
- [Usage](#usage)
- [Project structure](#project-structure)
- [Discussion of the results](#discussion-of-the-results)
- [Author](#author)
- [Credits](#credits)
- [Requirements](#requirements)


# Purpose
To build a recommendation system* that will help find movies similar to those you like based on thousands of movie ratings by other people. Say you want to enjoy a movie this evening and cannot come with a movie to watch, but you know that recently you liked a movie, like "Knives Out", and so why not watch something similar? This web app will help you with just that! <br/> 
*Strictly speaking, this not a recommendation system, but rather a tool that finds similar movies based on user ratings and then applies weights based on movie genres

# Web Application
The app is hosted at https://find-movies-like.herokuapp.com
<br/>
Instructions:<br/>

1. Simply start typing name of the movie you liked in the search bar, and as you type suggestions of movie names that are available will appear in the drop-down

2. Select the movie you want from the drop down and press the "Find Movies!" button or hit "Enter" on your keyboard.
3. The resulting 12 most similar movies will appear. You can click on any movie title to open up a new tab with google search query for that title.


# Data
The data consists of movie ratings by ~ 40 thousand of users for 2 thousand movies that were rated the most in the MovieTweeting Database (https://github.com/sidooms/MovieTweetings). The whole database consists of many more ratings, but to ensure better quality and relevance only those movies that were rated at least 80 times were selected. <br/>
The database was created by Simon Dooms, who came up with the idea of using data from Twitter when anyone post that they an IMDB movie. At the moment (December 2020), the database used in the web appp has ratings up to September 2020. 

# Installation
1. In order to install the code and deploy the app locally please download from Github: `git clone https://github.com/alibekU/movies_recommendations.git`.
2. You may want to set up a new virtual environment: `python3 -m venv /path/to/new/virtual/environment` 
3. Then, use pip to install all the needed packages: `pip install -r requirements.txt`

# Usage
To run the web app locally after downloading files, go to the the 'movies_recommendations/' folder and:
1. Run this command:
    `python run.py`

2. Go to http://0.0.0.0:3001/

To rebuild a movie recommendations list: <br/>
1. Change the `global_database_filepath` parameter in the `global_parameters.py` file, which will be the name of the database file with recommendations.
2. Make any other changes to the parameters in that file or to the algorithm in the `create_recommendations.py` file.
3. Run `python create_recommendations.py` command to create new recommendations
4. Run the web app locally as described above in 'Usage'.


# Project structure 
| data\ <br/>
|- movies.dat - data on the movies from https://github.com/sidooms/MovieTweetings <br/>
|- ratings.dat - data on the ratings by users from https://github.com/sidooms/MovieTweetings <br/>

| recommendation_database\ - database files with movie recommendations generated by create_recommendations.py <br/>

| run.py - the main script of Flask web app <br/>
| process_data.py - functions to clean data, select movies and determine similar movies for each selected movie <br/>
| global_parameters.py - global parameters used

| templates\ <br/>
|- index.html - main html page template <br/>
|- results.html - a template for displaying results <br/>

| static\ <br/>
|- stylesheets\ <br/>
|-- styles.css - CSS file with styles

| requirements.txt - a list of required PIP packages, result of `pip freeeze` command <br/>
| Procfile - code for Flask app launch at Heroku <br/>
| analysis.ipynb - a Jupyter notebook with data exploration <br/>
| README.md - readme file <br/>

# Discussion of the results
To decide how movies are similar to each other I did the following:
1. Calculated user ratings similarity score between all pairs of movies by using the linear kernel function (https://scikit-learn.org/stable/modules/metrics.html#linear-kernel, and it is basically cosine similarity without norming) on each column of user-movie ratings matrix built from the ratings data
2. Calculated how many common reviewers are between each pair of movies, and used this information to:
- - Avoid comparing movies that have less than 5 common reviewers to avoid coincedental high score
- - Divide the each score from step 1 by corresponding number of common reviewers to have an average product of scores rather than the sum of products. This allows to avoid popular movies with many reviews being identified as most similar just because they have a large score in step 1.
3. Calculated movie genres similarity score between all pairs of movies by using cosine distance between each movie's genres and added these scores to the user ratings similarity score with weights to get the final score (weights are defined in global_parameters.py)
4. For each movie then selected N (defined in global_parameters.py) movies with highest resulting scores.
<br/>
The results were not measured at this point as I didn't come with a proper way for it. The problem is that there is no single correct way to compute similarity of the movies, and computing similarity in one way, and then testing it with some other formula does not make sense. What does it mean for two movies to be similar? It means that people will find these two movies alike, and if they enjoyed one movie, they are likely to enjoy the other, and the opposite goes for the case they disliked one of the movies. The best way to measure quality of such a tool is to conduct a user study and ask people to rate the recommendations. At this point this metric on a small set of users (basically my friends) look alright.

# Author 
Alibek Utyubayev. 
<br/>
Linkedin: https://www.linkedin.com/in/alibek-utyubayev-74402721/

# Credits
Credits to research of Simon Dooms (https://github.com/sidooms/MovieTweetings) for the collected data on movies which is used in the app, and to https://github.com/LukasSliacky/Flask-Autocomplete for the autocomplete and Flask integration.

# Requirements
Listed in requirements.txt file:<br/>
click==7.1.2
Flask==1.1.2
gunicorn==20.0.4
itsdangerous==1.1.0
Jinja2==2.11.2
joblib==0.17.0
llvmlite==0.34.0
MarkupSafe==1.1.1
numpy==1.19.4
pandas==1.1.4
python-dateutil==2.8.1
pytz==2020.4
scikit-learn==0.23.2
scipy==1.5.4
six==1.15.0
sklearn==0.0
SQLAlchemy==1.3.20
threadpoolctl==2.1.0
Werkzeug==1.0.1
WTForms==2.3.3



