from flask import Flask, Response, render_template, request
import json
from wtforms import TextField, Form
import pandas as pd
from sqlalchemy import create_engine
from global_parameters import *

# assign values from global_parameters.py to local variables
# number_movies_returned - how many most similar movies output when requested for each movie
number_movies_returned  = global_number_movies_returned
# database_filepath - name of the sqlalchemy database file where recommendations are stored
database_filepath = global_database_filepath

app = Flask(__name__)


# extract movies data
engine = create_engine('sqlite:///'+ database_filepath)
movies_data = pd.read_sql_table('Closest_movies', engine)

engine.dispose()

# get the movie titles
movie_titles = list(movies_data['movie_title'])

# SearchForm class will allow us to have autocomplete feature
class SearchForm(Form):
    movie_autocomplete = TextField('Movie name', id='movie_autocomplete')


@app.route('/autocomplete', methods=['GET', 'POST'])
def autocomplete():
    '''
        autocomplete - function to respond to a request from javascript fuction
        responsible for autocomplete feature. Sends all the movie titles to the front.
    '''
    return Response(json.dumps(movie_titles), mimetype='application/json')


@app.route('/', methods=['GET', 'POST'])
def index():
    form = SearchForm(request.form)
    return render_template("index.html", form=form)

@app.route('/results', methods=['GET', 'POST'])
def results():
    requested_movie = request.args.get('movie_autocomplete', '')
    # get the recommendatuins on requested movie
    movie_row = movies_data[movies_data['movie_title'] == requested_movie].reset_index()
    similar_movies = []
    if movie_row.shape[0] != 0:    
        for i in range(number_movies_returned):
            column_name = 'closest_movie_{}'.format(i+1)
            # get the id of i-th closest movie
            id = movie_row.loc[0,column_name]
            # use the id to get the name of the closest movie
            movie_i = movies_data.loc[id, 'movie_title']
            similar_movies.append(movie_i)
    form = SearchForm(request.form)
    return render_template(
        'results.html',
        requested_movie=requested_movie,
        similar_movies=similar_movies,
        form=form
    )

def main():
    app.run(host='0.0.0.0', port=3001, debug=True)

if __name__ == '__main__':
    main()