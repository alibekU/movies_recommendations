from flask import Flask, Response, render_template, request
import json
from wtforms import TextField, Form
import pandas as pd

app = Flask(__name__)

movies = pd.read_csv('data/movies.dat', sep='::', names=['movie_id', 'movie_title', 'genra'], header=None )
movie_titles = movies['movie_title'].to_list()

class SearchForm(Form):
    autocomplete = TextField('Movie name', id='movie_autocomplete')


@app.route('/autocomplete', methods=['GET'])
def autocomplete():
    return Response(json.dumps(movie_titles), mimetype='application/json')


@app.route('/', methods=['GET', 'POST'])
def index():
    form = SearchForm(request.form)
    return render_template("search.html", form=form)


if __name__ == '__main__':
    app.run(debug=True)