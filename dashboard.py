from flask import Flask, render_template, request, redirect, url_for
import pandas as pd

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from make_plots import *



app = Flask(__name__)

SPOTIFY_DATA = None


@app.before_request
def gather_data():
    global SPOTIFY_DATA
    
    SPOTIFY_DATA = # contact Spotify API via your own function 
    


@app.route("/")
def home():

    global SPOTIFY_DATA
    
    # Get the genres from query parameters
    genres = [request.args.get(genre) for genre in ["genre1", "genre2"]]
    
    # Create a dictionary of genres.
    # Maps simplified string formatting to original.
    mapped_genres = {original.lower().replace("-", "").replace(" ", ""): original for original in set(SPOTIFY_DATA.UpdatedGenre)}
    
    # Grab user data. If no input yet, start with two most populous.
    if genres[0] is not None:
        genres = [mapped_genres[genres[0]], mapped_genres[genres[1]]]
    else:
        genres = SPOTIFY_DATA.UpdatedGenre.value_counts()[:2].index.to_list()
    
    
    make_barplot(genres, SPOTIFY_DATA)
    sorted_genres, colors = make_boxplot(genres, SPOTIFY_DATA)
    result, dof = make_distribution(sorted_genres, SPOTIFY_DATA, colors)
    make_tdistribution(result, dof)
    
    return render_template("home.html", 
                            user_query=genres)
                            
@app.route("/submit", methods=["POST"])
def submit():
    
    # Capture the genres from user input, split on comma
    genres_str = request.form.get("user_query").split(",")
    
    # Remove whitespace, replace hyphen and any gaps. Lowercase.
    genre1 = genres_str[0].strip().replace("-", "").replace(" ", "").lower()
    genre2 = genres_str[1].strip().replace("-", "").replace(" ", "").lower()
    
    
    # Redirect to home with the genres as query parameter
    return redirect(url_for("home",
                            genre1=genre1,
                            genre2=genre2))

