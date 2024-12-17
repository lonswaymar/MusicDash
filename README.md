# The contents of this repository are insufficient to run the dashboard
since I've removed the portion that contacts the Spotify API.

To start it up again, begin by writing a function that grabs data from the
API and insert its call into the @app.before_request endpoint in dashboard.py

Have your API-calling-function return a pandas dataframe with columns:
Artist,Title,Duration,UpdatedGenre where `UpdatedGenre` refers to the genre 
classification of a given song and to which song durations will be compared.

If this is achieved, run `flask --app dashboard run` and navigate to your local
endpoint specified in the output to visualize an interactive dashboard that
compares durations of songs within user-specified genres.

The matplotlib figures pushed to the Flask server are shown in static/images.


