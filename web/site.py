import plotly
import plotly.graph_objs as go

import pandas as pd
import numpy as np
import json
import random
import datetime

from flask import Flask, render_template


app = Flask(__name__, static_url_path="/assets")
app.static_folder = "templates/assets"


def create_plot():
    N = 40
    x = np.linspace(0, 1, N)
    y = np.random.randn(N)
    df = pd.DataFrame({'x': x, 'y': y}) # creating a sample dataframe


    data = [
        go.Bar(
            x=df['x'], # assign x as the dataframe column 'x'
            y=df['y']
        )
    ]

    graphJSON = json.dumps(data, cls=plotly.utils.PlotlyJSONEncoder)

    return graphJSON


@app.route('/pollution_days')
def index_days():

    bar = create_plot()
    return render_template('test.html', plot=bar)


@app.route('/pollution_months')
def index_months():

    bar = create_plot()
    return render_template('test.html', plot=bar)


@app.route('/pollution_hours')
def index_hours():

    bar = create_plot()
    return render_template('test.html', plot=bar)


@app.route("/")
def index():
    return render_template("index.html")


@app.route('/api/predictions')
def predictions():
    now = datetime.date.today()
    return {"predictions": [[{"aqi": random.randint(0, 500), "so2": 100, "b": 100, "co": 100, "no2": 100, "o3": 100, "pm10": 100, "pm25": 100, "time": f"{h:02}:00", "date": "today" if d == 0 else ("tomorrow" if d == 1 else (now+datetime.timedelta(d)).strftime("%d. %m."))} for h in range(24)] for d in range(5)]}


if __name__ == "__main__":
    app.run(debug=True)
