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

def create_plot_co(csv):

    site_data = pd.read_csv(csv)
    print(csv)
    if "hour" in site_data:
        x_axis = site_data["hour"]
    else:
        if "day" in site_data:
            x_axis = site_data["day"]
        else:
            if "month" in site_data:
                x_axis = site_data["month"]
    
    data = [
        go.Bar(
            x=x_axis, # assign x as the dataframe column 'x'
            y= site_data["CO"]
        )
    ]

    graphJSON = json.dumps(data, cls=plotly.utils.PlotlyJSONEncoder)

    return graphJSON

def create_plot_aqi(csv):

    site_data = pd.read_csv(csv)
    print(csv)
    if "hour" in site_data:
        x_axis = site_data["hour"]
    else:
        if "day" in site_data:
            x_axis = site_data["day"]
        else:
            if "month" in site_data:
                x_axis = site_data["month"]

    data = [
        go.Bar(
            x=x_axis,    # assign x as the dataframe column 'x'
            y= site_data["AQI"]
        )
    ]

    graphJSON = json.dumps(data, cls=plotly.utils.PlotlyJSONEncoder)

    return graphJSON


def create_plot_confusion(csv):

    site_data = pd.read_csv(csv)
    class_names = ['month',	'hour',	'T', 'P', 'P0', 'CO', 'NO2', 'O3', 'PM10', 'PM25', 'SO2']
    layout = {
        "title": "Correlation Weather | Pollution ", 
        "xaxis": {"title": "Predicted value"}, 
        "yaxis": {"title": "Real value"}
    }

    data = [go.Heatmap(z=site_data.to_numpy()[:,1:],
                        x=class_names,
                        y=class_names,
                        hoverongaps=False)
    ]
    graphJSON = json.dumps(data, cls=plotly.utils.PlotlyJSONEncoder)

    return graphJSON

@app.route('/pollution_hours_co')
def index_hours_co():
    csv = 'data/website_daily.csv'
    bar = create_plot_co(csv)
    return render_template('test.html', plot=bar)

@app.route('/pollution_hours_aqi')
def index_hours_aqi():
    csv = 'data/website_daily.csv'
    bar = create_plot_aqi(csv)
    return render_template('test.html', plot=bar)


@app.route('/pollution_months_co')
def index_months_co():
    csv = 'data/website_yearly.csv'
    bar = create_plot_co(csv)
    return render_template('test.html', plot=bar)

@app.route('/pollution_months_aqi')
def index_months_aqi():
    csv = 'data/website_yearly.csv'
    bar = create_plot_aqi(csv)
    return render_template('test.html', plot=bar)

@app.route('/pollution_days_co')
def index_days_co():
    csv = 'data/website_monthly.csv'
    bar = create_plot_co(csv)
    return render_template('test.html', plot=bar)

@app.route('/pollution_days_aqi')
def index_days_aqi():
    csv = 'data/website_monthly.csv'
    bar = create_plot_aqi(csv)
    return render_template('test.html', plot=bar)

@app.route('/correlation')
def index_corellations():
    csv = 'data/correlations.csv'
    bar = create_plot_confusion(csv)
    return render_template('test.html', plot=bar)

@app.route("/")
def index():
    score = 30
    decided = '😌 Bad'
    opacity = 0.8
    if score > 25:
        decided =  '🙁 Poor'
        opacity = 0.5
    if score > 50:
        decided =  '🙂 Good'
        opacity = 0.1
    if score > 75:
        decided =  '😃 Excellent'
        opacity = 0
    data=[
    {
        'emoji': decided,
        'opacity': opacity
    }
]
    return render_template("index.html", data = data)


@app.route('/api/predictions')
def predictions():
    now = datetime.datetime.now() + datetime.timedelta(hours=2)
    return {"predictions": [[{"aqi": random.randint(0, 600), "so2": random.randint(0, 1600), "b": random.randint(0, 600), "co": random.randint(0, 35), "no2": random.randint(0, 400), "o3": random.randint(0, 750), "pm10": random.randint(0, 450), "pm25": random.randint(0, 250), "time": f"{h:02}:00", "date": "today" if d == 0 else ("tomorrow" if d == 1 else (now+datetime.timedelta(d)).strftime("%d. %m."))} for h in range(now.hour if d == 0 else 0, 24)] for d in range(5)]}


if __name__ == "__main__":
    app.run(debug=True)
