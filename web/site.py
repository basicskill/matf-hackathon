import plotly
import plotly.graph_objs as go

import pandas as pd
import numpy as np
import json

from flask import Flask , render_template
app = Flask(__name__,static_url_path="/assets")
app.static_folder="templates/assets"

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


@app.route('/api/predictions')
def predictions():
    return {"predictions": list(range(24 * 7))}


if __name__ == "__main__":
    app.run(debug =True)