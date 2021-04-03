"""Script for data loading, cleaning and visualization."""

import pandas as pd
import numpy as np
import datetime


if __name__ == "__main__":

    data_frame = pd.read_csv("MATF_Hackathon_2021/BelgradeAirport_2021-2012.csv")

    print(data_frame.columns)