"""Script for data loading, cleaning and visualization."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def datetime2ymd(in_df, out_df):
    out_df["year"]  = in_df["DateTime"].astype(str).str[:4].astype(int)
    out_df["month"] = in_df["DateTime"].astype(str).str[5:7].astype(int)
    out_df["day"]   = in_df["DateTime"].astype(str).str[8:10].astype(int)
    out_df["hour"]  = in_df["DateTime"].astype(str).str[11:13].astype(int)
    out_df["min"]   = in_df["DateTime"].astype(str).str[14:].astype(int)
    
    return out_df

def datetime2dmy(in_df, out_df):
    out_df["year"]  = in_df["DateTime"].astype(str).str[6:10].astype(int)
    out_df["month"] = in_df["DateTime"].astype(str).str[3:5].astype(int)
    out_df["day"]   = in_df["DateTime"].astype(str).str[:2].astype(int)
    out_df["hour"]  = in_df["DateTime"].astype(str).str[11:13].astype(int)
    out_df["min"]   = in_df["DateTime"].astype(str).str[14:].astype(int)
    
    return out_df


def main():
    input_data = pd.read_csv("MATF_Hackathon_2021/BA_2012-2021.csv", sep=";")
    weather_data = pd.DataFrame()

    # Copy time stamps
    weather_data = datetime2dmy(input_data, weather_data)

    # Copy columns with relevant data
    useful_columns = ["T", "Po", "U", "Ff", "Tn"]
    weather_data[useful_columns] = input_data[useful_columns]

if __name__ == "__main__":
    main()