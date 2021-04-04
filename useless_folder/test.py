import pandas as pd
import numpy as np


def main():
    csv = 'data/correlations.csv'
    site_data = pd.read_csv(csv)
    print(site_data.to_numpy()[:,1:])


if __name__ == "__main__":
    main()