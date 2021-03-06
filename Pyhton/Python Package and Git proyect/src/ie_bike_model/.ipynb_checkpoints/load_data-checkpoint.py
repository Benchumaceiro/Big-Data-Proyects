import pandas as pd


def loading_data(data="hour.csv"):
    """
    Loading the data for the preprocessing, training and predicting.
    """
    hour = pd.read_csv(data, index_col="instant", parse_dates=True)
    return hour
