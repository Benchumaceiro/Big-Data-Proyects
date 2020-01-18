import pandas as pd


def loading_data(data="https://ndownloader.figshare.com/files/20976540"):
    """
    Loading the data for the preprocessing, training and predicting.
    """
    hour = pd.read_csv(data, index_col="instant", parse_dates=True)
    return hour
