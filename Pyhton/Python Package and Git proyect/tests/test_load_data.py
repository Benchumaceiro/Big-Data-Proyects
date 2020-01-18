import pytest
import pandas as pd
from ie_bike_model.load_data import loading_data

# from ie_bike_model_test.load_data import loading_data


def test_loading_data_return_DF():
    output = loading_data()
    assert type(output) is pd.DataFrame


def test_loading_data_return_numberColumns():
    output = loading_data()
    CSV = pd.read_csv(
        "https://ndownloader.figshare.com/files/20976540",
        index_col="instant",
        parse_dates=True,
    )
    expected_output = len(CSV.columns)
    assert len(output.columns) == expected_output


def test_loading_data_retur_ColumnNames():
    output = loading_data()
    CSV = pd.read_csv(
        "https://ndownloader.figshare.com/files/20976540",
        index_col="instant",
        parse_dates=True,
    )
    expected_output = print(CSV.columns)
    assert print(output.columns) == expected_output


def test_loading_data_retur_describe():
    output = loading_data()
    CSV = pd.read_csv(
        "https://ndownloader.figshare.com/files/20976540",
        index_col="instant",
        parse_dates=True,
    )
    expected_output = print(CSV.dtypes)
    assert print(output.dtypes) == expected_output
