import pytest
from ie_bike_model.model import test_train_split
from ie_bike_model.preprocess import whole_preprocessing
from ie_bike_model.load_data import loading_data
import pandas as pd


def test_test_train_split(data=whole_preprocessing(loading_data())):
    output = test_train_split(data)
    assert type(output) is tuple


def test_test_train_split_length_tuple(data=whole_preprocessing(loading_data())):
    output = len(test_train_split(data))
    expected_output = 4
    assert output == expected_output


def test_test_train_split_length(data=whole_preprocessing(loading_data())):
    output = len(test_train_split(data)[0])
    expected_output = 15211
    assert output == expected_output


def test_test_train_split_column_length(data=whole_preprocessing(loading_data())):
    col_num = test_train_split(data)[0]
    output = len(col_num.columns)
    expected_output = 80
    assert output == expected_output


def test_test_train_split_target_type(data=whole_preprocessing(loading_data())):
    cnt = test_train_split(data)[1]
    output = type(cnt)
    expected_output = pd.core.series.Series
    assert output == expected_output


def test_test_train_split_train_not_cnt(data=whole_preprocessing(loading_data())):
    output = test_train_split(data)[2].columns
    expected_output = "cnt"
    assert expected_output not in output
