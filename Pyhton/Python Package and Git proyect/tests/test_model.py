import pytest
from ie_bike_model.load_data import loading_data
from ie_bike_model.preprocess import whole_preprocessing
from ie_bike_model.model import predict
from ie_bike_model.model import dump
import functools
import os


def test_predict_data_cols_len():
    output = len(whole_preprocessing(loading_data()).columns)
    expected_output = 85
    assert output == expected_output


def test_predict_data_rows_len():
    output = len(whole_preprocessing(loading_data()))
    expected_output = 17379
    assert output == expected_output


def test_predict_output_type():
    output = type(predict(loading_data()))
    expected_output = str
    assert output == expected_output


def test_predict_output_len():
    output = len(predict(loading_data()))
    expected_output = 21
    assert output == expected_output


def test_predict_type():
    output = type(predict(loading_data()))
    expected_output = str
    assert output == expected_output
