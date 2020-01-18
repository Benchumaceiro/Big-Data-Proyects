import pytest
import pandas as pd
from pandas.util.testing import assert_frame_equal
from datetime import date, datetime
from ie_bike_model.preprocess import (
    convert_date_time,
    skewness,
    feature_engeneering,
    convert_to_category,
    get_dummies,
    cleaning_reg_ex,
    whole_preprocessing,
)


# def test_create_dupes():
# data = pd.DataFrame({'hr': [1, 2], 'season': ['summer', 'spring'], 'temp': [35, 20], 'hum': [0.23, 0.55], 'weekday': ['fri', 'sun']})
# expected_output = pd.DataFrame({'hr': [1, 2], 'season': ['summer', 'spring'], 'temp': [35, 20], 'hum': [0.23, 0.55], 'weekday': ['fri', 'sun'],
#'hr2': [1, 2], 'season2': ['summer', 'spring'], 'temp2': [35, 20], 'hum2': [0.23, 0.55], 'weekday2': ['fri', 'sun']})
# output = create_dupes(data)
# assert_frame_equal(expected_output, output)


def test_convert_date_time():
    output = convert_date_time()["dteday"]
    assert output.dtype == "<M8[ns]"


def test_skewness():
    data = pd.DataFrame({"windspeed": [1, 2], "cnt": [3, 10]})
    output = skewness(data)[["windspeed", "cnt"]]
    expected_output = pd.DataFrame(
        {"windspeed": [0.693147, 1.098612], "cnt": [1.732051, 3.162278]}
    )
    assert_frame_equal(expected_output, output)


def test_feature_engeneering():
    data = pd.DataFrame(
        {
            "hr": [10, 2, 7, 15],
            "weekday": [1, 1, 0, 0],
            "season": [3, 4, 1, 3],
            "temp": [35, 20, 21, 45],
            "hum": [0.23, 0.55, 0.67, 0.98],
        }
    )
    output = feature_engeneering(data)[
        [
            "IsOfficeHour",
            "IsDaytime",
            "IsRushHourMorning",
            "IsRushHourEvening",
            "IsHighSeason",
        ]
    ]
    expected_output = pd.DataFrame(
        {
            "IsOfficeHour": [1, 0, 0, 0],
            "IsDaytime": [1, 0, 1, 1],
            "IsRushHourMorning": [0, 0, 0, 0],
            "IsRushHourEvening": [0, 0, 0, 0],
            "IsHighSeason": [1, 0, 0, 1],
        }
    )
    assert_frame_equal(expected_output, output)


def test_convert_to_category():
    data = pd.DataFrame(
        {
            "season": ["a", "b"],
            "weekday": ["a", "c"],
            "workingday": ["a", "b"],
            "yr": ["a", "b"],
            "weathersit": ["a", "b"],
            "mnth": ["a", "b"],
            "hr": ["a", "b"],
            "holiday": ["a", "b"],
            "IsOfficeHour": ["a", "b"],
            "IsDaytime": ["a", "b"],
            "IsRushHourMorning": ["a", "b"],
            "IsRushHourEvening": ["a", "b"],
            "IsHighSeason": ["a", "b"],
            "temp_binned": ["a", "b"],
            "hum_binned": ["a", "b"],
        }
    )
    output = list(convert_to_category(data).dtypes)
    expected_output = list(data.astype("category").dtypes)
    assert output == expected_output


def test_get_dummies():
    data = pd.DataFrame({"season": ["a", "b"]})
    output = get_dummies(data)
    expected_output = pd.DataFrame({"season_a": [1, 0], "season_b": [0, 1]}).astype(
        "uint8"
    )
    assert_frame_equal(expected_output, output)
