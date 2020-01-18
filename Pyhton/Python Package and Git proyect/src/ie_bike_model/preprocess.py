from scipy.stats import skew
import numpy as np
import pandas as pd
import re

from ie_bike_model.load_data import loading_data

"""
Doing preprocessing steps: eliminating skewness, adding new custom features, dummies, train-test splitting.
Requires an optional argument of input data. Default value takes the data which is result of the function loading data() - by default it
loads 'hour.csv'.
"""


def convert_date_time(data=loading_data()):
    data["dteday"] = pd.to_datetime(data["dteday"])
    return data


def skewness(data=loading_data()):
    data["windspeed"] = np.log1p(data.windspeed)
    data["cnt"] = np.sqrt(data.cnt)
    return data


def feature_engeneering(data=loading_data()):
    data["IsOfficeHour"] = np.where(
        (data["hr"] >= 9) & (data["hr"] < 17) & (data["weekday"] == 1), 1, 0
    )
    data["IsDaytime"] = np.where((data["hr"] >= 6) & (data["hr"] < 22), 1, 0)
    data["IsRushHourMorning"] = np.where(
        (data["hr"] >= 6) & (data["hr"] < 10) & (data["weekday"] == 1), 1, 0
    )
    data["IsRushHourEvening"] = np.where(
        (data["hr"] >= 15) & (data["hr"] < 19) & (data["weekday"] == 1), 1, 0
    )
    data["IsHighSeason"] = np.where((data["season"] == 3), 1, 0)

    bins = [0, 0.19, 0.49, 0.69, 0.89, 1]
    data["temp_binned"] = pd.cut(data["temp"], bins)
    data["hum_binned"] = pd.cut(data["hum"], bins)
    return data


def convert_to_category(data=loading_data()):
    convert_to_category = [
        "season",
        "yr",
        "mnth",
        "hr",
        "holiday",
        "weekday",
        "workingday",
        "weathersit",
        "IsOfficeHour",
        "IsDaytime",
        "IsRushHourMorning",
        "IsRushHourEvening",
        "IsHighSeason",
        "temp_binned",
        "hum_binned",
    ]
    for col in convert_to_category:
        data[col] = data[col].astype("category")
    return data


def get_dummies(data=loading_data()):
    data = pd.get_dummies(data)
    return data


def cleaning_reg_ex(data=loading_data()):
    regex = re.compile(r"\[|\]|<", re.IGNORECASE)
    data.columns = [
        regex.sub("_", col) if any(x in str(col) for x in set(("[", "]", "<"))) else col
        for col in data.columns.values
    ]
    return data


def whole_preprocessing(data):
    hour = data
    hour = convert_date_time(hour)
    hour = skewness(hour)
    hour = feature_engeneering(hour)
    hour = convert_to_category(hour)
    hour_d = get_dummies(hour)
    hour_d = cleaning_reg_ex(hour_d)
    return hour_d
