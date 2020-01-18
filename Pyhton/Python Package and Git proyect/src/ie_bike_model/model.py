import pandas as pd
import numpy as np
from os import environ
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from joblib import dump, load
from ie_bike_model.preprocess import (
    whole_preprocessing,
    convert_date_time,
    skewness,
    feature_engeneering,
    convert_to_category,
    get_dummies,
    cleaning_reg_ex,
)
from ie_bike_model.load_data import loading_data


def test_train_split(data=whole_preprocessing(loading_data())):
    hour_d_train = data.iloc[0:15211]
    hour_d_test = data.iloc[15212:17379]
    hour_d_train = hour_d_train.drop(
        columns=["dteday", "casual", "atemp", "registered"]
    )
    hour_d_test = hour_d_test.drop(columns=["dteday", "casual", "registered", "atemp"])

    # seperate the independent and target variable on testing data
    hour_d_train_x = hour_d_train.drop(columns=["cnt"], axis=1)
    hour_d_train_y = hour_d_train["cnt"]

    # seperate the independent and target variable on test data
    hour_d_test_x = hour_d_test.drop(columns=["cnt"], axis=1)
    hour_d_test_y = hour_d_test["cnt"]
    return hour_d_train_x, hour_d_train_y, hour_d_test_x, hour_d_test_y


def train_and_persist(train_x, train_label, test_x, test_label):
    """
    Fits an XGBoost with the best hyperparameters for hour.csv data. Optional arguments are hour_d_train_x, hour_d_train_y which is
    preprocessed train set split into labels and features. By default takes preprocessed hour_d_train_x, hour_d_train_y with the function
    preprocessing from the module preprocess_data

    Args:

    train_x: dataset with train features
    train_label: dataset with train labels
    test_x: dataset with test features
    test_label: dataset with test labels

    Returns:

    Fitted XGBoost classifier.
    """
    hour_d_train_x = train_x
    hour_d_train_y = train_label
    hour_d_test_x = test_x
    hour_d_test_y = test_label

    xgb = XGBRegressor(
        max_depth=6,
        learning_rate=0.06,
        n_estimators=1000,
        objective="reg:squarederror",
        subsample=0.8,
        colsample_bytree=0.5,
        seed=1234,
        gamma=1.5,
    )

    clf = xgb.fit(hour_d_train_x, hour_d_train_y, verbose=10)
    dump(clf, "model.pkl")

    return clf


def predict(data):
    """
    Predicts the fitted model. Takes as an argument raw data, preprocesses it and does test-train splitting

    Args:

    data: directory with raw data without any preprocessing or test train splitting

    Returns:

    result_xgb(int64): predicted count of bikes

    """

    hour_d = whole_preprocessing(data)

    hour_d_train_x, hour_d_train_y, hour_d_test_x, hour_d_test_y = test_train_split(
        hour_d
    )

    try:
        result_xgb = clf.predict(hour_d_test_x)
        print("R-squared for Train: %.2f" % clf.score(hour_d_train_x, hour_d_train_y))
        print("R-squared for Test: %.2f" % clf.score(hour_d_test_x, hour_d_test_y))
        RMSE = np.sqrt(np.mean((hour_d_test_y * 2 - result_xgb * 2) ** 2))
        MSE = RMSE ** 2
        print("MSE ={}".format(MSE))
        print("RMSE = {}".format(RMSE))
    except:
        try:
            clf = load("model.pkl")
            result_xgb = clf.predict(hour_d_test_x)
            print(
                "R-squared for Train: %.2f" % clf.score(hour_d_train_x, hour_d_train_y)
            )
            print("R-squared for Test: %.2f" % clf.score(hour_d_test_x, hour_d_test_y))
            RMSE = np.sqrt(np.mean((hour_d_test_y * 2 - result_xgb * 2) ** 2))
            MSE = RMSE ** 2
            print("MSE ={}".format(MSE))
            print("RMSE = {}".format(RMSE))
        except:
            try:
                clf = train_and_persist(
                    hour_d_train_x, hour_d_train_y, hour_d_test_x, hour_d_test_y
                )
                result_xgb = clf.predict(hour_d_test_x)
                print(
                    "R-squared for Train: %.2f"
                    % clf.score(hour_d_train_x, hour_d_train_y)
                )
                print(
                    "R-squared for Test: %.2f" % clf.score(hour_d_test_x, hour_d_test_y)
                )
                RMSE = np.sqrt(np.mean((hour_d_test_y * 2 - result_xgb * 2) ** 2))
                MSE = RMSE ** 2
                print("MSE ={}".format(MSE))
                print("RMSE = {}".format(RMSE))
            except:
                pass

    return "predictions-available"
