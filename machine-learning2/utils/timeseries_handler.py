import time
import pandas as pd
import numpy as np

from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
from statsmodels.tsa.arima.model import ARIMA
from typing import Optional, Union, List, Tuple, Dict, Any
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV


def difference(dataset, interval=1):
    """
    Create a differenced series to make a time series stationary.

    Parameters
    ----------
    dataset : array-like
        The original time series data.
    interval : int, optional
        The lag interval to calculate the difference, by default 1.

    Returns
    -------
    pd.Series
        The differenced time series.
    """

    diff = []
    for i in range(interval, len(dataset)):
        value = float(dataset[i] - dataset[i - interval])
        diff.append(value)
    return pd.Series(diff)


def inverse_difference(history, yhat, interval=1):
    """
    Revert a differenced value to its original scale.

    Parameters
    ----------
    history : list
        The historical time series data.
    yhat : float
        The forecasted value to be inverted.
    interval : int, optional
        The lag interval used in differencing, by default 1.

    Returns
    -------
    float
        The value in its original scale.
    """
    return float(yhat + history[-interval])


def evaluate_arima_model(
    train_set: Union[List[float], np.ndarray, pd.Series],
    order: Tuple[int, int, int],
    val_set: Optional[Union[List[float], np.ndarray, pd.Series]] = None,
    validation: bool = False,
) -> Union[float, Tuple[float, List[float]]]:
    """
    Evaluate an ARIMA model using either cross-validation or a train-test split.

    Parameters
    ----------
    train_set : array-like
        The time series data for training.
    order : tuple of int
        The (p, d, q) order of the ARIMA model.
    y : array-like, optional
        The validation dataset for cross-validation, by default None.
    validation : bool, optional
        Whether to use cross-validation with a separate validation set, by default False.

    Returns
    -------
    Union[float, Tuple[float, List[float]]]
        - If validation is False: A dictionary with RMSE, MAE, runtime, and predictions.
        - If validation is True: A dictionary with additional validation metrics.
    """

    if validation:
        history = [x for x in train_set]
        months_in_year = 12

        results = {}
        predictions = []

        model = ARIMA(train_set, order=order)
        model_fit = model.fit()

        forecast = model_fit.forecast()

        yhat = inverse_difference(history, forecast.iloc[0], months_in_year)
        predictions.append(yhat)

        history.append(y[0])
        print(">Predicted=%.3f, Expected=%.3f" % (yhat, y[0]))

        start_time = time.perf_counter()
        for t in range(1, len(val_set)):
            diff = difference(history, months_in_year)

            model = ARIMA(diff, order=order)
            model_fit = model.fit()

            forecast = model_fit.forecast()

            yhat = inverse_difference(
                history, forecast.iloc[0], months_in_year)
            predictions.append(yhat)

            obs = val_set[t]
            history.append(obs)

            print(">Predicted=%.3f, Expected=%.3f" % (yhat, obs))
        end_time = time.perf_counter()

        mse = mean_squared_error(val_set, predictions)
        rmse = sqrt(mse)

        mae = mean_absolute_error(val_set, predictions)
        results["rmse"] = rmse
        results["mae"] = mae
        results["runtime"] = end_time - start_time
        results["y"] = val_set
        results["yhat"] = predictions

        return results

    train_size = int(len(train_set) * 0.80)
    train, test = train_set[0:train_size], train_set[train_size:]
    history = [x for x in train_set]
    months_in_year = 12

    results = {}
    predictions = []

    start_time = time.perf_counter()
    for t in range(len(test)):
        diff = difference(history, months_in_year)

        model = ARIMA(diff, order=order)
        model_fit = model.fit()

        forecast = model_fit.forecast()

        yhat = inverse_difference(history, forecast.iloc[0], months_in_year)
        predictions.append(yhat)

        obs = test[t]
        history.append(obs)
    end_time = time.perf_counter()

    mse = mean_squared_error(test, predictions)
    rmse = sqrt(mse)

    mae = mean_absolute_error(test, predictions)

    results["rmse"] = rmse
    results["mae"] = mae
    results["runtime"] = end_time - start_time
    results["y"] = test
    results["yhat"] = predictions

    return results


def create_features(df, label=None):
    if not df.index.freq:
        df.index = pd.date_range(start=df.index[0], periods=len(df), freq="MS")

    df["date"] = df.index
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["dayofyear"] = df["date"].dt.dayofyear
    df["dayofweek"] = df["date"].dt.dayofweek
    df["quarter"] = df["date"].dt.quarter
    X = df[["year", "month", "dayofyear", "dayofweek", "quarter"]]
    if label:
        y = df[label]
        return X, y
    return X


def create_lagged_features(data, window_size):
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i-window_size:i].values)  # Previous 12 months
        y.append(data[i])  # Next month's value (target)
    return np.array(X), np.array(y)


def evaluate_xgb_model(
    train_set: Union[List[float], np.ndarray, pd.Series],
    label: str = "target",
    forecast_horizon: int = 12,
    window_size: int = 12,
    lag: bool = False,
    grid_search: bool = False,
    param_grid: Optional[Dict[str, Any]] = None,
) -> Union[float, Tuple[float, List[float]]]:
    forecast_horizon = 12

    train_size = len(train_set) - forecast_horizon
    train, forecast_period = train_set[:train_size], train_set[train_size:]

    if grid_search:
        X_train, y_train = create_lagged_features(train[label], window_size)

        xgb_model = XGBRegressor()
        tscv = TimeSeriesSplit(n_splits=5)

        grid_search = GridSearchCV(
            estimator=xgb_model,
            param_grid=param_grid,
            cv=tscv,
            scoring="neg_mean_squared_error",
            verbose=1,
            n_jobs=-1
        )

        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_

        print(f"Best Parameters: {grid_search.best_params_}")
        print(
            f"Best CV Score (neg_mean_squared_error): {grid_search.best_score_}")

        results = {}
        predictions = []

        start_time = time.perf_counter()
        for i in range(forecast_horizon):
            # Create feature for the next point to forecast using the last 'window_size' values
            X_next = train_set[label][train_size - window_size +
                                      i: train_size + i].values.reshape(1, -1)

            # Forecast the next point
            next_forecast = best_model.predict(X_next)
            predictions.append(next_forecast[0])

            # Update the training set with the observed value after the forecast
            X_train = np.vstack([X_train, X_next])
            y_train = np.append(y_train, forecast_period[label].iloc[i])

            best_model.fit(X_train, y_train)
        end_time = time.perf_counter()

        rmse = np.sqrt(mean_squared_error(
            forecast_period[label][:forecast_horizon], predictions))
        results["rmse"] = rmse
        print(f"RMSE for XGBoost 1-Year Forecast: {rmse}")

        mae = mean_absolute_error(
            forecast_period[label][:forecast_horizon], predictions)
        results["mae"] = mae
        print(f"MAE for XGBoost 1-Year Forecast: {mae}")

        results["runtime"] = end_time - start_time
        results["y"] = forecast_period
        results["yhat"] = predictions

        return results

    if lag:
        X_train, y_train = create_lagged_features(train[label], window_size)

        results = {}
        predictions = []

        start_time = time.perf_counter()
        for i in range(forecast_horizon):
            # Create feature for the next point to forecast using the last 12 months
            X_next = train_set[label][train_size -
                                      window_size+i:train_size+i].values.reshape(1, -1)

            # Initialize XGBoost model
            xgb_model = XGBRegressor()

            # Train the model
            xgb_model.fit(X_train, y_train)

            # Forecast the next point
            next_forecast = xgb_model.predict(X_next)
            predictions.append(next_forecast[0])

            # Update training set with actual observed value after the forecast
            X_train = np.vstack([X_train, X_next])
            y_train = np.append(y_train, forecast_period[label].iloc[i])
        end_time = time.perf_counter()

        rmse = np.sqrt(mean_squared_error(
            forecast_period[label][:forecast_horizon], predictions))
        results["rmse"] = rmse
        print(f"RMSE for XGBoost 1-Year Forecast: {rmse}")

        mae = mean_absolute_error(
            forecast_period[label][:forecast_horizon], predictions)
        results["mae"] = mae
        print(f"MAE for XGBoost 1-Year Forecast: {mae}")

        results["runtime"] = end_time - start_time
        results["y"] = forecast_period
        results["yhat"] = predictions

        return results

    X_train, y_train = create_features(train_set, label=label)
    X_forecast = create_features(forecast_period)

    results = {}
    xgb_model = XGBRegressor()
    xgb_model.fit(X_train, y_train)

    start_time = time.perf_counter()
    xgboost_forecast = xgb_model.predict(X_forecast[:forecast_horizon])
    end_time = time.perf_counter()

    print("Forecast:", xgboost_forecast)

    rmse = np.sqrt(mean_squared_error(
        forecast_period[label][:forecast_horizon], xgboost_forecast))
    results["rmse"] = rmse
    print(f"RMSE for XGBoost 1-Year Forecast: {rmse}")

    mae = mean_absolute_error(
        forecast_period[label][:forecast_horizon], xgboost_forecast)
    results["mae"] = mae
    print(f"MAE for XGBoost 1-Year Forecast: {mae}")

    results["runtime"] = end_time - start_time
    results["y"] = forecast_period[label][:forecast_horizon]
    results["yhat"] = xgboost_forecast

    return results
