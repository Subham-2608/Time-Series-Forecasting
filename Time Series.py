##### Library Import
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA

##### Read Data
df = pd.read_csv("C:/Users/searc/Documents/Company/TIME FORECASTING/all_stocks_5yr.csv")
for dirname, _,filenames in os.walk("C:/Users/searc/Documents/Company/TIME FORECASTING"):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df = df[df["Name"] == "AAL"].copy().dropna()
df['date'] = pd.to_datetime(df["date"])
df = df.sort_values("date")
df = df.set_index("date")

def arima_forecasting(df):

    d_values = {}
    p_values = q_values = range(0, 4)
    best_model = {}
    train_split = len(df) * 0.8

    for column in df.columns[:5]:

        best_aic = np.inf
        best_order = None
        predictions = []
        d = 0

        train = df[column].iloc[:int(train_split)].dropna()
        series = train.dropna()
        test = df[column].iloc[int(train_split):].dropna()

        while True:

            result = adfuller(series)

            p_value = result[1]
            adf_test_statistic = result[0]

            if p_value < 0.05:

                print(
                    f"The series '{column}' is stationary "
                    f"(p-value: {p_value:.4f}, "
                    f"ADF Statistic: {adf_test_statistic:.4f})"
                )

                d_values[column] = d
                break

            series = series.diff().dropna()

            print(
                f"The series '{column}' is non-stationary "
                f"(p-value: {p_value:.4f}, "
                f"ADF Statistic: {adf_test_statistic:.4f})"
            )

            d += 1

            if d >= 3:

                print("Maximum differencing reached")

                d_values[column] = d
                break

        for p in p_values:

            for q in q_values:

                order = (p, d, q)

                try:

                    model = ARIMA(
                        train.to_numpy(),
                        order=order,
                        enforce_stationarity=False,
                        enforce_invertibility=False
                    )

                    model_fit = model.fit()

                    aic = model_fit.aic

                    if aic < best_aic:

                        best_aic = aic
                        best_order = order
                        best_model[column] = model_fit

                except:

                    continue

        history = list(train)

        for actual_value in test:

            model = ARIMA(
                history,
                order=best_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            )

            model_fit = model.fit()

            prediction = model_fit.forecast(steps=1)

            prediction = prediction[0]

            predictions.append(prediction)

            history.append(actual_value)

        predictions = np.array(predictions)

        r2 = r2_score(test, predictions)

        rmse = np.sqrt(
            mean_squared_error(test, predictions)
        )

        print(
            f"Best ARIMA order for '{column}': "
            f"{best_order}, "
            f"AIC: {best_aic:.4f}, "
            f"R2: {r2:.4f}, "
            f"RMSE: {rmse:.4f}"
        )

    return d_values, best_model