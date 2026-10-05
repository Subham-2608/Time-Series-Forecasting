# Time Series Forecasting Using ARIMA

## Overview

This project implements a time-series forecasting pipeline using the ARIMA
(AutoRegressive Integrated Moving Average) model to forecast stock-market
variables.

The project uses historical stock data and demonstrates the complete workflow
from data preprocessing and stationarity testing to ARIMA model selection,
walk-forward forecasting, and model evaluation.

## Objectives

- Analyze historical stock-market time-series data.
- Check stationarity using the Augmented Dickey-Fuller (ADF) test.
- Determine the required order of differencing (`d`).
- Identify suitable ARIMA `(p,d,q)` parameters using AIC.
- Perform one-step-ahead walk-forward forecasting.
- Evaluate forecasting performance using RMSE and R².
- Build a reusable Python function for the forecasting workflow.

## Dataset

The project uses the `all_stocks_5yr.csv` dataset.

The dataset contains the following variables:

- `date` – Trading date
- `open` – Opening stock price
- `high` – Highest stock price
- `low` – Lowest stock price
- `close` – Closing stock price
- `volume` – Trading volume
- `Name` – Stock ticker/company identifier

For this project, the stock ticker `AAL` is selected for analysis.

## Methodology

The forecasting pipeline consists of the following steps:

### 1. Data Preprocessing

- Load the CSV dataset.
- Select the required stock.
- Remove missing observations.
- Convert the date column to datetime format.
- Sort observations chronologically.
- Set the date as the index.

### 2. Train-Test Split

The dataset is divided chronologically:

- 80% → Training data
- 20% → Testing data

The test data is kept separate for evaluating forecasting performance.

### 3. Stationarity Testing

The Augmented Dickey-Fuller (ADF) test is used to determine whether each
time series is stationary.

If the series is non-stationary, differencing is applied until the series
becomes stationary or a maximum differencing order of 3 is reached.

The resulting differencing order is used as the `d` parameter of ARIMA.

### 4. ARIMA Model Selection

Different combinations of:

- `p` = autoregressive order
- `d` = differencing order
- `q` = moving-average order

are evaluated.

The values of `p` and `q` are searched from:

```python
range(0, 4)

## Results

The ARIMA models were evaluated on the held-out 20% test dataset using
RMSE and R². The ARIMA `(p,d,q)` parameters were selected based on the
lowest AIC among the tested parameter combinations.

| Variable | Differencing (d) | Best ARIMA Model | AIC | RMSE | R² |
|----------|------------------:|------------------|----:|-----:|---:|
| Open     | 1 | (0, 1, 3) | 2648.6119 | 0.8815 | 0.9466 |
| High     | 1 | (1, 1, 3) | 2334.8879 | 0.8714 | 0.9482 |
| Low      | 1 | (0, 1, 3) | 2531.2240 | 0.8532 | 0.9495 |
| Close    | 1 | (0, 1, 3) | 2545.8345 | 0.8922 | 0.9451 |

### Key Observations

- The `open`, `high`, `low`, and `close` series were initially
  non-stationary and required first-order differencing (`d=1`).
- The `low` variable achieved the lowest RMSE (0.8532) and highest R²
  (0.9495) among the evaluated variables.
- The `high` variable achieved an R² of 0.9482 with an RMSE of 0.8714.
- The selected ARIMA models achieved R² values above 0.94 for all four
  evaluated price variables.
- Forecasting was performed using walk-forward one-step-ahead prediction,
  where each newly observed actual value was incorporated into the history
  before generating the next prediction.

