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
range(0, 4)# Time-Series-Forecasting
