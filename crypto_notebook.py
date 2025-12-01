from pathlib import Path
import yfinance as yf
import pandas as pd

DATA_DIR = Path("data")  # not ../data
RAW_DIR = DATA_DIR / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

symbol = "BTC-USD"
start = "2016-01-01"
end = None  # today

df = yf.download(symbol, start=start, end=end, interval="1d", progress=False)
df.reset_index(inplace=True)
df.to_csv(RAW_DIR / f"{symbol}.csv", index=False)
print("Saved:", RAW_DIR / f"{symbol}.csv")

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from pathlib import Path
sns.set_style("darkgrid")

DATA_DIR = Path("../data")
RAW_DIR = DATA_DIR / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)
symbol = "BTC-USD"
start = "2016-01-01"
end = None  # None = up to today
df = yf.download(symbol, start=start, end=end, interval="1d", progress=False)
df.reset_index(inplace=True)
df.head()
# save raw
df.to_csv(RAW_DIR / f"{symbol}.csv", index=False)
print("Saved:", RAW_DIR / f"{symbol}.csv")
import pandas as pd
import os

# ensure processed folder exists
os.makedirs("../data/processed", exist_ok=True)

# load the raw data
df = pd.read_csv("../data/raw/BTC-USD.csv", parse_dates=["Date"], index_col="Date")

# numeric columns (Adj Close removed)
numeric_cols = ["Open", "High", "Low", "Close", "Volume"]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# daily frequency
df = df.asfreq("D")

# forward fill missing values
df = df.ffill()

# save cleaned data
df.to_csv("../data/processed/BTC-USD-clean.csv")

df.head()
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 5))
plt.plot(df["Close"])
plt.title("BTC Closing Price Over Time")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.grid(True)
plt.show()
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Use only 'Close' price for forecasting
data = df[["Close"]].values

# Scaling to 0–1 range
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# Sequence length (how many past days the model sees)
sequence_length = 60  # 60 days history → predict next day

X = []
y = []

for i in range(sequence_length, len(scaled_data)):
    X.append(scaled_data[i-sequence_length:i])
    y.append(scaled_data[i])

X = np.array(X)
y = np.array(y)

# Train-test split (80% train)
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

X_train.shape, X_test.shape
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(60, 1)),
    Dropout(0.2),

    LSTM(64, return_sequences=False),
    Dropout(0.2),

    Dense(32),
    Dense(1)  # output: next day's price
])

model.compile(optimizer="adam", loss="mse")
model.summary()
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=20,
    batch_size=32
)
# Make predictions
predictions = model.predict(X_test)

# Undo scaling
predictions = scaler.inverse_transform(predictions)
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

import matplotlib.pyplot as plt

plt.figure(figsize=(12,6))
plt.plot(y_test_actual, label="Actual Price")
plt.plot(predictions, label="Predicted Price")
plt.legend()
plt.title("BTC Price Prediction - Actual vs Predicted")
plt.show()

from pathlib import Path
import numpy as np

# --- 1️⃣ Save trained LSTM model ---
MODELS_DIR = Path("../models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

model.save(MODELS_DIR / "lstm_model.h5")
print("LSTM model saved at:", MODELS_DIR / "lstm_model.h5")

# --- 2️⃣ Predict next 7 days ---
# Use the last 60 days of scaled data
last_60 = scaled_data[-60:]  # scaled_data from your preprocessing
future = []

current_input = last_60.reshape(1, 60, 1)

for _ in range(7):
    next_price = model.predict(current_input, verbose=0)[0][0]
    future.append(next_price)
    # append the new prediction to the input sequence
    current_input = np.append(current_input[:, 1:, :], [[[next_price]]], axis=1)

# Convert predictions back to original price scale
future_prices = scaler.inverse_transform(np.array(future).reshape(-1,1))

print("Next 7 days predicted BTC prices (USD):")
print(future_prices)
import matplotlib.pyplot as plt

# Last 60 days of actual prices
last_60_actual = df['Close'].values[-60:]

# Combine last 60 days and future predictions
combined = np.concatenate([last_60_actual, future_prices.flatten()])

# X-axis: days
days = np.arange(len(combined))

plt.figure(figsize=(12,6))
plt.plot(days[:60], last_60_actual, label="Recent Actual Price", color="blue")
plt.plot(days[59:], combined[59:], label="Predicted Next 7 Days", color="orange", linestyle="--", marker='o')
plt.title("BTC Price - Last 60 Days + Next 7 Days Forecast")
plt.xlabel("Days")
plt.ylabel("BTC Price (USD)")
plt.legend()
plt.grid(True)
plt.show()

import pandas as pd
import numpy as np
from pathlib import Path

# Last 60 days of actual prices
actual_prices = df['Close'].values[-60:]

# Future 7 days predicted by LSTM
scaled_last_60 = scaled_data[-60:].reshape(1, 60, 1)
future = []
current_input = scaled_last_60.copy()

for _ in range(7):
    next_price = model.predict(current_input, verbose=0)[0][0]
    future.append(next_price)
    current_input = np.append(current_input[:, 1:, :], [[[next_price]]], axis=1)

future_prices = scaler.inverse_transform(np.array(future).reshape(-1,1)).flatten()

# Combine last 60 days + next 7 days
combined_dates = pd.date_range(start=df.index[-60], periods=60+7, freq='D')
combined_lstm = np.concatenate([actual_prices, future_prices])

# Create forecast DataFrame
reports_dir = Path("reports")
reports_dir.mkdir(exist_ok=True)

fc = pd.DataFrame({
    "date": combined_dates,
    "actual": np.concatenate([actual_prices, [np.nan]*7]),
    "lstm": combined_lstm,
    "arima": np.nan,    # placeholder
    "prophet": np.nan   # placeholder
})

# Save forecast CSV
fc.to_csv(reports_dir / "test_forecasts.csv", index=False)
print("Saved forecast CSV at:", reports_dir / "test_forecasts.csv")

import pandas as pd
import numpy as np
from pmdarima import auto_arima
from prophet import Prophet
from pathlib import Path

# --- 1️⃣ ARIMA Forecast ---
def arima_forecast(df, forecast_days=7):
    train = df['Close']
    model = auto_arima(train, seasonal=False, stepwise=True, suppress_warnings=True)
    forecast = model.predict(n_periods=forecast_days)
    last_60 = train.values[-60:]
    combined = np.concatenate([last_60, forecast])
    return combined

# --- 2️⃣ Prophet Forecast ---
def prophet_forecast(df, forecast_days=7):
    prophet_df = df[['Close']].reset_index().rename(columns={'Date':'ds', 'Close':'y'})
    model = Prophet(daily_seasonality=True)
    model.fit(prophet_df)
    future = model.make_future_dataframe(periods=forecast_days)
    forecast = model.predict(future)
    last_60 = df['Close'].values[-60:]
    prophet_pred = forecast['yhat'].values[-forecast_days:]
    combined = np.concatenate([last_60, prophet_pred])
    return combined

# --- Ensure reports directory exists ---
reports_dir = Path("reports")  # relative to root folder
reports_dir.mkdir(exist_ok=True)

# --- Forecasts ---
lstm_combined = combined_lstm          # from your LSTM predictions
arima_combined = arima_forecast(df)    # df is your cleaned data
prophet_combined = prophet_forecast(df) # df is your cleaned data

# --- Combine last 60 days + next 7 days for dates ---
combined_dates = pd.date_range(start=df.index[-60], periods=60+7, freq='D')

# --- Create final forecast DataFrame ---
fc = pd.DataFrame({
    "date": combined_dates,
    "actual": np.concatenate([df['Close'].values[-60:], [np.nan]*7]),
    "lstm": lstm_combined,
    "arima": arima_combined,
    "prophet": prophet_combined
})

# --- Save CSV ---
fc.to_csv(reports_dir / "test_forecasts.csv", index=False)
print("Updated forecast CSV with LSTM, ARIMA, Prophet at:", reports_dir / "test_forecasts.csv")

arima_combined = arima_forecast(df)
prophet_combined = prophet_forecast(df)

fc = pd.DataFrame({
    "date": combined_dates,
    "actual": np.concatenate([df['Close'].values[-60:], [np.nan]*7]),
    "lstm": combined_lstm,
    "arima": arima_combined,
    "prophet": prophet_combined
})

fc.to_csv(reports_dir / "test_forecasts.csv", index=False)

from pathlib import Path
import pandas as pd
import numpy as np

reports_dir = Path("reports")
reports_dir.mkdir(exist_ok=True)

# --- Forecasts from your previous LSTM, ARIMA, Prophet ---
lstm_combined = combined_lstm
arima_combined = arima_forecast(df)
prophet_combined = prophet_forecast(df)

# --- Correct date alignment ---
last_60_dates = list(df.index[-60:])  # historical
future_7_dates = list(pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=7))  # future
combined_dates = last_60_dates + future_7_dates

# --- Build final dataframe ---
fc = pd.DataFrame({
    "date": combined_dates,
    # Only historical Actual; future set to NaN
    "actual": list(df['Close'].values[-60:]) + [np.nan]*7,
    "lstm": lstm_combined,
    "arima": arima_combined,
    "prophet": prophet_combined
})

# --- Save CSV ---
fc.to_csv(reports_dir / "test_forecasts.csv", index=False)
print("Updated forecast CSV created")





