# --- Install Required Libraries ---
!pip install yfinance prophet --quiet

# --- Imports ---
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# --- Step 1: Load Data ---
df = yf.download("AAPL", start="2015-01-01", end="2023-12-31")
df = df[['Close']]
df.dropna(inplace=True)

# --- Step 2: Train-Test Split ---
split = int(len(df) * 0.8)
train, test = df[:split], df[split:]

# --- Step 3: ARIMA Model ---
df_arima = df.copy()
df_arima.index = pd.to_datetime(df_arima.index)
df_arima = df_arima.asfreq('B')
train_arima = df_arima[:split]
test_arima = df_arima[split:]

model_arima = ARIMA(train_arima, order=(5, 1, 0))
model_arima_fit = model_arima.fit()
forecast_arima = model_arima_fit.forecast(steps=len(test_arima))
forecast_arima.index = test_arima.index

# --- Step 4: LSTM Model ---
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)

def create_sequences(data, seq_length=60):
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i])
        y.append(data[i])
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_data)
split_seq = int(0.8 * len(X))
X_train, X_test = X[:split_seq], X[split_seq:]
y_train, y_test = y[:split_seq], y[split_seq:]

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

model_lstm = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    LSTM(50),
    Dense(1)
])
model_lstm.compile(optimizer='adam', loss='mean_squared_error')
model_lstm.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)

predictions = model_lstm.predict(X_test)
predicted_prices = scaler.inverse_transform(predictions)
true_prices = scaler.inverse_transform(y_test)

# --- Step 5: Prophet Model ---
df_prophet = df.reset_index().rename(columns={"Date": "ds", "Close": "y"})
df_train = df_prophet[:split]
df_test = df_prophet[split:]

model_prophet = Prophet()
model_prophet.fit(df_train)
future = model_prophet.make_future_dataframe(periods=len(df_test))
forecast = model_prophet.predict(future)

forecast_filtered = forecast[['ds', 'yhat']].set_index('ds').join(df.set_index('Date')).dropna()

# --- Step 6: Evaluation ---
def print_metrics(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    print(f"🔹 {name}")
    print(f"  MAE: {mae:.2f}")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  MAPE: {mape:.2f}%\n")

print_metrics("ARIMA", test['Close'], forecast_arima)
print_metrics("LSTM", true_prices, predicted_prices)
print_metrics("Prophet", forecast_filtered['Close'], forecast_filtered['yhat'])

# --- Step 7: Optional Plots ---
plt.figure(figsize=(12, 6))
plt.plot(test.index, test['Close'], label='Actual', color='black')
plt.plot(test.index, forecast_arima, label='ARIMA', linestyle='--')
plt.title("ARIMA Forecast vs Actual")
plt.legend()
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(true_prices, label='Actual', color='black')
plt.plot(predicted_prices, label='LSTM', linestyle='--')
plt.title("LSTM Forecast vs Actual")
plt.legend()
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(forecast_filtered.index, forecast_filtered['Close'], label='Actual', color='black')
plt.plot(forecast_filtered.index, forecast_filtered['yhat'], label='Prophet', linestyle='--')
plt.title("Prophet Forecast vs Actual")
plt.legend()
plt.show()
