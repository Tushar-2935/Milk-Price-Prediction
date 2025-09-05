import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Title
st.title("Milk Price Forecasting in Maharashtra")

# Load Data
DATA_PATH = "milk_price_prediction_maharashtra_2015_2025_daily_final.csv"
data = pd.read_csv(DATA_PATH, parse_dates=["date"])

# Show raw data
if st.checkbox("Show raw data"):
    st.write(data.head())

# Plot correlation heatmap
if st.checkbox("Show correlation heatmap"):
    fig, ax = plt.subplots(figsize=(10,6))
    corr = data.select_dtypes(include=np.number).corr()
    sns.heatmap(corr[['milk_price']].sort_values(by='milk_price', ascending=False), annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

# Prophet model
st.subheader("Prophet Forecast")
train = data[data['date'] < '2024-01-01'].copy()
test = data[data['date'] >= '2024-01-01'].copy()

model = Prophet(yearly_seasonality=True, daily_seasonality=False)
exog_vars = ['inflation_rate','feed_cost','energy_cost','global_price','transport_cost','labor_rate','local_production','rainfall_mm','export_volume']
for var in exog_vars:
    model.add_regressor(var)

model.fit(train.rename(columns={'date':'ds','milk_price':'y'}))
future_test = test[['date']+exog_vars].rename(columns={'date':'ds'})
forecast = model.predict(future_test)

y_true = test['milk_price'].values
y_pred = forecast['yhat'].values
mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))

st.write("MAE:", mae, " RMSE:", rmse)

fig, ax = plt.subplots(figsize=(12,6))
ax.plot(test['date'], y_true, label="Actual")
ax.plot(test['date'], y_pred, label="Forecast")
ax.legend()
st.pyplot(fig)
