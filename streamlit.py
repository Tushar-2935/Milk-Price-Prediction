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

st.subheader("Price Trends")
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(data["date"], data["milk_price"], label="Milk Price", color="blue")
ax.set_xlabel("Date")
ax.set_ylabel("Price")
ax.legend()
st.pyplot(fig, clear_figure=True)

st.subheader("Milk Price Forecast (Prophet Model)")

    # Train Prophet
prophet_data = data.rename(columns={'date': 'ds', 'milk_price': 'y'})
model = Prophet(yearly_seasonality=True, daily_seasonality=False)
model.fit(prophet_data)

    # Future predictions (till 2030 for flexibility)
future = model.make_future_dataframe(periods=3*365)
forecast = model.predict(future)

    # Dropdown for year selection
years = sorted(list(set(forecast['ds'].dt.year)))
selected_year = st.selectbox("Select Year for Forecast", years, index=years.index(2026))

    # Filter selected year & group by month
forecast_year = forecast[forecast['ds'].dt.year == selected_year]
monthly_forecast = forecast_year.groupby(forecast_year['ds'].dt.to_period("M")).mean(numeric_only=True)

    # Plot monthly forecast
fig2, ax2 = plt.subplots(figsize=(10, 4))
ax2.plot(monthly_forecast.index.to_timestamp(), monthly_forecast['yhat'], marker="o", label="Forecast")
ax2.fill_between(monthly_forecast.index.to_timestamp(),monthly_forecast['yhat_lower'], monthly_forecast['yhat_upper'],alpha=0.2, color="blue", label="Confidence Interval")
ax2.set_title(f"Monthly Forecast for {selected_year}")
ax2.set_xlabel("Month")
ax2.set_ylabel("Milk Price")
ax2.legend()
st.pyplot(fig2, clear_figure=True)

    # -----------------
    # Accuracy Check (Forecast vs Actual)
    # -----------------
st.subheader("📊 Forecast vs Actual (Backtesting)")

    # Compare last 2 years of actual vs forecast
history_years = [2023, 2024, 2025]
actual = prophet_data[prophet_data['ds'].dt.year.isin(history_years)]
forecast_hist = forecast[forecast['ds'].dt.year.isin(history_years)]

    # Merge
comparison = pd.merge(actual, forecast_hist[['ds', 'yhat']], on='ds', how='inner')

    # Plot
fig_acc, ax_acc = plt.subplots(figsize=(10, 4))
ax_acc.plot(comparison['ds'], comparison['y'], label="Actual", color="black")
ax_acc.plot(comparison['ds'], comparison['yhat'], label="Forecast", color="orange")
ax_acc.set_title("Actual vs Forecasted Milk Prices")
ax_acc.set_xlabel("Date")
ax_acc.set_ylabel("Price")
ax_acc.legend()
st.pyplot(fig_acc, clear_figure=True)
