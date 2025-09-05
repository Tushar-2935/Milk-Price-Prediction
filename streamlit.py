import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Page setup
st.set_page_config(page_title="Milk Price Forecasting", layout="wide")
st.title("🥛 Milk Price Forecasting in Maharashtra (2015–2025)")

# Load Data
@st.cache_data
def load_data():
    df = pd.read_csv("milk_price_prediction_maharashtra_2015_2025_daily_final.csv", parse_dates=["date"])
    return df

data = load_data()

# Sidebar controls
st.sidebar.header("Options")
show_raw = st.sidebar.checkbox("Show raw data")
show_corr = st.sidebar.checkbox("Show correlation heatmap")
show_prophet = st.sidebar.checkbox("Show Prophet Analysis")
show_trend = st.sidebar.checkbox("Actual Vs Prediction")
show_scenario_analysis = st.sidebar.checkbox("Scenario Analysis")

# Show raw data
if show_raw:
    st.subheader("Raw Data Sample")
    st.dataframe(data.head(20))

# Correlation heatmap
if show_corr:
    st.subheader("Correlation with Milk Price")
    fig, ax = plt.subplots(figsize=(8, 6))
    corr = data.select_dtypes(include=np.number).corr()
    sns.heatmap(
        corr[['milk_price']].sort_values(by='milk_price', ascending=False),
        annot=True, cmap='coolwarm', ax=ax
    )
    st.pyplot(fig)

# Time series trend
if show_trend:
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
# Boxplots by month & year
if show_prophet:
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
    ax2.fill_between(monthly_forecast.index.to_timestamp(),
                     monthly_forecast['yhat_lower'], monthly_forecast['yhat_upper'],
                     alpha=0.2, color="blue", label="Confidence Interval")
    ax2.set_title(f"Monthly Forecast for {selected_year}")
    ax2.set_xlabel("Month")
    ax2.set_ylabel("Milk Price")
    ax2.legend()
    st.pyplot(fig2, clear_figure=True)
# Prophet forecasting
if show_scenario_analysis:
    st.subheader("Scenario Analysis")

    feed_cost_change = st.slider("Feed Cost Change (%)", -20, 20, 0)
    inflation_change = st.slider("Inflation Change (%)", -20, 20, 0)

    st.write(f"📌 Scenario: Feed cost {feed_cost_change}%, Inflation {inflation_change}%")

    # Apply adjustments
    scenario_forecast = forecast.copy()
    scenario_forecast['yhat'] = (
        scenario_forecast['yhat'] * (1 + feed_cost_change/100 + inflation_change/200)
    )

    # Show for 2026 as example
    forecast_2026 = forecast[forecast['ds'].dt.year == 2026]
    scenario_2026 = scenario_forecast[scenario_forecast['ds'].dt.year == 2026]

    fig3, ax3 = plt.subplots(figsize=(10, 4))
    ax3.plot(forecast_2026['ds'], forecast_2026['yhat'], label="Baseline Forecast")
    ax3.plot(scenario_2026['ds'], scenario_2026['yhat'], label="Scenario Forecast", linestyle="--")
    ax3.set_title("Scenario Forecast for 2026")
    ax3.set_xlabel("Date")
    ax3.set_ylabel("Milk Price")
    ax3.legend()
    st.pyplot(fig3, clear_figure=True)
