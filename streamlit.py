import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet

# -----------------
# Title & Sidebar
# -----------------
st.set_page_config(page_title="Milk Price Prediction", layout="wide")
st.title("🥛 Milk Price Prediction Dashboard")

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv(
        r"C:\Users\hp\Downloads\milk_price_prediction_maharashtra_2015_2025_daily_final.csv",
        parse_dates=["date"]
    )

data = load_data()

# Tabs
tab1, tab2, tab3 = st.tabs(["📊 EDA", "📉📈 Forecast", "📈 Scenario Analysis"])

# -----------------
# Tab 1: EDA
# -----------------
with tab1:
    st.subheader("Price Trends")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(data["date"], data["milk_price"], label="Milk Price", color="blue")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    st.pyplot(fig, clear_figure=True)

# -----------------
# Tab 2: Forecast
# -----------------
with tab2:
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


# -----------------
# Tab 3: Scenario Analysis
# -----------------
with tab3:
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
