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
show_trend = st.sidebar.checkbox("Show price trends")
show_box = st.sidebar.checkbox("Show boxplots by month & year")
show_forecast = st.sidebar.checkbox("Run Prophet forecast")

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
    st.subheader("Milk Price Over Time")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(data['date'], data['milk_price'], label="Milk Price", color="blue")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (₹/L)")
    ax.legend()
    st.pyplot(fig)

# Boxplots by month & year
if show_box:
    st.subheader("Seasonality Patterns")
    data['year'] = data['date'].dt.year
    data['month'] = data['date'].dt.month

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.boxplot(x="month", y="milk_price", data=data, ax=ax)
        ax.set_title("Monthly Distribution")
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.boxplot(x="year", y="milk_price", data=data, ax=ax)
        ax.set_title("Yearly Distribution")
        st.pyplot(fig)

# Prophet forecasting
if show_forecast:
    st.subheader("Prophet Forecasting Model")

    # Split train/test
    train = data[data['date'] < '2024-01-01'].copy()
    test = data[data['date'] >= '2024-01-01'].copy()

    # Prophet setup
    model = Prophet(yearly_seasonality=True, daily_seasonality=False)
    exog_vars = [
        'inflation_rate', 'feed_cost', 'energy_cost', 'global_price',
        'transport_cost', 'labor_rate', 'local_production', 'rainfall_mm',
        'export_volume'
    ]
    for var in exog_vars:
        model.add_regressor(var)

    # Fit model
    model.fit(train.rename(columns={'date':'ds', 'milk_price':'y'}))

    # Forecast
    future_test = test[['date'] + exog_vars].rename(columns={'date':'ds'})
    forecast = model.predict(future_test)

    # Metrics
    y_true = test['milk_price'].values
    y_pred = forecast['yhat'].values
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    st.write(f"*MAE:* {mae:.2f}")
    st.write(f"*RMSE:* {rmse:.2f}")

    # Plot forecast
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(train['date'], train['milk_price'], label="Train", color="gray")
    ax.plot(test['date'], y_true, label="Actual Test", color="blue")
    ax.plot(test['date'], y_pred, label="Forecast", color="red")
    ax.legend()
    st.pyplot(fig)

    # Prophet built-in forecast components
    st.subheader("Forecast Components")
    fig2 = model.plot_components(forecast)
    st.pyplot(fig2)
