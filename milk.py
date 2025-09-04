#Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#reading the dataset
data = pd.read_csv(r'C:\Users\hp\Downloads\milk_price_prediction_maharashtra_2015_2025_daily_final.csv')
data.head()

data.shape

data.describe()

# %%
data.isnull().sum() #Checking for null values

# %% [markdown]
# No missing values

# %%
data.dtypes

# %%
data['date'] = pd.to_datetime(data['date'])

# %%
data.dtypes

# %%
data['year'] = data['date'].dt.year
data['month'] = data['date'].dt.month
data['day'] = data['date'].dt.day
data['day_of_week'] = data['date'].dt.dayofweek

# %%
data = pd.get_dummies(data, columns=['season', 'government_policy', 'state'], drop_first=True)

# %%
data.head()

# %%
data.columns

# %%
plt.figure(figsize=(10,6))
corr = data.select_dtypes(include=np.number).corr()
sns.heatmap(corr[['milk_price']].sort_values(by='milk_price', ascending=False), annot=True, cmap='coolwarm')
plt.title("Correlation of Features with Milk Price")
plt.show()

# %%
plt.figure(figsize=(10,6))
sns.boxplot(x='season_summer', y='milk_price', data=data)
plt.title("Milk Price Distribution in Summer")
plt.show()

plt.figure(figsize=(10,6))
sns.boxplot(x='season_winter', y='milk_price', data=data)
plt.title("Milk Price Distribution in Winter")
plt.show()

# %%
plt.figure(figsize=(12, 6))
plt.plot(data['date'], data['energy_cost'])
plt.xlabel('Date')
plt.ylabel('Energy Cost')
plt.title('Energy Cost Over Time')
plt.grid(True)
plt.show()

# %%
plt.figure(figsize=(12, 6))
plt.plot(data['date'], data['global_price'])
plt.xlabel('Date')
plt.ylabel('Global Price')
plt.title('Global Price Over Time')
plt.grid(True)
plt.show()

# %%
plt.figure(figsize=(12, 6))
plt.plot(data['date'], data['milk_price'])
plt.xlabel('Date')
plt.ylabel('Milk Price')
plt.title('Milk Price Over Time in Maharashtra')
plt.grid(True)
plt.show()

# %%
plt.figure(figsize=(8,5))
sns.histplot(data['milk_price'], bins=30, kde=True, color='green')
plt.title("Distribution of Milk Prices")
plt.show()

#Training prophet model
#Importing libraries
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
from sklearn.metrics import mean_absolute_error, mean_squared_error

# %%
prophet_data = data[['date', 'milk_price']].rename(columns={'date': 'ds', 'milk_price': 'y'})
prophet_data_exog = prophet_data.copy()
# Split data into train and test data , data before 2024 is train data and data after 2024 is test data
train = data[data['date'] < '2024-01-01'].copy()
test = data[data['date'] >= '2024-01-01'].copy()

# Fit model on train set
model1 = Prophet(yearly_seasonality=True, daily_seasonality=False)
#Taking variables which have high correlation with milk price
exog_vars = ['inflation_rate', 'feed_cost', 'energy_cost','global_price','transport_cost','labor_rate','local_production','rainfall_mm','export_volume']
for var in exog_vars:
    model1.add_regressor(var) # Add regressors before fitting
model1.fit(train.rename(columns={'date': 'ds', 'milk_price': 'y'}))

# Create future dataframe for test period
future_test = test[['date'] + exog_vars].rename(columns={'date': 'ds'})

# Predict on test set
forecast_test = model1.predict(future_test)

# Evaluate
y_true = test['milk_price'].values
y_pred = forecast_test['yhat'].values

mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

print("MAE:", mae)
print("RMSE:", rmse)
print("MAPE (%):", mape)

# %%
plt.figure(figsize=(14,6))
plt.plot(test['date'], y_true, label='Actual', color='blue')
plt.plot(test['date'], y_pred, label='Predicted', color='red')
plt.title("Prophet Model - Actual vs Predicted Milk Price")
plt.xlabel("Date")
plt.ylabel("Milk Price")
plt.legend()
plt.show()


# %%
fig1 = plot_plotly(model1, forecast_test)
fig1.show()

# %%
# Refit Prophet on the full dataset (till end of 2025)
model_final = Prophet(yearly_seasonality=True, daily_seasonality=False)
for var in exog_vars:
    model_final.add_regressor(var)

model_final.fit(data.rename(columns={'date': 'ds', 'milk_price': 'y'}))

# Make future dataframe for 365 days beyond last date
future = model_final.make_future_dataframe(periods=365, freq='D')

# Add exogenous variables to future
for var in exog_vars:
    # Simple approach: extend last known value forward
    future[var] = list(data[var].values) + [data[var].values[-1]]*365

# Forecast for 2026
forecast_future = model_final.predict(future)

# Extract 2026 forecast only
forecast_2026 = forecast_future[forecast_future['ds'].dt.year == 2026]

print(forecast_2026[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head())


# %%
# Plot forecast for 2026
plt.figure(figsize=(14,6))
plt.plot(data['date'], data['milk_price'], label='Historical (Actual)', color='blue')

# Plot 2026 forecast with confidence interval
plt.plot(forecast_2026['ds'], forecast_2026['yhat'], label='Forecast (2026)', color='red')
plt.fill_between(
    forecast_2026['ds'],
    forecast_2026['yhat_lower'],
    forecast_2026['yhat_upper'],
    color='pink', alpha=0.3, label='Confidence Interval'
)

plt.title("Prophet Forecast - Milk Prices for 2026")
plt.xlabel("Date")
plt.ylabel("Milk Price")
plt.legend()
plt.show()

# %%
forecast_2026 = forecast_future[forecast_future['ds'].dt.year == 2026].copy()
# Monthly average forecast
monthly_forecast = forecast_2026.set_index('ds')['yhat'].resample('M').mean()
print(monthly_forecast)

# %%
import statsmodels.api as sm

# Get residuals from test set
residuals = y_true - y_pred   # from your Prophet test prediction

# 1. Residual distribution
plt.figure(figsize=(12,5))
sns.histplot(residuals, kde=True, bins=30, color="purple")
plt.title("Residual Distribution (Prophet Model)")
plt.xlabel("Residual (Actual - Predicted)")
plt.show()

# 2. Residuals over time
plt.figure(figsize=(14,6))
plt.plot(test['date'], residuals, marker='o', linestyle='-', color="red")
plt.axhline(0, linestyle="--", color="black")
plt.title("Residuals over Time")
plt.xlabel("Date")
plt.ylabel("Residual")
plt.show()

# 3. Autocorrelation of residuals
sm.graphics.tsa.plot_acf(residuals, lags=40)
plt.title("ACF of Residuals")
plt.show()

# 4. Residual summary stats
print("Mean Residual:", np.mean(residuals))
print("Std of Residuals:", np.std(residuals))


# %% [markdown]
# 1.Residual Distribution (Histogram)
# 
# The residuals are centered close to 0 (mean residual ≈ 0.06).
# 
# Shape looks roughly bell-curved, though slightly skewed left.
# 
# No huge outliers → indicates Prophet captured the trend & seasonality quite well.
# 
# Interpretation: Predictions are unbiased overall, only small random noise left.
# 
# 2.Residuals over Time
# 
# The residuals fluctuate randomly around 0, with no strong patterns.
# 
# Variance looks stable (no signs of heteroskedasticity).
# 
# Occasional spikes, but no systematic upward/downward drift.
# 
# Interpretation: Prophet is not missing any obvious trend or recurring seasonality.
# 
# 3.ACF (Autocorrelation Function)
# 
# At lag > 0, nearly all autocorrelation bars fall inside the blue confidence interval.
# 
# This means there’s no strong autocorrelation left in residuals.
# 
# Interpretation: Prophet has captured most of the time-dependence. Residuals look like white noise.

# %%
#Scenario analysis for best and worst cases
future_2026 = future[future['ds'].dt.year == 2026].copy()

# Create scenario variations
scenarios = {
    "Baseline": future_2026.copy(),
    "Best-Case": future_2026.copy(),
    "Worst-Case": future_2026.copy()
}

# Best-case adjustments
scenarios["Best-Case"]["feed_cost"] *= 0.90   # -10%
scenarios["Best-Case"]["rainfall_mm"] *= 1.10 # +10%
scenarios["Best-Case"]["inflation_rate"] *= 0.95  # -5%
scenarios["Best-Case"]["energy_cost"] *= 0.92
scenarios["Best-Case"]["global_price"] *= 1.05
scenarios["Best-Case"]["transport_cost"] *= 0.93
scenarios["Best-Case"]["labor_rate"] *= 1.03
scenarios["Best-Case"]["local_production"] *= 1.08
scenarios["Best-Case"]["export_volume"] *= 1.06

# Worst-case adjustments
scenarios["Worst-Case"]["feed_cost"] *= 1.15   # +15%
scenarios["Worst-Case"]["rainfall_mm"] *= 0.85 # -15%
scenarios["Worst-Case"]["inflation_rate"] *= 1.10  # +10%
scenarios["Worst-Case"]["energy_cost"] *= 1.12
scenarios["Worst-Case"]["global_price"] *= 0.95
scenarios["Worst-Case"]["transport_cost"] *= 1.10
scenarios["Worst-Case"]["labor_rate"] *= 1.08
scenarios["Worst-Case"]["local_production"] *= 0.90
scenarios["Worst-Case"]["export_volume"] *= 0.93

# Generate forecasts for each scenario
scenario_forecasts = {}
for name, scenario_df in scenarios.items():
    forecast = model1.predict(scenario_df)
    scenario_forecasts[name] = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]
    scenario_forecasts[name]["Scenario"] = name

# Combine all forecasts into one DataFrame
all_forecasts = pd.concat(scenario_forecasts.values())

# %%
plt.figure(figsize=(12,6))
for name, forecast in scenario_forecasts.items():
    plt.plot(forecast['ds'], forecast['yhat'], label=name)

plt.fill_between(
    scenario_forecasts["Baseline"]['ds'],
    scenario_forecasts["Baseline"]['yhat_lower'],
    scenario_forecasts["Baseline"]['yhat_upper'],
    color="gray", alpha=0.2, label="Baseline CI"
)

plt.title("Milk Price Forecast Scenarios for 2026")
plt.xlabel("Date")
plt.ylabel("Milk Price")
plt.legend()
plt.show()

# %%
# Convert forecasts into monthly averages
monthly_summary = (
    all_forecasts
    .groupby([all_forecasts['ds'].dt.to_period('M'), "Scenario"])["yhat"]
    .mean()
    .reset_index()
)

# Pivot for easy comparison
monthly_summary = monthly_summary.pivot(index="ds", columns="Scenario", values="yhat")
monthly_summary.index = monthly_summary.index.astype(str)  # format YYYY-MM
print(monthly_summary.head(12))  # first 12 months of 2026

# %%
# Baseline forecast for 2026
baseline_forecast = forecast_2026.copy()
baseline_mean = baseline_forecast['yhat'].mean()

sensitivity_results = {}

# Variables to test
exog_vars = ['inflation_rate', 'feed_cost', 'energy_cost',
             'global_price', 'transport_cost', 'labor_rate',
             'local_production', 'rainfall_mm', 'export_volume']

for var in exog_vars:
    for shock, label in [(1.1, "+10%"), (0.9, "-10%")]:
        future_test = future_2026.copy()
        future_test[var] = future_test[var] * shock

        forecast_shock = model1.predict(future_test)
        shock_mean = forecast_shock['yhat'].mean()

        sensitivity_results[(var, label)] = shock_mean - baseline_mean

# Convert results to DataFrame
sens_df = pd.DataFrame(sensitivity_results.items(),
                       columns=['Scenario', 'Impact'])
sens_df[['Variable','Shock']] = pd.DataFrame(sens_df['Scenario'].tolist(),
                                             index=sens_df.index)
sens_df.drop(columns=['Scenario'], inplace=True)

# Plot tornado chart
plt.figure(figsize=(10,6))
for var in exog_vars:
    subset = sens_df[sens_df['Variable']==var]
    plt.barh(var, subset.loc[subset['Shock']=="+10%", 'Impact'], color='lightgreen')
    plt.barh(var, subset.loc[subset['Shock']=="-10%", 'Impact'], color='red')

plt.axvline(0, color='black', linestyle='--')
plt.title("Sensitivity of Milk Price Forecast (±10% Change in Regressors)")
plt.xlabel("Impact on Average Forecasted Price (2026)")
plt.ylabel("Regressor")
plt.show()


# %% [markdown]
# Top drivers of risk/opportunity (2026 forecast):
# 
# [Feed Cost, Inflation Rate, Global Price]
# 
# Moderate influence:
# 
# [Local Production, Energy Cost]
# 
# Low influence:
# 
# [Rainfall, Export Volume, Transport Cost, Labor Rate]

# %%
n_sims = 300               # number of Monte Carlo runs (increase if you want finer estimates)
forecast_days = 365         # days to forecast (2026)
start_date = data['date'].max()  # last observed date
regressors = exog_vars        # list of regressors used in Prophet
model_prophet = model1         # your fitted Prophet model
rng = np.random.default_rng(42)

# Import trange for progress bar
from tqdm.auto import trange

# Create base future df (dates)
future_dates = pd.date_range(start=start_date + pd.Timedelta(days=1),
                             periods=forecast_days, freq='D')
future_base = pd.DataFrame({'ds': future_dates})

#Derive historical stats for regressors (mean, std, covariance)
# Use recent window (e.g., last 365 days) or entire history
hist_window = data[data['date'] > (data['date'].max() - pd.Timedelta(days=365))].copy()
mu = hist_window[regressors].mean().values             # vector of means
sigma = hist_window[regressors].std(ddof=1).values     # std deviations
cov = hist_window[regressors].cov().values            # covariance matrix (for correlated sampling)

# Regularize cov if needed (small diagonal add)
cov += np.eye(len(regressors)) * 1e-6

#Compute residuals from training (to bootstrap noise)
#Get training residuals: actual - prophet_pred on validation/train
#If you have test residuals from earlier, use training residuals for bootstrap
#Example: use residuals array from earlier (residuals_train or residuals). If not:
#Fit/compute preds on in-sample train (or use test residuals)
#Here, assume you have 'train' and predicted 'prophet_train_pred' -> residuals_train.
residuals_train = (train['milk_price'].values -
                   model_prophet.predict(train.rename(columns={'date':'ds','milk_price':'y'}))[ 'yhat'].values)
residuals_train = residuals_train[~np.isnan(residuals_train)]  # clean

#4) Monte Carlo loop: sample regressors and predict
pred_matrix = np.zeros((forecast_days, n_sims))  # store yhat (without residuals), we can later add residual noise

# Option A: Sample regressors independently (simple)
# Option B: Sample regressors jointly using multivariate normal (captures correlations) — used here
for s in trange(n_sims):
    # sample a multivariate draw for the whole forecast horizon:
    # simple approach: assume daily i.i.d. draws from MVN(mu, cov)
    sampled = rng.multivariate_normal(mean=mu, cov=cov, size=forecast_days)
    # build future df for Prophet (must contain all regressors with correct column names)
    fut = future_base.copy()
    for i, r in enumerate(regressors):
        fut[r] = sampled[:, i]
    # optionally you can apply a smoothing or trend to sampled regressors instead of i.i.d.
    # predict with Prophet (no residual added yet)
    pred = model_prophet.predict(fut)['yhat'].values
    pred_matrix[:, s] = pred

#5) Add residual bootstrap (optional)
# For each simulation, add a random residual sampled (with replacement) for each day,
# or add day-specific noise sampled from residuals distribution.
# We'll sample residuals_i.i.d. from residuals_train
resid_samples = rng.choice(residuals_train, size=(forecast_days, n_sims), replace=True)
pred_matrix_with_noise = pred_matrix + resid_samples

#6) Aggregate results: percentiles and probabilities
percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
quantiles = np.percentile(pred_matrix_with_noise, q=percentiles, axis=1)  # shape: (len(percentiles), days)

# Build result DataFrame
res_df = pd.DataFrame({
    'ds': future_dates
})
for i, p in enumerate(percentiles):
    res_df[f'p{p}'] = quantiles[i, :]

# compute mean & std
res_df['mean'] = pred_matrix_with_noise.mean(axis=1)
res_df['std'] = pred_matrix_with_noise.std(axis=1)

# Example: probability that price > 70 each day
threshold = 70
prob_gt_70 = (pred_matrix_with_noise > threshold).mean(axis=1)
res_df['prob_gt_70'] = prob_gt_70

# ---- 7) Monthly aggregation of distributions (means / percentiles) ----
res_df.set_index('ds', inplace=True)
monthly = res_df.resample('M').agg({
    'mean': 'mean',
    'std': 'mean',
    'p5': 'mean',
    'p50': 'mean',
    'p95': 'mean',
    'prob_gt_70': 'mean'
})
monthly.index = monthly.index.to_period('M').astype(str)
print(monthly.head())

# ---- 8) Plot fan chart (daily) ----
plt.figure(figsize=(14,6))
plt.plot(data['date'], data['milk_price'], color='blue', alpha=0.6, label='Historical')

# plot median and some percentiles
plt.plot(res_df.index, res_df['p50'], color='red', label='Median forecast')
plt.fill_between(res_df.index, res_df['p10'], res_df['p90'], color='pink', alpha=0.4, label='10-90%')
plt.fill_between(res_df.index, res_df['p25'], res_df['p75'], color='salmon', alpha=0.6, label='25-75%')
plt.plot(res_df.index, res_df['p5'], color='none')  # just to ensure p5 exists
plt.title('Monte Carlo + Residual-bootstrap Fan Chart (2026)')
plt.legend()
plt.show()

# ---- 9) Example: probability over a month
#print("Probability average price in Jan-2026 > 70:",
#     (pred_matrix_with_noise[res_df.index.month==1].mean(axis=0).mean() > 70).mean())  # rough example
yearly_avg_by_sim = pred_matrix_with_noise.mean(axis=0)  # shape: (n_sims,)

# Probability
threshold = 70
prob_year_gt70 = (yearly_avg_by_sim > threshold).mean()

print(f"Probability that 2026 average milk price > {threshold}: {prob_year_gt70:.3f}")
# ---- 10) Save results
res_df.reset_index().to_csv('montecarlo_forecast_2026_daily.csv', index=False)
monthly.to_csv('montecarlo_forecast_2026_monthly_summary.csv')

# %%
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
# Features (exogenous variables + lag features)
exog_vars = ['inflation_rate', 'feed_cost', 'energy_cost',
             'global_price', 'transport_cost', 'labor_rate',
             'local_production', 'rainfall_mm', 'export_volume']

# Add lag features (milk price shifted by previous days)
data['lag1'] = data['milk_price'].shift(1)
data['lag7'] = data['milk_price'].shift(7)
data['lag30'] = data['milk_price'].shift(30)

# Drop NaN rows caused by lagging
data_ml = data.dropna().copy()

X = data_ml[exog_vars + ['lag1', 'lag7', 'lag30']]
y = data_ml['milk_price']

# Train-test split (train before 2024, test from 2024 onwards)
X_train = X[data_ml['date'] < '2024-01-01']
y_train = y[data_ml['date'] < '2024-01-01']
X_test = X[data_ml['date'] >= '2024-01-01']
y_test = y[data_ml['date'] >= '2024-01-01']

# Train Random Forest
rf_model = RandomForestRegressor(n_estimators=200, random_state=42)
rf_model.fit(X_train, y_train)

# Predictions
y_pred_rf = rf_model.predict(X_test)

# %%
# Evaluation
mae = mean_absolute_error(y_test, y_pred_rf)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_rf))
mape = np.mean(np.abs((y_test - y_pred_rf) / y_test)) * 100

print("Random Forest Results")
print("MAE:", mae)
print("RMSE:", rmse)
print("MAPE (%):", mape)

# %%
import matplotlib.pyplot as plt

# Plot actual vs predicted
plt.figure(figsize=(14,6))
plt.plot(data_ml['date'][data_ml['date'] >= '2024-01-01'], y_test, label='Actual', color='blue')
plt.plot(data_ml['date'][data_ml['date'] >= '2024-01-01'], y_pred_rf, label='Random Forest Prediction', color='red')

plt.title("Random Forest - Milk Price Prediction (Test Set)")
plt.xlabel("Date")
plt.ylabel("Milk Price")
plt.legend()
plt.show()


# %% [markdown]
# # **XG boost**

# %%
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor, plot_importance

# Features
exog_vars = ['inflation_rate', 'feed_cost', 'energy_cost',
             'global_price', 'transport_cost', 'labor_rate',
             'local_production', 'rainfall_mm', 'export_volume']

# Add lag features
data['lag1'] = data['milk_price'].shift(1)
data['lag7'] = data['milk_price'].shift(7)
data['lag30'] = data['milk_price'].shift(30)

data_ml = data.dropna().copy()

X = data_ml[exog_vars + ['lag1', 'lag7', 'lag30']]
y = data_ml['milk_price']

# Train-test split
X_train = X[data_ml['date'] < '2024-01-01']
y_train = y[data_ml['date'] < '2024-01-01']
X_test = X[data_ml['date'] >= '2024-01-01']
y_test = y[data_ml['date'] >= '2024-01-01']

# Train XGBoost
xgb_model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

xgb_model.fit(X_train, y_train)

# Predictions
y_pred_xgb = xgb_model.predict(X_test)


# %%
# Evaluation
mae = mean_absolute_error(y_test, y_pred_xgb)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
mape = np.mean(np.abs((y_test - y_pred_xgb) / y_test)) * 100

print("XGBoost Results")
print("MAE:", mae)
print("RMSE:", rmse)
print("MAPE (%):", mape)


# %%
plt.figure(figsize=(14,6))
plt.plot(data_ml['date'][data_ml['date'] >= '2024-01-01'], y_test, label="Actual", color="blue")
plt.plot(data_ml['date'][data_ml['date'] >= '2024-01-01'], y_pred_xgb, label="XGBoost Prediction", color="red")
plt.title("XGBoost - Milk Price Prediction (Test Set)")
plt.xlabel("Date")
plt.ylabel("Milk Price")
plt.legend()
plt.show()

# %%
plt.figure(figsize=(10,6))
plot_importance(xgb_model, importance_type="gain", max_num_features=15, height=0.5)
plt.title("XGBoost Feature Importance (Top Drivers of Milk Price)")
plt.show()


# %% [markdown]
# #**Hybrid Model - Prophet + XG boost**

# %%
#Prophet Model
prophet_data = data[['date', 'milk_price']].rename(columns={'date': 'ds', 'milk_price': 'y'})
exog_vars = ['inflation_rate', 'feed_cost', 'energy_cost',
             'global_price', 'transport_cost', 'labor_rate',
             'local_production', 'rainfall_mm', 'export_volume']

# Train-test split
train = data[data['date'] < '2024-01-01'].copy()
test = data[data['date'] >= '2024-01-01'].copy()

# Fit Prophet
model_p = Prophet(yearly_seasonality=True, daily_seasonality=False)
for var in exog_vars:
    model_p.add_regressor(var)
model_p.fit(train.rename(columns={'date': 'ds', 'milk_price': 'y'}))

# Prophet prediction on test set
future_test = test[['date'] + exog_vars].rename(columns={'date': 'ds'})
forecast_p = model_p.predict(future_test)
prophet_pred = forecast_p['yhat'].values

# Step 2: Residuals for the training data
# Add lag features to the original data first
data['lag1'] = data['milk_price'].shift(1)
data['lag7'] = data['milk_price'].shift(7)
data['lag30'] = data['milk_price'].shift(30)

# Drop NaN rows caused by lagging
data_ml = data.dropna().copy()

# Align the train/test split with the lagged data
train_ml = data_ml[data_ml['date'] < '2024-01-01'].copy()
test_ml = data_ml[data_ml['date'] >= '2024-01-01'].copy()

# Get the dates and regressors from the lagged training data for Prophet prediction
train_dates_ml = train_ml[['date'] + exog_vars].rename(columns={'date': 'ds'})

# Predict with Prophet on the lagged training dates
forecast_train_p = model_p.predict(train_dates_ml)

# Align prophet predictions with train_ml data based on date
prophet_train_pred_aligned = forecast_train_p.set_index('ds').reindex(train_ml['date']).reset_index()['yhat'].values


# Calculate residuals for the lagged training data
residuals_train = train_ml['milk_price'].values - prophet_train_pred_aligned


# Step 3: XGBoost on Residuals
X_train_ml = train_ml[exog_vars + ['lag1', 'lag7', 'lag30']]
X_test_ml = test_ml[exog_vars + ['lag1', 'lag7', 'lag30']]


xgb_model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

xgb_model.fit(X_train_ml, residuals_train)

# Predict residuals on test set
xgb_residuals_pred = xgb_model.predict(X_test_ml)


# Step 4: Hybrid Forecast
final_pred = prophet_pred + xgb_residuals_pred

# %%
# Evaluation
y_true = test_ml['milk_price'].values # Use test_ml for y_true
mae = mean_absolute_error(y_true, final_pred)
rmse = np.sqrt(mean_squared_error(y_true, final_pred))
mape = np.mean(np.abs((y_true - final_pred) / y_true)) * 100

print("Hybrid Prophet + XGBoost Results")
print("MAE:", mae)
print("RMSE:", rmse)
print("MAPE (%):", mape)

# %%
# Plot Results
plt.figure(figsize=(14,6))
plt.plot(test_ml['date'], y_true, label="Actual", color="blue")
plt.plot(test['date'], prophet_pred, label="Prophet Only", color="green", linestyle="--")
plt.plot(test_ml['date'], final_pred, label="Hybrid Prophet+XGB", color="red")
plt.title("Hybrid Prophet + XGBoost Milk Price Prediction")
plt.xlabel("Date")
plt.ylabel("Milk Price")
plt.legend()
plt.show()

# %% [markdown]
# Linear Regression Model

# %%
from sklearn.linear_model import LinearRegression
exog_vars = ['inflation_rate', 'feed_cost', 'energy_cost','global_price', 'transport_cost', 'labor_rate','local_production', 'rainfall_mm', 'export_volume']
X = data[exog_vars]
y = data['milk_price']

# Train-test split (same as before: train until 2024, test from 2024 onward)
X_train = X[data['date'] < '2024-01-01']
X_test = X[data['date'] >= '2024-01-01']
y_train = y[data['date'] < '2024-01-01']
y_test = y[data['date'] >= '2024-01-01']

lr_model = LinearRegression()
lr_model.fit(X_train,y_train)

y_pred_lr = lr_model.predict(X_test)

# %%
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

print("Linear Regression Results:")
print("MAE:", mae)
print("RMSE:", rmse)
print("MAPE (%):", mape)

# %%
plt.figure(figsize=(12,6))
plt.plot(data['date'][data['date'] >= '2024-01-01'], y_test, label='Actual', color='blue')
plt.plot(data['date'][data['date'] >= '2024-01-01'], y_pred, label='Predicted (LR)', color='red')
plt.title("Milk Price Prediction - Linear Regression")
plt.xlabel("Date")
plt.ylabel("Milk Price")
plt.legend()
plt.show()

# %%
results = {
    "Prophet": {"MAE": 1.03, "RMSE": 1.29, "MAPE": 1.50},
    "Random Forest": {"MAE": 1.13, "RMSE": 1.40, "MAPE": 1.65},
    "XGBoost": {"MAE": 1.06, "RMSE": 1.33, "MAPE": 1.55},
    "Hybrid": {"MAE": 1.05, "RMSE": 1.31, "MAPE": 1.52},
    "Linear Regression": {"MAE": 1.03, "RMSE": 1.29, "MAPE": 1.50}  # from your LR run
}
metrics = ["MAE", "RMSE", "MAPE"]
x = np.arange(len(results))  # number of models
width = 0.25  # bar width

fig, ax = plt.subplots(figsize=(12,6))

bars = []
for i, metric in enumerate(metrics):
    values = [results[model][metric] for model in results]
    b = ax.bar(x + i*width, values, width, label=metric)
    bars.append((b, values))

# Add values on top of bars
for b, values in bars:
    for rect, val in zip(b, values):
        ax.text(
            rect.get_x() + rect.get_width()/2,
            rect.get_height() + 0.02,
            f"{val:.2f}",
            ha="center", va="bottom", fontsize=9
        )

# Formatting
ax.set_xticks(x + width)
ax.set_xticklabels(results.keys(), rotation=20)
ax.set_ylabel("Error Value")
ax.set_title("Model Performance Comparison")
ax.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# Prophet model is performing the best

# %% [markdown]
# Two Things you can Add: Note for Tushar
# 
# 1- Can add more Supervised Prediction Algorithms and them compare each other value in Bar graph.
# 
# 2- Can Use Supervised Classification Algoriths and then also can multiple of them then compare each other Value.
# 
# 3- Optional: Can Automate the process and sent the Milk Price daily/Weekly to Someone through Mail and Other in Advance. (Tell me if you want to automate this Project Like this.)



# %%
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

