import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox
# Load and preprocess the dataset
file_path = "og_file.csv"  # Replace with your file path
df = pd.read_csv(file_path)

#add lagged features
# Add lagged features
for lag in range(1, 4): #add lagged features for 1 to 3 days
    for col in ['temp', 'precip', 'streamflow']:
        df[f'{col}_lag{lag}'] = df[col].shift(lag)
df = df.dropna()

#use 1994 to 2009 for training and 2009 to 2015 for testing
train = df[df['year'] < 2009]
test = df[df['year'] >= 2009]

#test set for model evaluation
test_pred = test.copy()

#test set for model forecasting
# Replace streamflow lag with NaN except for the first row in the test set
test = test.reset_index(drop=True)
test.loc[:, ['streamflow_lag1', 'streamflow_lag2', 'streamflow_lag3']] = np.nan

#prepare xtrain and ytrain for training
X_train = train.drop(columns=['streamflow'])
y_train = train['streamflow']

# Define the parameter grid to search
param_grid = {
    'n_estimators': [100, 200, 500],
    'max_depth': [2, 5, 10], 
    'learning_rate': [0.01, 0.1, 0.3], 
    # 'subsample': [0.5, 0.7, 1],
    # 'colsample_bytree': [0.5, 0.7, 1],
}

# Initialize the XGBRegressor
xgb = XGBRegressor(random_state=0)

print("XGBoost Hyperparameters tuning...")
# Perform Grid Search to find the best parameters
xgb_grid = GridSearchCV(xgb, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
xgb_grid.fit(X_train, y_train)

# Get the best model
best_xgb = xgb_grid.best_estimator_

# Display the best hyperparameters and cross-validation mean squared error, root mean squared error, and r2 score
print(f"Best XGBoost Parameters:\n{xgb_grid.best_params_}")
# print(f"CV MSE = {-xgb_grid.best_score_:.3f}")
print(f"CV RMSE = {np.sqrt(-xgb_grid.best_score_):.2f}")

#fit the best model on the entire training dataset
print('Fitting best XGBoost on entire training dataset...')
best_xgb.fit(X_train, y_train)

#calculate residuals for the training set
residuals_train = y_train - best_xgb.predict(X_train) 
#make dataframe of residuals with date
residual_train_date = pd.to_datetime(train[['year', 'month', 'day']])
residuals_train = pd.DataFrame({'date': residual_train_date, 'residuals': residuals_train})
residuals_train = residuals_train.set_index('date')

#evaluate the model on the test set, rmse and r2 score
X_test = test_pred.drop(columns=['streamflow'])
y_test = test_pred['streamflow']
y_pred = best_xgb.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print(f"Test RMSE: {rmse:.2f} & R2 Score: {r2:.2f}")

#calculate residuals for the test set
residuals_test = y_test - y_pred
#make dataframe of residuals with date
residual_test_date = pd.to_datetime(test_pred[['year', 'month', 'day']])
residuals_test = pd.DataFrame({'date': residual_test_date, 'residuals': residuals_test})
residuals_test = residuals_test.set_index('date')

#feature importance plot
importances = best_xgb.feature_importances_
#sort the importances
indices = np.argsort(importances)[::-1]
#plot the importances
plt.figure(figsize=(6, 4))
plt.bar(X_train.columns[indices], importances[indices])
plt.title("Feature Importance/Variance Explained Plot")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# Make 15-day forecast
forecast_df = pd.DataFrame(columns=['year', 'month', 'day', 'Observed', 'Forecast'])
for test_year in range(2009, 2016):
    test_df = test[test['year'] == test_year]
    for month in range(1, 12):
        test_monthly = test_df[test_df['month'] == month]
        for day in range(1, 29):
            test_day = test_monthly[test_monthly['day'] == day]
            if day == 1:
                streamflow_lag1 = test_pred.loc[(test_pred['year'] == test_year) & (test_pred['month'] == month) & (test_pred['day'] == day), 'streamflow_lag1'].values[0]
                streamflow_lag2 = test_pred.loc[(test_pred['year'] == test_year) & (test_pred['month'] == month) & (test_pred['day'] == day), 'streamflow_lag2'].values[0]
                streamflow_lag3 = test_pred.loc[(test_pred['year'] == test_year) & (test_pred['month'] == month) & (test_pred['day'] == day), 'streamflow_lag3'].values[0]
            test_day = test_day.assign(streamflow_lag1=streamflow_lag1, streamflow_lag2=streamflow_lag2, streamflow_lag3=streamflow_lag3)
            X_test = test_day.drop(columns=['streamflow'])
            forecast = best_xgb.predict(X_test)
            forecast_df = pd.concat([forecast_df, pd.DataFrame({'year': [test_year], 'month': [month], 'day': [day], 'Observed': [test_day['streamflow'].values[0]], 'Forecast': [forecast[0]]})], ignore_index=True)
            streamflow_lag1, streamflow_lag2, streamflow_lag3 = forecast[0], streamflow_lag1, streamflow_lag2
forecast_df['Date'] = pd.to_datetime(forecast_df[['year', 'month', 'day']])

# visualize the actual vs forecasted values from the forecast_df
#find average of observed and forecasted values for each day
avg_day = forecast_df.groupby('day').mean()
plt.figure(figsize=(6, 4))
plt.plot(avg_day['Observed'], label='Observed')
plt.plot(avg_day['Forecast'], label='Forecast')
plt.title("Average Observed vs Forecasted Streamflow")
plt.xlabel("Day")
plt.ylabel("Streamflow")
plt.ylim(0, None)
plt.legend()
plt.show()

#find rmse for each day forecast and observed values
rmse_df = pd.DataFrame(columns=['Day', 'RMSE'])
for day in list(range(1,29)):
    day_df = forecast_df[forecast_df['day'] == day]
    mse = mean_squared_error(day_df['Observed'], day_df['Forecast'])
    rmse = mse ** 0.5
    rmse_df = pd.concat([rmse_df, pd.DataFrame({'Day': [day], 'RMSE': [rmse]})])
#plot rmse for each day
plt.figure(figsize=(6, 4))
plt.plot(rmse_df['Day'], rmse_df['RMSE'])
plt.title("RMSE for different forecasting horizons")
plt.xlabel("Day")
plt.ylabel("RMSE")
plt.ylim(0, None)
plt.show() 


############################################################################################################################################################################
#check for autocorrelation in residuals
# Ljung-Box test for residuals to check for autocorrelation in residuals
# Null hypothesis: Residuals are white noise (no autocorrelation)
# If p-value < 0.05, reject null hypothesis, residuals are not white noise
lb_test_results = acorr_ljungbox(residuals_train['residuals'], lags=90)
lb_test_stat = lb_test_results['lb_stat'].values
lb_p_value = lb_test_results['lb_pvalue'].values
#make boxplot of lb_p_value, also show points, and threshold line at 0.05
plt.figure(figsize=(6, 4))
plt.boxplot(lb_p_value, showfliers=False)
plt.axhline(y=0.05, color='r', linestyle='--', label="Threshold at 0.05")
plt.title("Ljung-Box Test P-Values for\nNull Hypothesis of No Autocorrelation")
plt.ylabel("P-Value")
plt.xlabel('1-50 day lag')
plt.legend()
plt.show()


# Fit auto_arima to find the best ARIMA parameters on the training set
y_train_arima = residuals_train['residuals']
y_test_arima = residuals_test['residuals']

#only train on last 3 years of data to speed up the process
# y_train_arima = y_train_arima[y_train_arima.index.year >= 2008]

print("Finding the best ARIMA parameters...")
arima_model = auto_arima(y_train_arima, # residuals from XGBoost model
                         exogeneous=None, # exogeneous variables
                         seasonal=True, # seasonal is true if data has seasonality
                         m=12, # seasonility of 12 months
                         approximation=True , # use approximation to speed up the search
                         stepwise=True, # stepwise is true to speed up the search
                         suppress_warnings=True, # suppress warnings
                         error_action="ignore", # ignore orders that don't converge
                         trace=True) # print results while training
print(f"Best ARIMA Order: {arima_model.order}")
print(f"Best Seasonal Order: {arima_model.seasonal_order}")
best_order = arima_model.order
best_seasonal_order = arima_model.seasonal_order

# Fit the best ARIMA model on entire the training set
arima_model = SARIMAX(y_train_arima, order=best_order, seasonal_order=best_seasonal_order, exog=None).fit()
# Make prediction on entire training set and calculate residuals
predict_train = arima_model.fittedvalues
residuals = y_train_arima.values - predict_train
residuals = residuals.values

# Analyze residuals, plot time series of residuals
plt.figure(figsize=(6, 4))
plt.plot(residuals)
plt.title("Residuals of ARIMA Model")
plt.xlabel("Time")
plt.ylabel("Residuals")
plt.show()

# Ljung-Box test for residuals to check for autocorrelation in residuals
# Null hypothesis: Residuals are white noise (no autocorrelation)
# If p-value < 0.05, reject null hypothesis, residuals are not white noise
lb_test_results = acorr_ljungbox(residuals, lags=50) 
lb_test_stat = lb_test_results['lb_stat'].values
lb_p_value = lb_test_results['lb_pvalue'].values
#make boxplot of lb_p_value, also show points, and threshold line at 0.05
plt.figure(figsize=(6, 4))
plt.boxplot(lb_p_value, showfliers=True)
plt.axhline(y=0.05, color='r', linestyle='--', label="Threshold at 0.05")
plt.title("Ljung-Box Test P-Values for\nNull Hypothesis of No Autocorrelation")
plt.ylabel("P-Value")
plt.xlabel('1-50 day lag')
plt.legend()
plt.show()

# Use the best fitted ARIMA model to forecast 28-day streamflow residuals
resid_forecast_df = pd.DataFrame(columns=['Date', 'Observed', 'Forecast'])
for test_year in range(2009, 2016):
    y_test_arima_year = y_test_arima[y_test_arima.index.year == test_year]
    for month in range(1, 12):  # Loop through all 12 months
        y_test_arima_monthly = y_test_arima_year[y_test_arima_year.index.month == month]
        try:
            # Fit SARIMAX model and forecast for 15 steps
            results = SARIMAX(y_test_arima_monthly, order=best_order, seasonal_order=best_seasonal_order, exog=None).fit(maxiter=500, disp=False)
            forecast_monthly = results.forecast(steps=28, exog=None)
            
            # Append forecast data to the DataFrame
            resid_forecast_df = pd.concat([resid_forecast_df, pd.DataFrame({
                'Date': forecast_monthly.index, 
                'Observed': y_test_arima_monthly.values[0:28], 
                'Forecast': forecast_monthly.values
            })], ignore_index=True)
        except np.linalg.LinAlgError:
            print(f"LinAlgError for year {test_year}, month {month}. Skipping this month.")
resid_forecast_df = resid_forecast_df.reset_index(drop=True)

#Final streamflow forecast is sum of XGBoost streamflow forecast and ARIMA residuals forecast
# Merge the XGBoost forecast and ARIMA residuals forecast for same dates
final_forecast_df = pd.merge(forecast_df, resid_forecast_df, on='Date', suffixes=('_xgb', '_arima'))
final_forecast_df['Final_Forecast'] = final_forecast_df['Forecast_xgb'] + final_forecast_df['Forecast_arima']

# visualize the actual vs forecasted values from the forecast_df
final_forecast_df['day'] = final_forecast_df['Date'].dt.day
#find average of observed and forecasted values for each day
avg_day = final_forecast_df.groupby('day').mean()
plt.figure(figsize=(6, 4))
plt.plot(avg_day['Observed_xgb'], label='Observed')
plt.plot(avg_day['Final_Forecast'], label='XGBoost+ARIMA Forecast')
plt.plot(avg_day['Forecast_xgb'], label='XGBoost Forecast')
plt.title("Observed vs Forecasted Streamflow")
plt.xlabel("Day")
plt.ylabel("Streamflow")
# plt.ylim(0, None)
plt.legend(loc='best')
plt.show()

#find rmse for each day forecast and observed values
final_rmse_df = pd.DataFrame(columns=['Day', 'RMSE'])
for day in list(range(1,28)):
    day_df = final_forecast_df[final_forecast_df['day'] == day]
    mse = mean_squared_error(day_df['Observed_xgb'], day_df['Final_Forecast'])
    rmse = mse ** 0.5
    final_rmse_df = pd.concat([final_rmse_df, pd.DataFrame({'Day': [day], 'RMSE': [rmse]})])
#plot rmse for each day
plt.figure(figsize=(6, 4))
plt.plot(final_rmse_df['Day'], final_rmse_df['RMSE'], label='XGBoost+ARIMA Forecast')
plt.plot(rmse_df['Day'], rmse_df['RMSE'], label='XGBoost Forecast')
plt.title("RMSE for different forecasting horizons")
plt.xlabel("Day")
plt.ylabel("RMSE")
# plt.ylim(0, None)
plt.legend(loc='best')
plt.show()


#This concludes that ARIMA model post-processing of XGBoost model residuals
#doesn't improve foreasting performance, this might be due to the fact that 
#XGBoost model already captures the trend and seasonality in the data.