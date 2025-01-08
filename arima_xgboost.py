import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
import itertools
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import TimeSeriesSplit
from xgboost_model import XGBRegressor
from sklearn.model_selection import GridSearchCV

#keep track of time
start_time = time.time()

def arima_full_process(df, column_name, test_sizes=[1, 3, 5, 7, 15], p_range=(0, 3), d_range=(0, 1), q_range=(0, 3)):
    """
    Perform preprocessing, hyperparameter tuning, model fitting, and evaluate performance on different test sizes.
    Then predict the entire time series and return a dataframe with the original and predicted values.

    Parameters:
        df (pd.DataFrame): DataFrame with a datetime index and a column for streamflow data.
        column_name (str): Name of the column containing the streamflow data.
        test_sizes (list): List of test sizes in days for performance evaluation.
        p_range, d_range, q_range (tuple): Ranges for ARIMA hyperparameters.

    Returns:
        pd.DataFrame: DataFrame containing the date, streamflow, and predicted ARIMA streamflow.
    """
    # Step 1: Read streamflow data
    data = df[column_name]

    # Step 2: Hyperparameter tuning with cross-validation
    print("Performing ARIMA hyperparameter tuning with cross-validation...")
    best_score = float('inf')
    best_params = None
    parameter_grid = list(itertools.product(range(p_range[0], p_range[1] + 1),
                                            range(d_range[0], d_range[1] + 1),
                                            range(q_range[0], q_range[1] + 1)))

    # Cross-validation setup
    n_splits = 3
    tscv = TimeSeriesSplit(n_splits=n_splits)

    for p, d, q in parameter_grid:
        try:
            cv_scores = []
            for train_index, test_index in tscv.split(data):
                train, test = data[train_index], data[test_index]
                model = ARIMA(train, order=(p, d, q))
                results = model.fit()
                predictions = results.forecast(steps=len(test))
                mse = mean_squared_error(test, predictions)
                cv_scores.append(mse)
            
            # Calculate average cross-validation score
            avg_cv_score = np.mean(cv_scores)
            if avg_cv_score < best_score:
                best_score = avg_cv_score
                best_params = (p, d, q)
        except Exception as e:
            continue

    print(f"Best Parameters: ARIMA{best_params} with CV MSE = {best_score:.2f} & RMSE = {np.sqrt(best_score):.2f}")

    # Step 3: Fit the best model on the full data
    print(f"Fitting ARIMA with best parameter {best_params} on the full dataset...")
    final_model = ARIMA(data, order=best_params)
    final_results = final_model.fit()

    # Need to check residuals, if residuals are not white noise, then ARIMA is not a good fit
    residuals = final_results.resid # Get residuals of the ARIMA model, which is the difference between the observed and predicted values
    plt.figure(figsize=(6, 4))
    plt.plot(residuals)
    plt.title("Residuals of ARIMA Model")
    plt.xlabel("Time")
    plt.ylabel("Residuals")
    plt.tight_layout()
    plt.show()

    # Step 4: Foreast for different forecast horizons, test_sizes
    predictions = []
    for test_size in test_sizes:
        forecast = final_results.forecast(steps=test_size) # Forecast for the test_size days,
        predictions.extend(forecast) # Append the forecast to the predictions list
    print("Forecast made for different forecast horizons.")
    # Step 5: return arima residuals and forecast
    return residuals, predictions


################################################################################
#fit ARIMA model on streamflow data
# Read the original data
df = pd.read_csv('og_file.csv')
#extract last 30 days of data for testing
test_df = df.tail(30)
#drop last 30 days of data from original dataframe
df = df.head(len(df)-30)

#make date column using year, month, day
df['date'] = pd.to_datetime(df[['year', 'month', 'day']])
df = df.set_index('date')
#handle missing values
df = df.interpolate(method='time').fillna(method='bfill').fillna(method='ffill') #ffill and bfill fills na with previous and next values based on time index of the data
df_arima = df[['streamflow']].asfreq('D')
#call the arima function
arima_residuals, arima_forecasts = arima_full_process(df_arima, column_name="streamflow", test_sizes=[30], p_range=(0, 2), d_range=(0, 1), q_range=(0, 2)) # Test sizes of 3, 7, 15, 30 days

# Calculate performance metrics for different forecast horizons
horizons = [1, 2, 3, 4, 5, 6, 7, 15]
metrics = {'Forecast Horizon (Days)': [], 'MSE': [], 'RMSE': [], 'PBIAS (%)': []}

for h in horizons:
    # mse = mean_squared_error(df_arima['streamflow'].tail(h), output_arima[:h])
    mse = mean_squared_error(test_df['streamflow'][:h], arima_forecasts[:h])
    rmse = np.sqrt(mse)
    pbias = np.mean((arima_forecasts[:h] - test_df['streamflow'][:h]) / test_df['streamflow'][:h]) * 100
    metrics['Forecast Horizon (Days)'].append(h)
    metrics['MSE'].append(mse)
    metrics['RMSE'].append(rmse)
    metrics['PBIAS (%)'].append(pbias)
summary_table = pd.DataFrame(metrics)

print("\nSummary Table of Performance Metrics for ARIMA Forecast Horizons:")
print(summary_table)

print('Time taken to complete arima:', time.time()-start_time)


################################################################################
#fit XGBoost model to predict residuals of ARIMA model
#add residuals column to the original dataframe
df['residuals'] = arima_residuals
#drop streamflow column and reset index
df = df.drop(columns=['streamflow']).reset_index(drop=True)

#split data, do hyperparameter tuning and fit model
# Split the data into training and testing sets
X_train = df.drop(columns=['residuals'])
y_train = df['residuals']

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

print("Hyperparameter tuning: Performing Grid Search to find the best parameters...")
# Perform Grid Search to find the best parameters
xgb_grid = GridSearchCV(xgb, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
xgb_grid.fit(X_train, y_train)

# Get the best model
best_xgb = xgb_grid.best_estimator_

# Display the best hyperparameters and cross-validation mean squared error and root mean squared error
print(f"Best XGBoost Parameters: {xgb_grid.best_params_} with CV MSE = {xgb_grid.best_score_:.2f} & RMSE = {np.sqrt(-xgb_grid.best_score_):.2f}")

#fit the best model on the entire dataset
print('Fitting the best XGBoost model on the entire dataset...')
best_xgb.fit(X_train, y_train)

#make feature importance plot
# importances = best_xgb.feature_importances_
# plt.figure(figsize=(6, 4))
# plt.bar(X.columns, importances)
# plt.title("Feature Importance/Variance Explained Plot")
# plt.xlabel("Features")
# plt.ylabel("Importance")
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()

#predict residuals for the last 30 days
X_test = test_df.drop(columns=['streamflow']).reset_index(drop=True)
residuals_pred = best_xgb.predict(X_test)

################################################################################
#add XGBoost residuals to ARIMA predictions to get final predictions of streamflow
final_predictions = arima_forecasts + residuals_pred

# Calculate performance metrics on final predictions for different forecast horizons
horizons = [1, 2, 3, 4, 5, 6, 7, 15]
metrics = {'Forecast Horizon (Days)': [], 'MSE': [], 'RMSE': [], 'PBIAS (%)': []}
for h in horizons:
    mse = mean_squared_error(test_df['streamflow'][:h], final_predictions[:h])
    rmse = np.sqrt(mse)
    pbias = np.mean((final_predictions[:h] - test_df['streamflow'][:h]) / test_df['streamflow'][:h]) * 100
    metrics['Forecast Horizon (Days)'].append(h)
    metrics['MSE'].append(mse)
    metrics['RMSE'].append(rmse)
    metrics['PBIAS (%)'].append(pbias)

summary_table = pd.DataFrame(metrics)
print("\nSummary Table of Performance Metrics for Final Forecast Horizons:")
print(summary_table)

print('Time taken to complete arima+xgboost:', time.time()-start_time)
