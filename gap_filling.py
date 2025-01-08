import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
import itertools
import matplotlib.pyplot as plt
import time

#keep tract of time
start_time = time.time()

def arima_full_process(df, column_name, test_sizes=[1, 3, 7, 15, 30], p_range=(0, 3), d_range=(0, 1), q_range=(0, 3)):
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
    # Step 1: read streamflow data
    data = df[column_name]

    # Step 2: Hyperparameter tuning
    print("Performing hyperparameter tuning...")
    best_score = float('inf')
    best_params = None
    parameter_grid = list(itertools.product(range(p_range[0], p_range[1] + 1),
                                            range(d_range[0], d_range[1] + 1),
                                            range(q_range[0], q_range[1] + 1)))

    for p, d, q in parameter_grid:
        try:
            model = ARIMA(data, order=(p, d, q))
            results = model.fit()
            predictions = results.fittedvalues
            mse = mean_squared_error(data, predictions)
            if mse < best_score:
                best_score = mse
                best_params = (p, d, q)
        except Exception as e:
            continue

    print(f"Best Parameters: ARIMA{best_params} with MSE = {best_score:.3f}")

    # Step 3: Fit the best model on the full data
    print(f"Fitting ARIMA{best_params} on the full dataset...")
    final_model = ARIMA(data, order=best_params)
    final_results = final_model.fit()

    # Step 4: Evaluate the performance for different test sizes
    performance_results = []
    for test_days in test_sizes:
        print(f"\nEvaluating for multiple test sizes...")

        # Split data into training and testing sets
        train, test = data[:-test_days], data[-test_days:]

        # Forecast on the test set using the fitted model
        forecast = final_results.forecast(steps=test_days)

        # Evaluate performance
        mse = mean_squared_error(test, forecast)
        mae = mean_absolute_error(test, forecast)
        # r2 = r2_score(test, forecast)

        # Store results
        performance_results.append({
            'Forecast period (days)': test_days,
            'Best p, d, q': best_params,
            'MSE': mse,
            'MAE': mae,
            # 'R²': r2
        })

    # Display performance summary
    print("\nPerformance Summary:")
    results_df = pd.DataFrame(performance_results)
    print(results_df)

    # Step 5: Make predictions for the entire time series
    # print("Generating predictions for the entire time series...")
    predictions = final_results.predict(start=0, end=len(data)-1)

    # Step 6: return arima predictions
    return predictions

################################################################################
#fit ARIMA model on streamflow data
# Read the original data
df = pd.read_csv('og_file.csv')
#make date column using year, month, day
df['date'] = pd.to_datetime(df[['year', 'month', 'day']])
df = df.set_index('date')
#handle missing values
df = df.interpolate(method='time').fillna(method='bfill').fillna(method='ffill') #ffill and bfill fills na with previous and next values based on time index of the data
df_arima = df[['streamflow']].asfreq('D')
#call the function
output_arima = arima_full_process(df_arima, column_name="streamflow", test_sizes=[1, 3, 7, 15, 30], p_range=(0, 2), d_range=(0, 1), q_range=(0, 2)) # Test sizes of 3, 7, 15, 30 days

#show summary performance of arima on entire dataset, mse, mae, r2
print("\nPerformance Metrics for Entire Dataset, observed vs ARIMA streamflow:")
mse = mean_squared_error(df_arima['streamflow'], output_arima)
mae = mean_absolute_error(df_arima['streamflow'], output_arima)
r2 = r2_score(df_arima['streamflow'], output_arima)

print(f"Mean Squared Error (MSE): {mse:.3f}")
print(f"Mean Absolute Error (MAE): {mae:.3f}")
print(f"R^2 Score: {r2:.3f}")

print('Time taken to complete arima:', time.time()-start_time)


################################################################################
#fit XGBoost model to predict residuals of ARIMA model
#add residuals column to the original dataframe
df['residuals'] = df['streamflow'] - output_arima
true_streamflow = df['streamflow']
#drop streamflow column and reset index
df = df.drop(columns=['streamflow']).reset_index()
#drop date column
df = df.drop(columns=['date'])

#split data, do hyperparameter tuning and fit model
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV

# Split the data into training and testing sets
X = df.drop(columns=['residuals'])
y = df['residuals']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Define the parameter grid to search
param_grid = {
    'n_estimators': [100, 200, 500],
    'max_depth': [2, 5, 10], 
    'learning_rate': [0.01, 0.1, 0.3], 
    # 'subsample': [0.5, 0.7, 1],
    # 'colsample_bytree': [0.5, 0.7, 1],
    'n_jobs': [-1], # Use all available cores
}

# Initialize the XGBRegressor
xgb = XGBRegressor(random_state=0)

print("Hyperparameter tuning: Performing Grid Search to find the best parameters...")
# Perform Grid Search to find the best parameters
xgb_grid = GridSearchCV(xgb, param_grid, cv=3)
xgb_grid.fit(X_train, y_train)

# Get the best model
best_xgb = xgb_grid.best_estimator_

# Make predictions
y_pred = best_xgb.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

#Display the best hyper parameters
print("Best Hyperparameters found using Grid Search:")
print(xgb_grid.best_params_)

# Display the evaluation metrics on test set
print("\nPerformance Metrics for XGBoost model on residuals:")
print(f"Mean Squared Error (MSE): {mse:.3f}")
print(f"Mean Absolute Error (MAE): {mae:.3f}")
print(f"R^2 Score: {r2:.3f}")

#make feature importance plot
importances = best_xgb.feature_importances_
plt.figure(figsize=(6, 4))
plt.bar(X.columns, importances)
plt.title("Feature Importance/Variance Explained Plot")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


#make final streamflow predictions by adding ARIMA predictions and XGBoost predictions
final_streamflow_predictions = output_arima + best_xgb.predict(X)
#make it array
final_streamflow_predictions = np.array(final_streamflow_predictions)
#add final streamflow predictions to the original dataframe
df['final_streamflow_predictions'] = final_streamflow_predictions

#display summary performance of arima+xgboost on entire dataset, mse, mae, r2
print("\nPerformance Metrics for Entire Dataset, observed vs ARIMA+XGBoost streamflow:")
mse = mean_squared_error(true_streamflow, df['final_streamflow_predictions'])
mae = mean_absolute_error(true_streamflow, df['final_streamflow_predictions'])
r2 = r2_score(true_streamflow, df['final_streamflow_predictions'])

print(f"Mean Squared Error (MSE): {mse:.3f}")
print(f"Mean Absolute Error (MAE): {mae:.3f}")
print(f"R^2 Score: {r2:.3f}")

print('Time taken to complete arima+xgboost:', time.time()-start_time)