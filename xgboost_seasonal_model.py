import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV

# Load and preprocess the dataset
file_path = "og_file.csv"  # Replace with your file path
df = pd.read_csv(file_path)

#add lagged features
# Add lagged features
for lag in range(1, 4): #add lagged features for 1 to 3 days
    for col in ['temp', 'precip', 'streamflow']:
        df[f'{col}_lag{lag}'] = df[col].shift(lag)
df = df.dropna()

#add seasonality features
df['date'] = pd.to_datetime(df[['year', 'month', 'day']])
df.set_index('date', inplace=True)
# Extract temporal features
df['month'] = df.index.month
df['day_of_year'] = df.index.day_of_year
# Create cyclic features for month
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
# Create cyclic features for day_of_year
df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
# Drop unnecessary columns 
df.drop(columns=['day','month', 'day_of_year'], inplace=True)

#use 1994 to 2009 for training and 2009 to 2015 for testing
train = df[df['year'] < 2009]
test = df[df['year'] >= 2009]

#test set for model evaluation
test_pred = test.copy()

#test set for model forecasting
# Replace streamflow lag with NaN except for the first row in the test set
# test = test.reset_index(drop=True)
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

#evaluate the model on the test set, rmse and r2 score
X_test = test_pred.drop(columns=['streamflow'])
y_test = test_pred['streamflow']
y_pred = best_xgb.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print(f"Test RMSE: {rmse:.2f} & R2 Score: {r2:.2f}")

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
        test_monthly = test_df[test_df.index.month == month]
        for day in range(1, 29):
            test_day = test_monthly[test_monthly.index.day == day]
            if day == 1:
                streamflow_lag1 = test_pred.loc[(test_pred['year'] == test_year) & (test_pred.index.month == month) & (test_pred.index.day == day), 'streamflow_lag1'].values[0]
                streamflow_lag2 = test_pred.loc[(test_pred['year'] == test_year) & (test_pred.index.month == month) & (test_pred.index.day == day), 'streamflow_lag2'].values[0]
                streamflow_lag3 = test_pred.loc[(test_pred['year'] == test_year) & (test_pred.index.month == month) & (test_pred.index.day == day), 'streamflow_lag3'].values[0]
            test_day = test_day.assign(streamflow_lag1=streamflow_lag1, streamflow_lag2=streamflow_lag2, streamflow_lag3=streamflow_lag3)
            X_test = test_day.drop(columns=['streamflow'])
            forecast = best_xgb.predict(X_test)
            forecast_df = pd.concat([forecast_df, pd.DataFrame({'year': [test_year], 'month': [month], 'day': [day], 'Observed': [test_day['streamflow'].values[0]], 'Forecast': [forecast[0]]})], ignore_index=True)
            streamflow_lag1, streamflow_lag2, streamflow_lag3 = forecast[0], streamflow_lag1, streamflow_lag2
            

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


#adding seasonal features to the dataset did not improve the model performance