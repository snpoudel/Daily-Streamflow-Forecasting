import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from mpi4py import MPI
#ignore warnings
import warnings
warnings.filterwarnings('ignore')

#set up MPI communicator
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

#create a start time stamp
start_time = pd.Timestamp.now()

station_id = pd.read_csv('station_id.csv', dtype={'station_id': str})

# Load and preprocess the dataset
# id = '01096000'
id = station_id['station_id'][rank]

file_path = f"data/hbv_input_{id}.csv"  # Replace with your file path
df = pd.read_csv(file_path)
df.drop(columns=['id', 'latitude', 'date'], inplace=True)
#change column names
df.columns = ['year', 'month', 'day', 'temp', 'precip', 'streamflow']

#use 1994 to 2009 for training and 2009 to 2015 for testing
train = df[df['year'] < 2009]
test = df[df['year'] >= 2009]

#test set for model evaluation
test_pred = test.copy()

#test set for model forecasting
test = test.reset_index(drop=True)

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
        test_monthly = test_df[test_df['month'] == month]
        for day in range(1, 29):
            test_day = test_monthly[test_monthly['day'] == day]
            X_test = test_day.drop(columns=['streamflow'])
            forecast = best_xgb.predict(X_test)
            forecast_df = pd.concat([forecast_df, pd.DataFrame({'year': [test_year], 'month': [month], 'day': [day], 'Observed': [test_day['streamflow'].values[0]], 'Forecast': [forecast[0]]})], ignore_index=True)

#save forecast_df as csv
forecast_df.to_csv(f'output/xgboost/xgboost{id}.csv', index=False)    
#save cv-rmse, test-rmse, r2 score as csv by making a dataframe
results_metrics = pd.DataFrame({'CV RMSE': [np.sqrt(-xgb_grid.best_score_)], 'Test RMSE': [rmse], 'R2 Score': [r2]})
results_metrics.to_csv(f'output/xgboost/metrics/metrics{id}.csv', index=False)

#total time taken, save as csv
if id == '01096000':
    end_time = pd.Timestamp.now()
    time_taken = end_time - start_time
    time_taken_df = pd.DataFrame({'time_taken': [time_taken]})
    time_taken_df.to_csv(f'output/time_taken/xgboost{id}.csv', index=False)
print('Completed!!!')
# # visualize the actual vs forecasted values from the forecast_df
# #find average of observed and forecasted values for each day
# avg_day = forecast_df.groupby('day').mean()
# plt.figure(figsize=(6, 4))
# plt.plot(avg_day['Observed'], label='Observed')
# plt.plot(avg_day['Forecast'], label='Forecast')
# plt.title("Average Observed vs Forecasted Streamflow")
# plt.xlabel("Day")
# plt.ylabel("Streamflow")
# plt.ylim(0, None)
# plt.legend()
# plt.show()

# #find rmse for each day forecast and observed values
# rmse_df = pd.DataFrame(columns=['Day', 'RMSE'])
# for day in list(range(1,29)):
#     day_df = forecast_df[forecast_df['day'] == day]
#     mse = mean_squared_error(day_df['Observed'], day_df['Forecast'])
#     rmse = mse ** 0.5
#     rmse_df = pd.concat([rmse_df, pd.DataFrame({'Day': [day], 'RMSE': [rmse]})])
# #plot rmse for each day
# plt.figure(figsize=(6, 4))
# plt.plot(rmse_df['Day'], rmse_df['RMSE'])
# plt.title("RMSE for different forecasting horizons")
# plt.xlabel("Day")
# plt.ylabel("RMSE")
# plt.ylim(0, None)
# plt.show()