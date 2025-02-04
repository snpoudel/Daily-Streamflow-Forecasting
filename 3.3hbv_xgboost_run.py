#import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from hbv_model import hbv #imported from local python script
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from synthetic_precip import synthetic_precip #takes original precip and returns synthetic precip with rmse 1, 2, 3, & 4 mm/day
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
station_id = station_id['station_id'][rank]
################################################################################################################################################################################################################################
##--HBV--##
param_value = pd.read_csv(f'output/hbv/parameters/param{station_id}.csv')
param_value = param_value.drop(columns=['station_id'])
param_value = param_value.values[0]

df = pd.read_csv(f"data/hbv_input_{station_id}.csv")
#only used data upto 2005 for calibration
# df = df[df["year"] < 2009]
#hbv model input
p = df["precip"]
temp = df["tavg"]
date = df["date"]
latitude = df["latitude"]
routing = 1 # 0: no routing, 1 allows running
q_obs = df["qobs"] #validation data / observed flow


#run the model to the entire dataset
q_sim,_ ,_  = hbv(param_value, p, temp, date, latitude, routing)
#add the simulated flow to the dataframe
df['qsim'] = q_sim
#add residuals to the dataframe
df['residual'] = q_obs - q_sim

#only keep data for training period
df_train_residual = df[df["year"] < 2009]
df_train_residual = df_train_residual[['date', 'precip', 'tavg', 'residual']]
df_train_residual = df_train_residual.rename(columns={'tavg': 'temp'})


#only keep data for testing period
df_test_residual = df[df["year"] >= 2009]
df_test_residual = df_test_residual[['date', 'precip', 'tavg', 'residual']]
df_test_residual = df_test_residual.rename(columns={'tavg': 'temp'})

#get forecasted dataframe csv 
forecast_df = pd.read_csv(f"output/hbv/hbv{station_id}.csv")
forecast_df['Date'] = pd.to_datetime(forecast_df['Date'])
################################################################################################################################################################################################################################
##--XGBoost--##
#XGBoost model for streamflow residual forecasting
# combine train and test residuals dataframes
df = pd.concat([df_train_residual, df_test_residual], ignore_index=True)
df['date'] = pd.to_datetime(df['date'])  # Convert 'date' column to datetime
#add year, month, day columns to the dataframe
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
#drop date column
df = df.drop(columns=['date'])

#add lagged features
# Add lagged features
for lag in range(1, 4): #add lagged features for 1 to 3 days
    for col in ['temp', 'precip', 'residual']:
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
test.loc[:, ['residual_lag1', 'residual_lag2', 'residual_lag3']] = np.nan

#prepare xtrain and ytrain for training
X_train = train.drop(columns=['residual'])
y_train = train['residual']

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
X_test = test_pred.drop(columns=['residual'])
y_test = test_pred['residual']
y_pred = best_xgb.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print(f"Test RMSE: {rmse:.2f} & R2 Score: {r2:.2f}")


# Make 28-day forecast
forecast_residual_df = pd.DataFrame(columns=['year', 'month', 'day', 'Observed', 'Forecast'])
for test_year in range(2009, 2016):
    test_df = test[test['year'] == test_year]
    for month in range(1, 12):
        test_monthly = test_df[test_df['month'] == month]
        for day in range(1, 29):
            test_day = test_monthly[test_monthly['day'] == day]
            if day == 1:
                residual_lag1 = test_pred.loc[(test_pred['year'] == test_year) & (test_pred['month'] == month) & (test_pred['day'] == day), 'residual_lag1'].values[0]
                residual_lag2 = test_pred.loc[(test_pred['year'] == test_year) & (test_pred['month'] == month) & (test_pred['day'] == day), 'residual_lag2'].values[0]
                residual_lag3 = test_pred.loc[(test_pred['year'] == test_year) & (test_pred['month'] == month) & (test_pred['day'] == day), 'residual_lag3'].values[0]
            test_day = test_day.assign(residual_lag1=residual_lag1, residual_lag2=residual_lag2, residual_lag3=residual_lag3)
            X_test = test_day.drop(columns=['residual'])
            forecast = best_xgb.predict(X_test)
            forecast_residual_df = pd.concat([forecast_residual_df, pd.DataFrame({'year': [test_year], 'month': [month], 'day': [day], 'Observed': [test_day['residual'].values[0]], 'Forecast': [forecast[0]]})], ignore_index=True)
            residual_lag1, residual_lag2, residual_lag3 = forecast[0], residual_lag1, residual_lag2
forecast_residual_df = forecast_residual_df.reset_index(drop=True)            
forecast_residual_df['Date'] = pd.to_datetime(forecast_residual_df[['year', 'month', 'day']])


#Final streamflow forecast is the sum of the HBV model forecast and the XGBoost residual forecast
#merge the forecasted residuals with the forecasted streamflow from HBV model based on date
final_forecast_df = pd.merge(forecast_df, forecast_residual_df, on='Date', suffixes=('_hbv', '_xgb'))
final_forecast_df['Final_Forecast'] = final_forecast_df['Forecast_hbv'] + final_forecast_df['Forecast_xgb']

#save train cv rmse, test rmse and r2 score
df_nse = pd.DataFrame(columns=['train_rmse', 'test_rmse', 'test_r2'])
df_nse.to_csv(f"output/hbv_xgboost/metrics/metrics{station_id}.csv", index = False)

#save forecasted values
final_forecast_df.to_csv(f"output/hbv_xgboost/hbv_xgboost{station_id}.csv", index = False)

#total time taken, save as csv
if station_id == '01096000':
    end_time = pd.Timestamp.now()
    time_taken = end_time - start_time
    time_taken_df = pd.DataFrame({'time_taken': [time_taken]})
    time_taken_df.to_csv(f'output/time_taken/hbv_xgboost{station_id}.csv', index=False)
print('Completed!!!')



################################################################################################################################################################################################################################
##--Forecast with precipitation uncertainty--##
#read hbv foreacast under precipitation uncertainty
final_forecast_df = pd.read_csv(f'output/hbv/puq/hbv{station_id}.csv')
# Make 28-day forecast
train_uq = train.tail(3).reset_index(drop=True) #get last 3 day of train
test_uq = pd.concat([train_uq, test], ignore_index=True) #combine last 3 days of train with test
# test_uq = test_uq.drop(columns=['precip_lag1', 'precip_lag2', 'precip_lag3'])

#iterate for 10 synthetic precip with uncertainty
forecast_df = pd.DataFrame()
for iter in range(10):
    precip1, precip2, precip3, precip4 = synthetic_precip(test_uq['precip']) #synthetic precip with rmse 1, 2, 3, 4 mm/day
    precip_uq = [test_uq['precip'], precip1, precip2, precip3, precip4]
    error_level = ['rmse0', 'rmse1', 'rmse2', 'rmse3', 'rmse4']

    for index, puq in enumerate(precip_uq):
        test_uq['precip'] = puq
        test_uq.loc[:, ['precip_lag1', 'precip_lag2', 'precip_lag3']] = np.nan
        for lag in range(1, 4): #add lagged features for 1 to 3 days
            for col in [ 'precip']:
                test_uq[f'{col}_lag{lag}'] = test_uq[col].shift(lag)
        test_uq = test_uq[test_uq['year'] >= 2009]
        for test_year in range(2009, 2016):
            test_df = test_uq[test_uq['year'] == test_year]
            for month in range(1, 12):
                test_monthly = test_df[test_df['month'] == month]
                for day in range(1, 29):
                    test_day = test_monthly[test_monthly['day'] == day]
                    if day == 1:
                        residual_lag1 = test_pred.loc[(test_pred['year'] == test_year) & (test_pred['month'] == month) & (test_pred['day'] == day), 'residual_lag1'].values[0]
                        residual_lag2 = test_pred.loc[(test_pred['year'] == test_year) & (test_pred['month'] == month) & (test_pred['day'] == day), 'residual_lag2'].values[0]
                        residual_lag3 = test_pred.loc[(test_pred['year'] == test_year) & (test_pred['month'] == month) & (test_pred['day'] == day), 'residual_lag3'].values[0]
                    test_day = test_day.assign(residual_lag1=residual_lag1, residual_lag2=residual_lag2, residual_lag3=residual_lag3)
                    X_test = test_day.drop(columns=['residual'])
                    forecast = best_xgb.predict(X_test)
                    temp_df = pd.DataFrame({'year': [test_year], 'month': [month], 'day': [day], 'xgboost_observed': [test_day['residual'].values[0]], 'xgboost_forecast': [forecast[0]], 'error': error_level[index], 'iter': [iter]})
                    temp_df['date'] = pd.to_datetime(temp_df[['year', 'month', 'day']])
                    temp_df = temp_df.drop(columns=['year', 'month', 'day'])
                    #extract Forecast_hbv from final_forecast_df based on date
                    final_forecast_df = pd.read_csv(f'output/hbv/puq/hbv{station_id}.csv')
                    final_forecast_df = final_forecast_df[final_forecast_df['error'] == error_level[index]].reset_index(drop=True)
                    final_forecast_df['Date'] = pd.to_datetime(final_forecast_df['Date'])
                    temp_df['hbv_forecast'] = final_forecast_df.loc[final_forecast_df['Date'].isin(temp_df['date']) , 'Forecast'].values[0]
                    temp_df['hbv_observed'] = final_forecast_df.loc[final_forecast_df['Date'].isin(temp_df['date']) , 'Observed'].values[0]
                    #total forecast is the sum of hbv_forecast and xgboost_forecast
                    temp_df['total_forecast'] = temp_df['hbv_forecast'] + temp_df['xgboost_forecast']
                    forecast_df = pd.concat([forecast_df, temp_df], ignore_index=True)
                    residual_lag1, residual_lag2, residual_lag3 = forecast[0], residual_lag1, residual_lag2
                
#save forecast_df as csv
forecast_df.to_csv(f'output/hbv_xgboost/puq/hbv_xgboost{station_id}.csv', index=False)