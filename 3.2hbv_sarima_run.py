#import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima
from statsmodels.stats.diagnostic import acorr_ljungbox
from hbv_model import hbv
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
# rank = 25
# Load and preprocess the dataset
# id = '01096000'
station_id = station_id['station_id'][rank]

################################################################################################################################################################################################################################
##--HBV--##
param_value = pd.read_csv(f'output/hbv/parameters/param{station_id}.csv')
param_value = param_value.drop(columns=['station_id'])
param_value = param_value.values[0]
df = pd.read_csv(f"data/hbv_input_{station_id}.csv")

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
df['residuals'] = q_obs - q_sim

#only keep data for training period
residuals_train = df[df["year"] < 2009]
residuals_train['date'] = pd.to_datetime(residuals_train['date'])
residuals_train = residuals_train[['date', 'residuals']].set_index('date')
#remove first year of residuals,1994, as HBV takes some time to adjust its internal states correctly
residuals_train = residuals_train[residuals_train.index.year > 1994]

#only keep data for testing period
residuals_test = df[df["year"] >= 2009]
residuals_test['date'] = pd.to_datetime(residuals_test['date'])
residuals_test = residuals_test[['date', 'residuals']].set_index('date')

#get forecasted dataframe csv 
forecast_df = pd.read_csv(f"output/hbv/hbv{station_id}.csv")
forecast_df['Date'] = pd.to_datetime(forecast_df['Date'])

################################################################################################################################################################################################################################
##--SARIMA--##
# Fit auto_arima to find the best ARIMA parameters on the training set
y_train_arima = residuals_train['residuals']
y_test_arima = residuals_test['residuals']

print("Finding the best ARIMA parameters...")
arima_model = auto_arima(y_train_arima, # residuals from XGBoost model
                         exogeneous=None, # exogeneous variables
                         seasonal=False, # seasonal is true if data has seasonality
                         m=12, # seasonility of 12 months
                         approximation=True , # use approximation to speed up the search
                         stepwise=True, # stepwise is true to speed up the search
                         suppress_warnings=True, # suppress warnings
                         error_action="ignore", # ignore orders that don't converge
                         trace=False) # print results while training
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

# Use the best fitted ARIMA model to forecast 28-day streamflow residuals
#make first 28-day forecast
resid_forecast_df = pd.DataFrame(columns=['Date', 'Observed', 'Forecast'])
first_forecast = arima_model.forecast(steps=28, exog=None)
resid_forecast_df = pd.concat([resid_forecast_df, pd.DataFrame({
    'Date': y_test_arima.index[0:28], 
    'Observed': y_test_arima.values[0:28], 
    'Forecast': first_forecast
})], ignore_index=True)
#make 28-day forecast for each month in remaining years
for test_year in range(2009, 2016):
    y_test_arima_year = y_test_arima[y_test_arima.index.year == test_year]
    for month in range(1, 12):  # Loop through all 12 months
        y_test_arima_monthly = y_test_arima_year[y_test_arima_year.index.month == month]
        y_test_arima_monthplus1 = y_test_arima_year[y_test_arima_year.index.month == (month+1)]
        try:
            # Fit SARIMAX model and forecast for 15 steps
            results = SARIMAX(y_test_arima_monthly, order=best_order, seasonal_order=best_seasonal_order, exog=None).fit(maxiter=500, disp=False)
            forecast_monthly = results.forecast(steps=28, exog=None)
            
            # Append forecast data to the DataFrame
            resid_forecast_df = pd.concat([resid_forecast_df, pd.DataFrame({
                'Date': forecast_monthly.index, 
                'Observed': y_test_arima_monthplus1.values[0:28], 
                'Forecast': forecast_monthly.values
            })], ignore_index=True)
        except np.linalg.LinAlgError:
            print(f"LinAlgError for year {test_year}, month {month}. Skipping this month.")
resid_forecast_df = resid_forecast_df.reset_index(drop=True)
resid_forecast_df['Date'] = pd.to_datetime(resid_forecast_df['Date'])

#Final streamflow forecast is sum of HBV streamflow forecast and ARIMA residuals forecast
#make merged dataframe on Date
final_forecast_df = pd.merge(forecast_df, resid_forecast_df, on='Date', suffixes=('_hbv', '_arima'))
final_forecast_df['Final_Forecast'] = final_forecast_df['Forecast_hbv'] + final_forecast_df['Forecast_arima']

#save forecasted values
final_forecast_df.to_csv(f"output/hbv_sarima/hbv_sarima{station_id}.csv", index = False)
#save sarima parameters
sarima_param = pd.DataFrame({'order': [best_order], 'seasonal_order': [best_seasonal_order]})
sarima_param.to_csv(f"output/hbv_sarima/parameters/param{station_id}.csv", index = False)
#total time taken, save as csv
if station_id == '01096000':
    end_time = pd.Timestamp.now()
    time_taken = end_time - start_time
    time_taken_df = pd.DataFrame({'time_taken': [time_taken]})
    time_taken_df.to_csv(f'output/time_taken/hbv_sarima{station_id}.csv', index=False)
print('Completed!!!')


################################################################################################################################################################################################################################
##--Forecasting with precipitation uncertainty--##
#get HBV forecasted dataframe csv 
df = pd.read_csv(f"data/hbv_input_{station_id}.csv")
df = df[df['year'] >= 2009].reset_index(drop=True)

#run for 10 iterations of synthetic precip with uncertainty
forecast_df_uq = pd.DataFrame()
for iter in range(10):
    precip1, precip2, precip3, precip4 = synthetic_precip(df['precip']) #synthetic precip with rmse 1, 2, 3, 4 mm/day
    precip_uq = [df['precip'], precip1, precip2, precip3, precip4]
    error_level = ['rmse0', 'rmse1', 'rmse2', 'rmse3', 'rmse4']

    for index, puq in enumerate(precip_uq):
        forecast_df = pd.read_csv(f"output/hbv/puq/hbv{station_id}.csv")
        #only keep dataframe correspond to error level
        forecast_df = forecast_df[forecast_df['error'] == error_level[index]]
        forecast_df['Date'] = pd.to_datetime(forecast_df['Date'])
        ##--SARIMA--##
        forecast_sarima = final_forecast_df[['Date', 'Forecast_arima']]
        forecast_sarima['Date'] = pd.to_datetime(forecast_sarima['Date'])
        #merge forecast_df and forecast_sarima on Date
        forecast_df_temp = pd.merge(forecast_df, forecast_sarima, on='Date')
        forecast_df_temp['Final_Forecast'] = forecast_df_temp['Forecast'] + forecast_df_temp['Forecast_arima']
        forecast_df_temp['iter'] = iter
        #append to forecast_df_uq
        forecast_df_uq = pd.concat([forecast_df_uq, forecast_df_temp], ignore_index=True)

forecast_df_uq = forecast_df_uq.reset_index(drop=True)
forecast_df_uq['Date'] = pd.to_datetime(forecast_df_uq['Date'])

#save forecasted values
forecast_df_uq.to_csv(f"output/hbv_sarima/puq/hbv_sarima{station_id}.csv", index = False)

forecast_df_uq[forecast_df_uq['error'] == 'rmse0']