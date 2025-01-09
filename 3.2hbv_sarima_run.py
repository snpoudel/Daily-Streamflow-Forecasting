#import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima
from statsmodels.stats.diagnostic import acorr_ljungbox
from hbv_model import hbv
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
df['residuals'] = q_obs - q_sim

#only keep data for training period
residuals_train = df[df["year"] < 2009]
residuals_train['date'] = pd.to_datetime(residuals_train['date'])
residuals_train = residuals_train[['date', 'residuals']].set_index('date')

#only keep data for testing period
residuals_test = df[df["year"] >= 2009]
residuals_test['date'] = pd.to_datetime(residuals_test['date'])
residuals_test = residuals_test[['date', 'residuals']].set_index('date')

#get forecasted dataframe csv 
forecast_df = pd.read_csv(f"output/hbv/hbv{station_id}.csv")
forecast_df['Date'] = pd.to_datetime(forecast_df['Date'])



################################################################################################################################################################################################################################
##--SARIMA--##
#check for autocorrelation in residuals
# Ljung-Box test for residuals to check for autocorrelation in residuals
# Null hypothesis: Residuals are white noise (no autocorrelation)
# If p-value < 0.05, reject null hypothesis, residuals are not white noise
lb_test_results = acorr_ljungbox(residuals_train, lags=50)
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

print("Finding the best ARIMA parameters...")
arima_model = auto_arima(y_train_arima, # residuals from XGBoost model
                         exogeneous=None, # exogeneous variables
                         seasonal=True, # seasonal is true if data has seasonality
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
# # visualize the actual vs forecasted values from the forecast_df
# final_forecast_df['day'] = final_forecast_df['Date'].dt.day
# #find average of observed and forecasted values for each day
# avg_day = final_forecast_df.groupby('day').mean()
# plt.figure(figsize=(6, 4))
# plt.plot(avg_day['Observed_hbv'], label='Observed')
# plt.plot(avg_day['Final_Forecast'], label='HBV+SARIMA Forecast')
# plt.title("Observed vs Forecasted Streamflow")
# plt.xlabel("Day")
# plt.ylabel("Streamflow")
# plt.ylim(0, None)
# plt.legend()
# plt.grid()
# plt.show()


# #find rmse for each day forecast and observed values
# final_rmse_df = pd.DataFrame(columns=['Day', 'RMSE'])
# for day in list(range(1,28)):
#     day_df = final_forecast_df[final_forecast_df['day'] == day]
#     mse = mean_squared_error(day_df['Observed_hbv'], day_df['Final_Forecast'])
#     rmse = mse ** 0.5
#     final_rmse_df = pd.concat([final_rmse_df, pd.DataFrame({'Day': [day], 'RMSE': [rmse]})])
# #plot rmse for each day
# plt.figure(figsize=(6, 4))
# plt.plot(final_rmse_df['Day'], final_rmse_df['RMSE'])
# plt.title("RMSE for different forecasting horizons")
# plt.xlabel("Day")
# plt.ylabel("RMSE")
# plt.ylim(0, None)
# plt.show()
