import pandas as pd
import numpy as np
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from mpi4py import MPI
#ignore warnings
import warnings
warnings.filterwarnings('ignore')

#rmse function
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


#set up MPI communicator
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

#create a start time stamp
start_time = pd.Timestamp.now()

station_id = pd.read_csv('station_id.csv', dtype={'station_id': str})

# Load and preprocess the dataset
# rank = 25
id = station_id['station_id'][rank]
file_path = f"data/hbv_input_{id}.csv"  # Replace with your file path
df = pd.read_csv(file_path)
df.drop(columns=['id', 'latitude', 'date'], inplace=True)
#change column names 
df.columns = ['year', 'month', 'day', 'temp', 'precip', 'streamflow']

df['date'] = pd.to_datetime(df[['year', 'month', 'day']])
df = df.drop(columns=['year', 'month', 'day'])
df.set_index("date", inplace=True)
df.index = df.index.to_period('D')
#exogeneous variables
exo_vars = df[['temp', 'precip']]
#target variable
target_var = df[['streamflow']]

#use 1994 to 2009 for training and 2009 to 2015 for testing
y_train, y_test = target_var[df.index.year < 2009], target_var[df.index.year >= 2009]
x_train, x_test = exo_vars[df.index.year < 2009], exo_vars[df.index.year >= 2009]

# Fit auto_arima to find the best ARIMA parameters on the training set
print("Finding the best ARIMA parameters...")
arima_model = auto_arima(y_train, 
                         exogeneous=x_train, # exogeneous variables
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
arima_model = SARIMAX(y_train, order=best_order, seasonal_order=best_seasonal_order, exog=x_train).fit()
# Make prediction on entire training set and calculate residuals
predict_train = arima_model.fittedvalues
residuals = y_train['streamflow'] - predict_train
residuals = residuals.values

# # Analyze residuals, plot time series of residuals
# plt.figure(figsize=(6, 4))
# plt.plot(residuals)
# plt.title("Residuals of ARIMA Model")
# plt.xlabel("Time")
# plt.ylabel("Residuals")
# plt.show()

# # Ljung-Box test for residuals to check for autocorrelation in residuals
# # Null hypothesis: Residuals are white noise (no autocorrelation)
# # If p-value < 0.05, reject null hypothesis, residuals are not white noise
# lb_test_results = acorr_ljungbox(residuals, lags=50) 
# lb_test_stat = lb_test_results['lb_stat'].values
# lb_p_value = lb_test_results['lb_pvalue'].values
# #make boxplot of lb_p_value, also show points, and threshold line at 0.05
# plt.figure(figsize=(6, 4))
# plt.boxplot(lb_p_value, showfliers=True)
# plt.axhline(y=0.05, color='r', linestyle='--', label="Threshold at 0.05")
# plt.title("Ljung-Box Test P-Values for lag 1-50")
# plt.ylabel("P-Value")
# plt.legend()
# plt.show()

# Use the best fitted ARIMA model to forecast 28-day streamflow using exogeneous variables
forecast_df = pd.DataFrame(columns=['Date', 'Observed', 'Forecast'])
#make first 28-day forecast
first_forecast = arima_model.forecast(steps=28, exog=x_test[0:28])
forecast_df = pd.concat([forecast_df, pd.DataFrame({
    'Date': first_forecast.index, 
    'Observed': y_test['streamflow'].values[0:28], 
    'Forecast': first_forecast.values
})], ignore_index=True)
#make forecast for each month for remaining years
for test_year in range(2009, 2016):
    x_test_year = x_test[x_test.index.year == test_year]
    y_test_year = y_test[y_test.index.year == test_year]
    for month in range(1, 12):  # Loop through all 12 months
        x_test_monthly = x_test_year[x_test_year.index.month == month]
        y_test_monthly = y_test_year[y_test_year.index.month == month]
        y_test_monthlyplus1 = y_test_year[y_test_year.index.month == month+1]
        x_test_forecast = x_test_year[x_test_year.index.month == (month+1)]
        try:
            # Fit SARIMAX model and forecast for 15 steps
            results = SARIMAX(y_test_monthly, order=best_order, seasonal_order=best_seasonal_order, exog=x_test_monthly).fit()
            forecast_monthly = results.forecast(steps=28, exog=x_test_forecast[0:28])
            
            # Append forecast data to the DataFrame
            forecast_df = pd.concat([forecast_df, pd.DataFrame({
                'Date': forecast_monthly.index, 
                'Observed': y_test_monthlyplus1['streamflow'].values[0:28], 
                'Forecast': forecast_monthly.values
            })], ignore_index=True)
        except np.linalg.LinAlgError:
            print(f"LinAlgError for year {test_year}, month {month}. Skipping this month.")
forecast_df = forecast_df.reset_index(drop=True)
forecast_df['day'] = forecast_df['Date'].dt.day

#save as csv
forecast_df.to_csv(f'output/sarimax/sarimax{id}.csv', index=False)
#save best order and seasonal order as csv
best_arima_params = pd.DataFrame({'order': [best_order], 'seasonal_order': [best_seasonal_order]})
best_arima_params.to_csv(f'output/sarimax/parameters/params{id}.csv', index=False)

#total time taken, save as csv
if id == '01096000':
    end_time = pd.Timestamp.now()
    time_taken = end_time - start_time
    time_taken_df = pd.DataFrame({'time_taken': [time_taken]})
    time_taken_df.to_csv(f'output/time_taken/sarimax{id}.csv', index=False)

print('Completed!!!')

# # visualize the actual vs forecasted values from the forecast_df
# #find average of observed and forecasted values for each day
# avg_day = forecast_df.groupby('day').mean()
# plt.figure(figsize=(6, 4))
# plt.plot(avg_day['Observed'], label='Observed')
# plt.plot(avg_day['Forecast'], label='Forecast')
# plt.title("Observed vs Forecasted Streamflow")
# plt.xlabel("Day")
# plt.ylabel("Streamflow")
# plt.ylim(0, None)
# plt.legend()
# plt.show()

# #find rmse for each day forecast and observed values
# rmse_df = pd.DataFrame(columns=['Day', 'RMSE'])
# for day in list(range(1,28)):
#     day_df = forecast_df[forecast_df['day'] == day]
#     rmse_val = rmse(day_df['Observed'], day_df['Forecast'])
#     rmse_df = pd.concat([rmse_df, pd.DataFrame({'Day': [day], 'RMSE': [rmse_val]})])
# #plot rmse for each day
# plt.figure(figsize=(6, 3))
# plt.plot(rmse_df['Day'], rmse_df['RMSE'])
# plt.title("RMSE for different forecasting horizons")
# plt.xlabel("Day")
# plt.ylabel("RMSE")
# plt.ylim(0, None)
# plt.show()

# #find r2 score for each day forecast and observed values
# r2_df = pd.DataFrame(columns=['Day', 'R2'])
# for day in list(range(1,28)):
#     day_df = forecast_df[forecast_df['day'] == day]
#     r2 = r2_score(day_df['Observed'], day_df['Forecast'])
#     r2_df = pd.concat([r2_df, pd.DataFrame({'Day': [day], 'R2': [r2]})])
# #plot r2 score for each day
# plt.figure(figsize=(6, 3))
# plt.plot(r2_df['Day'], r2_df['R2'])
# plt.title("R2 Score for different forecasting horizons")
# plt.xlabel("Day")
# plt.ylabel("R2 Score")
# # plt.ylim(0, 1)
# plt.show()
