#import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from hbv_model import hbv #imported from local python script
from geneticalgorithm import geneticalgorithm as ga # install package first
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
################################################################################################################################################################################################################################
##--CALIBRATE--##
time_start = pd.Timestamp.now()
#function that reads station ID, takes input for that ID and outputs calibrated parameters and nse values
#read input csv file
station_id = '01094400'
df = pd.read_csv(f"data/hbv_input_{station_id}.csv")
#only used data upto 2005 for calibration
df = df[df["year"] < 2009]
#hbv model input
p = df["precip"]
temp = df["tavg"]
date = df["date"]
latitude = df["latitude"]
routing = 1 # 0: no routing, 1 allows running
q_obs = df["qobs"] #validation data / observed flow

##genetic algorithm for hbv model calibration
#reference: https://github.com/rmsolgi/geneticalgorithm
#write a function you want to minimize
def nse(pars):
    q_sim,_ ,_ = hbv(pars, p, temp, date, latitude, routing)
    denominator = np.sum((q_obs - (np.mean(q_obs)))**2)
    numerator = np.sum((q_obs - q_sim)**2)
    nse_value = 1 - (numerator/denominator)
    return -nse_value #minimize this (use negative sign if you need to maximize)

varbound = np.array([[1,1000], #fc #lower and upper bounds of the parameters
                    [1,7], #beta
                    [0.01,0.99], #pwp
                    [1,999], #l
                    [0.01,0.99], #ks
                    [0.01,0.99], #ki
                    [0.0001, 0.99], #kb
                    [0.001, 0.99], #kperc
                    [0.5,2], #coeff_pet
                    [0.01,10], #ddf
                    [0.5,1.5], #scf
                    [-1,4], #ts
                    [-1,4], #tm
                    [-1,4], #tti
                    [0, 0.2], #whc
                    [0.1,1], #crf
                    [1,10]]) #maxbas

algorithm_param = {
    'max_num_iteration': 5,              # Generations, higher is better, but requires more computational time
    'max_iteration_without_improv': None,   # Stopping criterion for lack of improvement
    'population_size': 10,                 #1500 Number of parameter-sets in a single iteration/generation(to start with population 10 times the number of parameters should be fine!)
    'parents_portion': 0.3,                 # Portion of new generation population filled by previous population
    'elit_ratio': 0.01,                     # Portion of the best individuals preserved unchanged
    'crossover_probability': 0.3,           # Chance of existing solution passing its characteristics to new trial solution
    'crossover_type': 'uniform',            # Create offspring by combining the parameters of selected parents
    'mutation_probability': 0.01            # Introduce random changes to the offspring’s parameters (0.1 is 10%)
}

model = ga(function = nse,
        dimension = 17, #number of parameters to be calibrated
        variable_type= 'real',
        variable_boundaries = varbound,
        algorithm_parameters = algorithm_param)

model.run()
#end of genetic algorithm
print('Time taken for calibration:', pd.Timestamp.now() - time_start)

#output of the genetic algorithm/best parameters
best_parameters = model.output_dict
param_value = best_parameters["variable"]
nse_value = best_parameters["function"]
nse_value = -nse_value #nse function gives -ve values, which is now reversed here to get true nse
#convert into a dataframe
df_param = pd.DataFrame(param_value).transpose()
df_param = df_param.rename(columns={0:"fc", 1:"beta", 2:"pwp", 3:"l", 4:"ks", 5:"ki",
                            6:"kb", 7:"kperc",  8:"coeff_pet", 9:"ddf", 10:"scf", 11:"ts",
                            12:"tm", 13:"tti", 14:"whc", 15:"crf", 16:"maxbas"})
df_param["station_id"] = str(station_id)
df_nse = pd.DataFrame([nse_value], columns=["nse"])
df_nse["station_id"] = str(station_id)
#save as a csv file
# df_param.to_csv(f"output/param_{station_id}.csv", index = False)
# df_nse.to_csv(f"output/nse_{station_id}.csv", index = False)

#run the model with the best parameters for the calibration data
q_sim_cal, storage_cal, sim_vars_cal = hbv(param_value, p, temp, date, latitude, routing)
#calculate residual and make a dataframe with date, precip, temp, residual
residual = q_obs - q_sim_cal
df_train_residual = pd.DataFrame({'date': date, 'precip': p, 'temp': temp, 'residual': residual})


#evaluate model performance using the best parameters for test data
df = pd.read_csv(f"data/hbv_input_{station_id}.csv")
test = df[df["year"] >= 2009].reset_index(drop=True)
test['date'] = pd.to_datetime(test['date'])
q_obs_test = test["qobs"]
q_sim_test, _, _ = hbv(param_value, test["precip"], test["tavg"], test["date"], test["latitude"], routing)
#calculate rmse, nse
rmse = mean_squared_error(q_obs_test, q_sim_test) ** 0.5
denominator = np.sum((q_obs_test - (np.mean(q_obs_test)))**2)
numerator = np.sum((q_obs_test - q_sim_test)**2)
nse_value = 1 - (numerator/denominator)
print(f"Test RMSE: {rmse:.2f} and NSE: {nse_value:.2f}")

#calculate residual and make a dataframe with date, precip, temp, residual
residual_test = q_obs_test - q_sim_test
df_test_residual = pd.DataFrame({'date': test['date'], 'precip': test['precip'], 'temp': test['tavg'], 'residual': residual_test})


################################################################################################################################################################################################################################
##--FORECAST--##
df = pd.read_csv(f"data/hbv_input_{station_id}.csv")
df['date'] = pd.to_datetime(df['date'])  # Convert 'date' column to datetime
#use data after 2009 for forecasting
test = df[df["year"] >= 2006]

forecast_df = pd.DataFrame(columns=['Date', 'Observed', 'Forecast'])
for test_year in range(2009, 2016):
    test_df = test[test['year'] == test_year]
    for month in range(1, 13):
        #get model states by running model for the last 365 days before the forecast period starts
        forecast_start_date = pd.Timestamp(f"{test_year}-{month}-01")
        forecast_end_date = forecast_start_date + pd.DateOffset(days=28)
        model_train_start_date = forecast_start_date - pd.DateOffset(years=1)
        df_states = df[(df['date'] >= model_train_start_date) & (df['date'] < forecast_end_date)]
        df_states = df_states.reset_index(drop=True)
        #run model
        q_sim,_ ,_ = hbv(param_value, df_states["precip"], df_states["tavg"], df_states["date"], df_states["latitude"], routing)
        #extract the last 28 days of the streamflow simulation
        forecast_streamflow = q_sim[-28:]
        forecast_date = df_states['date'][-28:]
        observed_streamflow = df_states['qobs'][-28:]
        forecast_df = pd.concat([forecast_df, pd.DataFrame({'Date': forecast_date, 'Observed': observed_streamflow, 'Forecast': forecast_streamflow})])
#save the forecasted values
forecast_df = forecast_df.reset_index(drop=True)

# visualize the actual vs forecasted values from the forecast_df
forecast_df['Date'] = pd.to_datetime(forecast_df['Date'])
forecast_df['day'] = forecast_df['Date'].dt.day
#find average of observed and forecasted values for each day
avg_day = forecast_df.groupby('day').mean()
plt.figure(figsize=(6, 4))
plt.plot(avg_day['Observed'], label='Observed')
plt.plot(avg_day['Forecast'], label='Forecast')
plt.title("Observed vs Forecasted Streamflow")
plt.xlabel("Day")
plt.ylabel("Streamflow")
plt.ylim(0, None)
plt.legend()
plt.grid()
plt.show()

#find rmse for each day forecast and observed values
rmse_df = pd.DataFrame(columns=['Day', 'RMSE'])
for day in list(range(1,28)):
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

#find nse for each day forecast and observed values
nse_df = pd.DataFrame(columns=['Day', 'NSE'])
for day in list(range(1,28)):
    day_df = forecast_df[forecast_df['day'] == day]
    denominator = np.sum((day_df['Observed'] - (np.mean(day_df['Observed'])))**2)
    numerator = np.sum((day_df['Observed'] - day_df['Forecast'])**2)
    nse = 1 - (numerator/denominator)
    nse_df = pd.concat([nse_df, pd.DataFrame({'Day': [day], 'NSE': [nse]})])
#plot nse for each day
plt.figure(figsize=(6, 4))
plt.plot(nse_df['Day'], nse_df['NSE'])
plt.title("NSE for different forecasting horizons")
plt.xlabel("Day")
plt.ylabel("NSE")
plt.ylim(0, None)
plt.show()


################################################################################################################################################################################################################################
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


# Make 15-day forecast
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

# visualize the actual vs forecasted values from the forecast_df
#find average of observed and forecasted values for each day
avg_day = final_forecast_df.groupby('day_hbv').mean()
plt.figure(figsize=(6, 4))
plt.plot(avg_day['Observed_hbv'], label='Observed')
plt.plot(avg_day['Final_Forecast'], label='HBV+XGBoost Forecast')
plt.title("Observed vs Forecasted Streamflow")
plt.xlabel("Day")
plt.ylabel("Streamflow")
plt.ylim(0, None)
plt.legend()
plt.grid()
plt.show()


#find rmse for each day forecast and observed values
rmse_df = pd.DataFrame(columns=['Day', 'RMSE'])
for day in list(range(1,29)):
    day_df = final_forecast_df[final_forecast_df['day_hbv'] == day]
    mse = mean_squared_error(day_df['Observed_hbv'], day_df['Final_Forecast'])
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