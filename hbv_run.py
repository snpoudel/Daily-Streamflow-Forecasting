#import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from hbv_model import hbv #imported from local python script
from hbv_forecast_model import hbv_forecast #imported from local python script
from geneticalgorithm import geneticalgorithm as ga # install package first

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
    'max_num_iteration': 50,              # Generations, higher is better, but requires more computational time
    'max_iteration_without_improv': None,   # Stopping criterion for lack of improvement
    'population_size': 200,                 #1500 Number of parameter-sets in a single iteration/generation(to start with population 10 times the number of parameters should be fine!)
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

#evaluate model performance using the best parameters for test data
df = pd.read_csv(f"data/hbv_input_{station_id}.csv")
test = df[df["year"] >= 2009]
q_obs = test["qobs"]
q_sim, _, _ = hbv(param_value, test["precip"], test["tavg"], test["date"], test["latitude"], routing)
#calculate rmse, nse
rmse = mean_squared_error(q_obs, q_sim) ** 0.5
nse = 1 - (np.sum((q_obs - np.mean(q_obs))**2) / np.sum((q_obs - q_sim)**2))
print(f"Test RMSE: {rmse} and NSE: {nse}")



################################################################################################################################################################################################################################
##--FORECAST--##
df = pd.read_csv(f"data/hbv_input_{'01094400'}.csv")
#use data after 2009 for forecasting
test = df[df["year"] >= 2006]

forecast_df = pd.DataFrame(columns=['Date', 'Observed', 'Forecast'])
for test_year in range(2009, 2016):
    test_df = test[test['year'] == test_year]
    for month in range(1, 12):
        test_monthly = test_df[test_df['month'] == month].reset_index(drop=True)
        test_monthly_next = test_df[test_df['month'] == month+1].reset_index(drop=True)
        calibrated_params = best_parameters["variable"]
        # #re-run the model with the calibrated parameters for each month and get storage and sim_vars for last step
        q_sim, storage, sim_vars = hbv(calibrated_params, test_monthly["precip"], test_monthly["tavg"], test_monthly["date"], test_monthly["latitude"], routing)
        #forecast for 28 days
        last_storage = storage #last storage value
        last_sim_vars = sim_vars #last sim_vars value
        for day in range(1, 29):
            test_day = test_monthly[test_monthly['day'] == day]
            test_day_next = test_monthly_next[test_monthly_next['day'] == day]
            forecast_streamflow, forecast_storage, forecast_sim_vars = hbv_forecast(calibrated_params, last_storage, last_sim_vars, [test_day["precip"].values[0]], test_day["tavg"].values[0], test_day['date'] ,test_day["latitude"].values[0], routing)
            forecast_df = pd.concat([forecast_df, pd.DataFrame({'Date': test_day_next['date'], 'Observed': test_day_next['qobs'], 'Forecast': forecast_streamflow})])
            #update the last values
            last_storage = forecast_storage
            last_sim_vars = forecast_sim_vars
#save the forecasted values


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
