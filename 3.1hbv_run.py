#import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from hbv_model import hbv #imported from local python script
from synthetic_precip import synthetic_precip #takes original precip and returns synthetic precip with rmse 1, 2, 3, & 4 mm/day
from geneticalgorithm import geneticalgorithm as ga # install package first
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
# ################################################################################################################################################################################################################################
# ##--CALIBRATION--##
# df = pd.read_csv(f"data/hbv_input_{station_id}.csv")
# #only used data upto 2005 for calibration
# df = df[df["year"] < 2009]
# #hbv model input
# p = df["precip"]
# temp = df["tavg"]
# date = df["date"]
# latitude = df["latitude"]
# routing = 1 # 0: no routing, 1 allows running
# q_obs = df["qobs"] #validation data / observed flow

# ##genetic algorithm for hbv model calibration
# #reference: https://github.com/rmsolgi/geneticalgorithm
# #write a function you want to minimize
# def nse(pars):
#     q_sim,_ ,_ = hbv(pars, p, temp, date, latitude, routing)
#     denominator = np.sum((q_obs - (np.mean(q_obs)))**2)
#     numerator = np.sum((q_obs - q_sim)**2)
#     nse_value = 1 - (numerator/denominator)
#     return -nse_value #minimize this (use negative sign if you need to maximize)

# varbound = np.array([[1,1000], #fc #lower and upper bounds of the parameters
#                     [1,7], #beta
#                     [0.01,0.99], #pwp
#                     [1,999], #l
#                     [0.01,0.99], #ks
#                     [0.01,0.99], #ki
#                     [0.0001, 0.99], #kb
#                     [0.001, 0.99], #kperc
#                     [0.5,2], #coeff_pet
#                     [0.01,10], #ddf
#                     [0.5,1.5], #scf
#                     [-1,4], #ts
#                     [-1,4], #tm
#                     [-1,4], #tti
#                     [0, 0.2], #whc
#                     [0.1,1], #crf
#                     [1,10]]) #maxbas

# algorithm_param = {
#     'max_num_iteration': 100,              # Generations, higher is better, but requires more computational time
#     'max_iteration_without_improv': None,   # Stopping criterion for lack of improvement
#     'population_size': 500,                 #1500 Number of parameter-sets in a single iteration/generation(to start with population 10 times the number of parameters should be fine!)
#     'parents_portion': 0.3,                 # Portion of new generation population filled by previous population
#     'elit_ratio': 0.01,                     # Portion of the best individuals preserved unchanged
#     'crossover_probability': 0.3,           # Chance of existing solution passing its characteristics to new trial solution
#     'crossover_type': 'uniform',            # Create offspring by combining the parameters of selected parents
#     'mutation_probability': 0.01            # Introduce random changes to the offspring’s parameters (0.1 is 10%)
# }

# model = ga(function = nse,
#         dimension = 17, #number of parameters to be calibrated
#         variable_type= 'real',
#         variable_boundaries = varbound,
#         algorithm_parameters = algorithm_param)

# model.run()
# #end of genetic algorithm

# #output of the genetic algorithm/best parameters
# best_parameters = model.output_dict
# param_value = best_parameters["variable"]
# nse_value = best_parameters["function"]
# nse_value = -nse_value #nse function gives -ve values, which is now reversed here to get true nse
# #convert into a dataframe
# df_param = pd.DataFrame(param_value).transpose()
# df_param = df_param.rename(columns={0:"fc", 1:"beta", 2:"pwp", 3:"l", 4:"ks", 5:"ki",
#                             6:"kb", 7:"kperc",  8:"coeff_pet", 9:"ddf", 10:"scf", 11:"ts",
#                             12:"tm", 13:"tti", 14:"whc", 15:"crf", 16:"maxbas"})
# df_param["station_id"] = str(station_id)
# df_nse = pd.DataFrame([nse_value], columns=["train_nse"])
# df_nse["station_id"] = str(station_id)
# #save as a csv file
# # df_param.to_csv(f"output/param_{station_id}.csv", index = False)
# # df_nse.to_csv(f"output/nse_{station_id}.csv", index = False)

# #run the model with the best parameters for the calibration data
# q_sim_cal, storage_cal, sim_vars_cal = hbv(param_value, p, temp, date, latitude, routing)

# #evaluate model performance using the best parameters for test data
# df = pd.read_csv(f"data/hbv_input_{station_id}.csv")
# test = df[df["year"] >= 2009].reset_index(drop=True)
# test['date'] = pd.to_datetime(test['date'])
# q_obs = test["qobs"]
# q_sim_test, _, _ = hbv(param_value, test["precip"], test["tavg"], test["date"], test["latitude"], routing)
# #calculate rmse, nse
# rmse = mean_squared_error(q_obs, q_sim_test) ** 0.5
# denominator = np.sum((q_obs - (np.mean(q_obs)))**2)
# numerator = np.sum((q_obs - q_sim_test)**2)
# nse_value = 1 - (numerator/denominator)
# df_nse['test_nse'] = nse_value
# print(f"Test RMSE: {rmse:.2f} and NSE: {nse_value:.2f}")


# ################################################################################################################################################################################################################################
# ##--FORECAST--##
# df = pd.read_csv(f"data/hbv_input_{station_id}.csv")
# df['date'] = pd.to_datetime(df['date'])  # Convert 'date' column to datetime
# #use data after 2009 for forecasting
# test = df[df["year"] >= 2009]

# forecast_df = pd.DataFrame(columns=['Date', 'Observed', 'Forecast'])
# for test_year in range(2009, 2016):
#     test_df = test[test['year'] == test_year]
#     for month in range(1, 13):
#         #get model states by running model for the last 365 days before the forecast period starts
#         forecast_start_date = pd.Timestamp(f"{test_year}-{month}-01")
#         forecast_end_date = forecast_start_date + pd.DateOffset(days=28)
#         model_train_start_date = forecast_start_date - pd.DateOffset(years=1)
#         df_states = df[(df['date'] >= model_train_start_date) & (df['date'] < forecast_end_date)]
#         df_states = df_states.reset_index(drop=True)
#         #run model
#         q_sim,_ ,_ = hbv(param_value, df_states["precip"], df_states["tavg"], df_states["date"], df_states["latitude"], routing)
#         #extract the last 28 days of the streamflow simulation
#         forecast_streamflow = q_sim[-28:]
#         forecast_date = df_states['date'][-28:]
#         observed_streamflow = df_states['qobs'][-28:]
#         forecast_df = pd.concat([forecast_df, pd.DataFrame({'Date': forecast_date, 'Observed': observed_streamflow, 'Forecast': forecast_streamflow})])
# #save the forecasted values
# forecast_df = forecast_df.reset_index(drop=True)

# # #save parameters
# # df_param.to_csv(f"output/hbv/parameters/param{station_id}.csv", index = False)
# # #save train and test nse values
# # df_nse.to_csv(f"output/hbv/metrics/metrics{station_id}.csv", index = False)
# # #save forecasted values
# # forecast_df.to_csv(f"output/hbv/hbv{station_id}.csv", index = False)

# #total time taken, save as csv
# if station_id == '01096000':
#     end_time = pd.Timestamp.now()
#     time_taken = end_time - start_time
#     time_taken_df = pd.DataFrame({'time_taken': [time_taken]})
#     time_taken_df.to_csv(f'output/time_taken/hbv{station_id}.csv', index=False)
# print('Completed!!!')



################################################################################################################################################################################################################################
##--FORECAST with precipitation uncertainty--##
df = pd.read_csv(f"data/hbv_input_{station_id}.csv")
df['date'] = pd.to_datetime(df['date'])  # Convert 'date' column to datetime

param_value = pd.read_csv(f'output/hbv/parameters/param{station_id}.csv')
param_value = param_value.drop(columns=['station_id'])
param_value = param_value.values[0]

#use data after 2009 for forecasting
test_uq = df[df["year"] >= 2009]

#run for 10 iterations of synthetic precip with uncertainty
forecast_df = pd.DataFrame(columns=['Date', 'Observed', 'Forecast', 'error', 'iter'])
for iter in range(10):
    precip1, precip2, precip3, precip4 = synthetic_precip(test_uq['precip'].values) #synthetic precip with rmse 1, 2, 3, 4 mm/day
    precip_uq = [test_uq['precip'].values, precip1.values, precip2.values, precip3.values, precip4.values]
    error_level = ['rmse0', 'rmse1', 'rmse2', 'rmse3', 'rmse4']

    for index, puq in enumerate(precip_uq):
        df = pd.read_csv(f"data/hbv_input_{station_id}.csv")
        df['date'] = pd.to_datetime(df['date'])
        #replace original precip after 2009 with synthetic precip with uncertainty
        df.loc[(df['year'] >= 2009), 'precip'] = puq
        for test_year in range(2009, 2016):
            for month in range(1, 13):
                #get model states by running model for the last 365 days before the forecast period starts
                forecast_start_date = pd.Timestamp(f"{test_year}-{month}-01")
                forecast_end_date = forecast_start_date + pd.DateOffset(days=28)
                model_train_start_date = forecast_start_date - pd.DateOffset(years=1)
                df_states = df[(df['date'] >= model_train_start_date) & (df['date'] < forecast_end_date)]
                df_states = df_states.reset_index(drop=True)
                #run model
                q_sim,_ ,_ = hbv(param_value, df_states["precip"], df_states["tavg"], df_states["date"], df_states["latitude"], routing=1)
                #extract the last 28 days of the streamflow simulation
                forecast_streamflow = q_sim[-28:]
                forecast_date = df_states['date'][-28:]
                observed_streamflow = df_states['qobs'][-28:]
                forecast_df = pd.concat([forecast_df, pd.DataFrame({'Date': forecast_date, 'Observed': observed_streamflow, 'Forecast': forecast_streamflow, 'error': error_level[index], 'iter': iter})])
#save the forecasted values
forecast_df = forecast_df.reset_index(drop=True)

#save forecasted values
forecast_df.to_csv(f"output/hbv/puq/hbv{station_id}.csv", index = False)
