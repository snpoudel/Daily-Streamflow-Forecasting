import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#use seaborn colorblind color palette
sns.set_palette("colorblind")

#read list of station id
station_id = pd.read_csv('station_id.csv', dtype={'station_id': str})

############################################################################################################################################################################################################################################
#function to calculate NSE
def nse(observed, simulated):
    denominator = np.sum((observed - (np.mean(observed)))**2)
    numerator = np.sum((observed - simulated)**2)
    nse = 1 - (numerator/denominator)
    return nse

forecast_days = [1, 3, 5, 7, 15]

#sarimax model
df_sarimax = pd.DataFrame()
for id in station_id['station_id']:
    df = pd.read_csv(f'output/sarimax/sarimax{id}.csv')
    nse_values = {f'nse_day{day}': nse(df[df['day'] == day]['Observed'], df[df['day'] == day]['Forecast']) for day in forecast_days}
    df_temp = pd.DataFrame({'station_id': [id], **nse_values})
    df_sarimax = pd.concat([df_sarimax, df_temp]).reset_index(drop=True)

#XGBoost model
df_xgboost = pd.DataFrame()
for id in station_id['station_id']:
    df = pd.read_csv(f'output/xgboost/xgboost{id}.csv')
    nse_values = {f'nse_day{day}': nse(df[df['day'] == day]['Observed'], df[df['day'] == day]['Forecast']) for day in forecast_days}
    df_temp = pd.DataFrame({'station_id': [id], **nse_values})
    df_xgboost = pd.concat([df_xgboost, df_temp]).reset_index(drop=True)

#XGBoost lag model
df_xgboost_lag = pd.DataFrame()
for id in station_id['station_id']:
    df = pd.read_csv(f'output/xgboost_lag/xgboost_lag{id}.csv')
    nse_values = {f'nse_day{day}': nse(df[df['day'] == day]['Observed'], df[df['day'] == day]['Forecast']) for day in forecast_days}
    df_temp = pd.DataFrame({'station_id': [id], **nse_values})
    df_xgboost_lag = pd.concat([df_xgboost_lag, df_temp]).reset_index(drop=True)

#HBV model
df_hbv = pd.DataFrame()
for id in station_id['station_id']:
    df = pd.read_csv(f'output/hbv/hbv{id}.csv')
    df['day'] = pd.to_datetime(df['Date']).dt.day
    nse_values = {f'nse_day{day}': nse(df[df['day'] == day]['Observed'], df[df['day'] == day]['Forecast']) for day in forecast_days}
    df_temp = pd.DataFrame({'station_id': [id], **nse_values})
    df_hbv = pd.concat([df_hbv, df_temp]).reset_index(drop=True)

#HBV SARIMA model
df_hbv_sarima = pd.DataFrame()
for id in station_id['station_id']:
    df = pd.read_csv(f'output/hbv_sarima/hbv_sarima{id}.csv')
    df['day'] = pd.to_datetime(df['Date']).dt.day
    nse_values = {f'nse_day{day}': nse(df[df['day'] == day]['Observed_hbv'], df[df['day'] == day]['Final_Forecast']) for day in forecast_days}
    df_temp = pd.DataFrame({'station_id': [id], **nse_values})
    df_hbv_sarima = pd.concat([df_hbv_sarima, df_temp]).reset_index(drop=True)

#HBV XGBoost model
df_hbv_xgboost = pd.DataFrame()
for id in station_id['station_id']:
    df = pd.read_csv(f'output/hbv_xgboost/hbv_xgboost{id}.csv')
    df['day'] = pd.to_datetime(df['Date']).dt.day
    nse_values = {f'nse_day{day}': nse(df[df['day'] == day]['Observed_hbv'], df[df['day'] == day]['Final_Forecast']) for day in forecast_days}
    df_temp = pd.DataFrame({'station_id': [id], **nse_values})
    df_hbv_xgboost = pd.concat([df_hbv_xgboost, df_temp]).reset_index(drop=True)

#merge all the dataframes by indicating each dataframe by model name
df_sarimax['model'] = 'sarimax'
df_xgboost['model'] = 'xgboost'
df_xgboost_lag['model'] = 'xgboost_lag'
df_hbv['model'] = 'hbv'
df_hbv_sarima['model'] = 'hbv_sarima'
df_hbv_xgboost['model'] = 'hbv_xgboost'

# df_merged = pd.concat([df_sarimax, df_xgboost, df_xgboost_lag, df_hbv, df_hbv_sarima, df_hbv_xgboost], ignore_index=True)
df_merged = pd.concat([df_xgboost_lag, df_hbv, df_hbv_sarima, df_hbv_xgboost], ignore_index=True)
#model factor levels
df_merged['model'] = pd.Categorical(df_merged['model'], categories=['hbv_xgboost','hbv_sarima', 'xgboost_lag', 'hbv'], ordered=True)

#melt the dataframe
df_melted = df_merged.melt(
    id_vars=["station_id", "model"],
    value_vars=["nse_day1", "nse_day3", "nse_day5", "nse_day7", "nse_day15"],
    var_name="NSE_Day",
    value_name="NSE_Value"
)

# Create a boxplot
plt.figure(figsize=(6, 4))
sns.boxplot(data=df_melted, x="NSE_Day", y="NSE_Value", hue="model",
            showfliers=False, width=0.6)
# Customize plot
plt.title("Forcast accuracy at different forecasting horizons")
plt.xlabel("Forecast Lead Time")
#change x axis labels
plt.xticks(ticks=[0, 1, 2, 3, 4], labels=['Day 1', 'Day 3', 'Day 5', 'Day 7', 'Day 15'])
plt.ylabel("NSE Value")
plt.legend(title='Model', loc='best')
# plt.legend(title="Model", bbox_to_anchor=(1.05, 1), loc='upper left')
# Show the plot
plt.ylim(0,None)
plt.grid(linestyle='--', alpha=0.8)
plt.tight_layout()
plt.show()
plt.savefig('output/short_forecast_NSEaccuracy.png', dpi=300)


############################################################################################################################################################################################################################################
#function to calculate rmse
def rmse(observed, simulated):
    rmse = np.sqrt(np.mean((observed - simulated)**2))
    return rmse

forecast_days = [1, 3, 5, 7, 15]

#sarimax model
df_sarimax = pd.DataFrame()
for id in station_id['station_id']:
    df = pd.read_csv(f'output/sarimax/sarimax{id}.csv')
    nse_values = {f'nse_day{day}': rmse(df[df['day'] == day]['Observed'], df[df['day'] == day]['Forecast']) for day in forecast_days}
    df_temp = pd.DataFrame({'station_id': [id], **nse_values})
    df_sarimax = pd.concat([df_sarimax, df_temp]).reset_index(drop=True)

#XGBoost model
df_xgboost = pd.DataFrame()
for id in station_id['station_id']:
    df = pd.read_csv(f'output/xgboost/xgboost{id}.csv')
    nse_values = {f'nse_day{day}': rmse(df[df['day'] == day]['Observed'], df[df['day'] == day]['Forecast']) for day in forecast_days}
    df_temp = pd.DataFrame({'station_id': [id], **nse_values})
    df_xgboost = pd.concat([df_xgboost, df_temp]).reset_index(drop=True)

#XGBoost lag model
df_xgboost_lag = pd.DataFrame()
for id in station_id['station_id']:
    df = pd.read_csv(f'output/xgboost_lag/xgboost_lag{id}.csv')
    nse_values = {f'nse_day{day}': rmse(df[df['day'] == day]['Observed'], df[df['day'] == day]['Forecast']) for day in forecast_days}
    df_temp = pd.DataFrame({'station_id': [id], **nse_values})
    df_xgboost_lag = pd.concat([df_xgboost_lag, df_temp]).reset_index(drop=True)

#HBV model
df_hbv = pd.DataFrame()
for id in station_id['station_id']:
    df = pd.read_csv(f'output/hbv/hbv{id}.csv')
    df['day'] = pd.to_datetime(df['Date']).dt.day
    nse_values = {f'nse_day{day}': rmse(df[df['day'] == day]['Observed'], df[df['day'] == day]['Forecast']) for day in forecast_days}
    df_temp = pd.DataFrame({'station_id': [id], **nse_values})
    df_hbv = pd.concat([df_hbv, df_temp]).reset_index(drop=True)

#HBV SARIMA model
df_hbv_sarima = pd.DataFrame()
for id in station_id['station_id']:
    df = pd.read_csv(f'output/hbv_sarima/hbv_sarima{id}.csv')
    df['day'] = pd.to_datetime(df['Date']).dt.day
    nse_values = {f'nse_day{day}': rmse(df[df['day'] == day]['Observed_hbv'], df[df['day'] == day]['Final_Forecast']) for day in forecast_days}
    df_temp = pd.DataFrame({'station_id': [id], **nse_values})
    df_hbv_sarima = pd.concat([df_hbv_sarima, df_temp]).reset_index(drop=True)

#HBV XGBoost model
df_hbv_xgboost = pd.DataFrame()
for id in station_id['station_id']:
    df = pd.read_csv(f'output/hbv_xgboost/hbv_xgboost{id}.csv')
    df['day'] = pd.to_datetime(df['Date']).dt.day
    nse_values = {f'nse_day{day}': rmse(df[df['day'] == day]['Observed_hbv'], df[df['day'] == day]['Final_Forecast']) for day in forecast_days}
    df_temp = pd.DataFrame({'station_id': [id], **nse_values})
    df_hbv_xgboost = pd.concat([df_hbv_xgboost, df_temp]).reset_index(drop=True)

#merge all the dataframes by indicating each dataframe by model name
df_sarimax['model'] = 'sarimax'
df_xgboost['model'] = 'xgboost'
df_xgboost_lag['model'] = 'xgboost_lag'
df_hbv['model'] = 'hbv'
df_hbv_sarima['model'] = 'hbv_sarima'
df_hbv_xgboost['model'] = 'hbv_xgboost'

# df_merged = pd.concat([df_sarimax, df_xgboost, df_xgboost_lag, df_hbv, df_hbv_sarima, df_hbv_xgboost], ignore_index=True)
df_merged = pd.concat([df_xgboost_lag, df_hbv, df_hbv_sarima, df_hbv_xgboost], ignore_index=True)
#model factor levels
df_merged['model'] = pd.Categorical(df_merged['model'], categories=['hbv_xgboost','hbv_sarima', 'xgboost_lag', 'hbv'], ordered=True)

#melt the dataframe
df_melted = df_merged.melt(
    id_vars=["station_id", "model"],
    value_vars=["nse_day1", "nse_day3", "nse_day5", "nse_day7", "nse_day15"],
    var_name="NSE_Day",
    value_name="NSE_Value"
)

# Create a boxplot
plt.figure(figsize=(6, 4))
sns.boxplot(data=df_melted, x="NSE_Day", y="NSE_Value", hue="model",
            showfliers=False, width=0.6)
# Customize plot
plt.title("Forcast accuracy at different forecasting horizons")
plt.xlabel("Forecast Lead Time")
#change x axis labels
plt.xticks(ticks=[0, 1, 2, 3, 4], labels=['Day 1', 'Day 3', 'Day 5', 'Day 7', 'Day 15'])
plt.ylabel("RMSE Value")
plt.legend(title='Model', loc='best')
# plt.legend(title="Model", bbox_to_anchor=(1.05, 1), loc='upper left')
# Show the plot
plt.ylim(0,None)
plt.grid(linestyle='--', alpha=0.8)
plt.tight_layout()
plt.show()
plt.savefig('output/short_forecast_RMSEaccuracy.png', dpi=300)
