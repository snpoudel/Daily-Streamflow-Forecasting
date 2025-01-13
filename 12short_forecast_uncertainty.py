import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_palette('colorblind')
station_id = pd.read_csv('station_id.csv', dtype=str)

############################################################################################################################################################################################################################################
# #function to calculate NSE
# def nse(observed, simulated):
#     denominator = np.sum((observed - (np.mean(observed)))**2)
#     numerator = np.sum((observed - simulated)**2)
#     nse = 1 - (numerator/denominator)
#     return nse

# #function to calculate rmse
# def nse(observed, simulated):
#     rmse = np.sqrt(np.mean((observed - simulated)**2))
#     return rmse

#function to calculate pbias
def nse(observed, simulated):
    pbias = 100 * np.sum(simulated - observed) / np.sum(observed)
    return pbias

forecast_days = [1, 3, 5, 7, 15, 28]
error = ['rmse0', 'rmse1', 'rmse2', 'rmse3', 'rmse4']

#Xgboost_lag model
df_xgboost_lag = pd.DataFrame()
for id in station_id['station_id']:
    for error_val in error:
        df = pd.read_csv(f'output/xgboost_lag/puq/xgboost_lag{id}.csv')
        df = df[df['error'] == error_val]
        nse_values = {f'nse_day{day}': nse(df[df['day'] == day]['Observed'], df[df['day'] == day]['Forecast']) for day in forecast_days}
        df_temp = pd.DataFrame({'station_id': [id], **nse_values})
        df_temp['error'] = error_val
        df_xgboost_lag = pd.concat([df_xgboost_lag, df_temp]).reset_index(drop=True)

#Hbv model
df_hbv = pd.DataFrame()
for id in station_id['station_id']:
    for error_val in error:
        df = pd.read_csv(f'output/hbv/puq/hbv{id}.csv')
        df = df[df['error'] == error_val]
        df['Date'] = pd.to_datetime(df['Date'])
        df['day'] = df['Date'].dt.day
        nse_values = {f'nse_day{day}': nse(df[df['day'] == day]['Observed'], df[df['day'] == day]['Forecast']) for day in forecast_days}
        df_temp = pd.DataFrame({'station_id': [id], **nse_values})
        df_temp['error'] = error_val
        df_hbv = pd.concat([df_hbv, df_temp]).reset_index(drop=True)

#Hbv_sarima model
df_hbv_sarima = pd.DataFrame()
for id in station_id['station_id']:
    for error_val in error:
        df = pd.read_csv(f'output/hbv_sarima/puq/hbv_sarima{id}.csv')
        df = df[df['error'] == error_val]
        df['Date'] = pd.to_datetime(df['Date'])
        df['day'] = df['Date'].dt.day
        nse_values = {f'nse_day{day}': nse(df[df['day'] == day]['Observed'], df[df['day'] == day]['Forecast']) for day in forecast_days}
        df_temp = pd.DataFrame({'station_id': [id], **nse_values})
        df_temp['error'] = error_val
        df_hbv_sarima = pd.concat([df_hbv_sarima, df_temp]).reset_index(drop=True)

#Hbv_xgboost model
df_hbv_xgboost = pd.DataFrame()
for id in station_id['station_id']:
    for error_val in error:
        df = pd.read_csv(f'output/hbv_xgboost/puq/hbv_xgboost{id}.csv')
        df = df[df['error'] == error_val]
        df['date'] = pd.to_datetime(df['date'])
        df['day'] = df['date'].dt.day
        nse_values = {f'nse_day{day}': nse(df[df['day'] == day]['hbv_observed'], df[df['day'] == day]['total_forecast']) for day in forecast_days}
        df_temp = pd.DataFrame({'station_id': [id], **nse_values})
        df_temp['error'] = error_val
        df_hbv_xgboost = pd.concat([df_hbv_xgboost, df_temp]).reset_index(drop=True)

#merge all the dataframes
df_xgboost_lag['model'] = 'xgboost_lag'
df_hbv['model'] = 'hbv'
df_hbv_sarima['model'] = 'hbv_sarima'
df_hbv_xgboost['model'] = 'hbv_xgboost'
df_all = pd.concat([df_xgboost_lag, df_hbv, df_hbv_sarima, df_hbv_xgboost]).reset_index(drop=True)
df_all['model'] = pd.Categorical(df_all['model'], categories=[ 'hbv', 'hbv_sarima', 'xgboost_lag', 'hbv_xgboost'], ordered=True)

# create boxplot
fig, axes = plt.subplots(4, 1, figsize=(9, 9), sharex=True)
# Plot for nse_day1
sns.boxplot(ax=axes[0], x='error', y='nse_day1', data=df_all, hue='model', showfliers=False, width=0.6)
axes[0].set_title('a) 1-day forecast accuracy at different error levels')
axes[0].set_ylabel('Bias (%)')
axes[0].get_legend().remove()
axes[0].grid(axis='y', linestyle='--', alpha=0.6)
axes[0].axhline(0, color='red', linestyle='--')
# axes[0].set_ylim(0, None)
# Plot for nse_day3
sns.boxplot(ax=axes[1], x='error', y='nse_day5', data=df_all, hue='model', showfliers=False, width=0.6)
axes[1].set_title('b) 5-day forecast accuracy at different error levels')
axes[1].set_ylabel('Bias (%)')
axes[1].get_legend().remove()
axes[1].grid(axis='y', linestyle='--', alpha=0.6)
axes[1].axhline(0, color='red', linestyle='--')
# axes[1].set_ylim(0, 1)
# Plot for nse_day5
sns.boxplot(ax=axes[2], x='error', y='nse_day15', data=df_all, hue='model', showfliers=False, width=0.6)
axes[2].set_title('c) 15-day forecast accuracy at different error levels')
axes[2].set_ylabel('Bias (%)')
axes[2].get_legend().remove()
axes[2].set_xlabel('Error in forecasted precipitation (mm/day)')
axes[2].grid(axis='y', linestyle='--', alpha=0.6)
axes[2].axhline(0, color='red', linestyle='--')
# axes[2].set_ylim(0, 1)
# Plot for nse_day7
sns.boxplot(ax=axes[3], x='error', y='nse_day28', data=df_all, hue='model', showfliers=False, width=0.6)
axes[3].set_title('d) 28-day forecast accuracy at different error levels')
axes[3].set_ylabel('Bias (%)')
axes[3].legend(title='Model', loc='lower center',  ncol=4, bbox_to_anchor=(0.5, -0.85))
axes[3].set_xlabel('Error in forecasted precipitation (mm/day)')
axes[3].grid(axis='y', linestyle='--', alpha=0.6)
axes[3].axhline(0, color='red', linestyle='--')
# axes[3].set_ylim(0, 1)

plt.tight_layout()
plt.savefig('figures/3short_forecast_uncertainty.png', dpi=300)
plt.show()