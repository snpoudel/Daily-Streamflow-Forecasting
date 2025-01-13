import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_palette("colorblind")

models = ['sarimax', 'xgboost', 'xgboost_lag', 'hbv', 'hbv_sarima', 'hbv_xgboost']

# model = 'sarimax'
time_df = pd.DataFrame()
for model in models:
    df = pd.read_csv(f'output/time_taken/{model}01096000.csv')
    time_seconds = df['time_taken']
    #time looks like 0 days 00:11:06.475292, extract total time in seconds
    time_seconds = time_seconds.apply(pd.to_timedelta).apply(lambda x: x.total_seconds())
    #make a dataframe and concatenate
    temp_df = pd.DataFrame({'model': model, 'time_seconds': time_seconds})
    time_df = pd.concat([time_df, temp_df])

num_cpu_cores = 6
# Adjust time taken for each model, all model except hbv is run on 6 cpu cores
time_df['time_seconds'] *= num_cpu_cores
time_df.loc[time_df['model'] == 'hbv', 'time_seconds'] /= num_cpu_cores

#time taken by hbv_xgboost is sum of time taken by hbv and xgboost
time_df.loc[time_df['model'] == 'hbv_xgboost', 'time_seconds'] = time_df[time_df['model'] == 'hbv']['time_seconds'].values[0] + time_df[time_df['model'] == 'hbv_xgboost']['time_seconds'].values[0]
#time taken by hbv_sarima is sum of time taken by hbv and sarimax
time_df.loc[time_df['model'] == 'hbv_sarima', 'time_seconds'] = time_df[time_df['model'] == 'hbv']['time_seconds'].values[0] + time_df[time_df['model'] == 'hbv_sarima']['time_seconds'].values[0]
#order in descending order
time_df = time_df.sort_values(by='time_seconds', ascending=False)

#only keep xgboost_lag, hbv_xgboost and hbv_sarima, hbv
time_df = time_df[time_df['model'].isin(['xgboost_lag', 'hbv_xgboost', 'hbv_sarima', 'hbv'])]

#make a bar plot with model on x-axis and time taken on y-axis
plt.figure(figsize=(6, 3))
sns.barplot(x='model', y='time_seconds', data=time_df, width=0.8, hue='model')
# plt.title("Time Taken by Each Model")
plt.xlabel("Models")
plt.xticks(labels=['HBV+XGBoost', 'HBV+ARIMA', 'HBV', 'XGBoost'], ticks=[0, 1, 2, 3])
# plt.xticks(rotation=45)
plt.ylabel("Time Taken (seconds)")
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('figures/time_taken.png', dpi=300)
plt.show()

