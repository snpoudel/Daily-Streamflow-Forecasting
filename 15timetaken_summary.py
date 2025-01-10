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
#order in descending order
time_df = time_df.sort_values(by='time_seconds', ascending=False)

#make a bar plot with model on x-axis and time taken on y-axis
plt.figure(figsize=(6, 4))
sns.barplot(x='model', y='time_seconds', data=time_df, palette='colorblind')
plt.title("Time Taken by Each Model")
plt.xlabel("Models")
plt.xticks(rotation=45)
plt.ylabel("Time Taken (seconds)")
plt.tight_layout()
plt.show()
