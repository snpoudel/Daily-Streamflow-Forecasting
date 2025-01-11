import numpy as np
import pandas as pd
station_id = pd.read_csv('station_id.csv', dtype={'station_id': str})
# id = '01173500'

sarimax_params = pd.DataFrame()
for id in station_id['station_id']:
    df = pd.read_csv(f'output/sarimax/parameters/params{id}.csv')
    order = df['order'][0]
    order1, order2, order3 = order[1:-1].split(', ')
    sorder = df['seasonal_order'][0]
    sorder1, sorder2, sorder3, sorder4 = sorder[1:-1].split(', ')

    temp_df = pd.DataFrame({'station_id': [id], 'order1': [order1], 'order2': [order2], 'order3': [order3], 'sorder1': [sorder1], 'sorder2': [sorder2], 'sorder3': [sorder3], 'sorder4': [sorder4]})
    sarimax_params = pd.concat([sarimax_params, temp_df])

#find min and max values of order and seasonal order
max_order1 = sarimax_params['order1'].max()
max_order2 = sarimax_params['order2'].max()
max_order3 = sarimax_params['order3'].max()
max_sorder1 = sarimax_params['sorder1'].max()
max_sorder2 = sarimax_params['sorder2'].max()
max_sorder3 = sarimax_params['sorder3'].max()
max_sorder4 = sarimax_params['sorder4'].max()

print(f'order: {max_order1}, {max_order2}, {max_order3}')
print(f'seasonal order: {max_sorder1}, {max_sorder2}, {max_sorder3}, {max_sorder4}')