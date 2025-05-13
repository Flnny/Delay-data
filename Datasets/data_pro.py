import pandas as pd
import yaml
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler

file_name = 'filtered_flight_data.csv'

df = pd.read_csv(file_name)

def remove_outliers_percentile(df, column, lower=0.01, upper=0.99):
    lower_bound = df[column].quantile(lower)
    upper_bound = df[column].quantile(upper)
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# 使用示例
df = remove_outliers_percentile(df, 'ARR_DELAY')

df['FL_DATE'] = pd.to_datetime(df['FL_DATE'])
df['FL_YEAR'] = df['FL_DATE'].dt.year
df.rename(columns={'MONTH': 'FL_MONTH', 'DAY_OF_MONTH': 'FL_DAY', 'DAY_OF_WEEK': 'FL_WEEK'}, inplace=True)

df.drop(columns=['FL_DATE'], inplace=True)

time_columns = ['CRS_DEP_TIME', 'DEP_TIME', 'WHEELS_OFF', 'WHEELS_ON', 'CRS_ARR_TIME', 'ARR_TIME']
for col in time_columns:
    df[col] = pd.to_datetime(df[col]).dt.strftime('%H:%M:%S')
    df[col + '_MIN'] = pd.to_datetime(df[col]).dt.hour * 60 + pd.to_datetime(df[col]).dt.minute

df.drop(columns=time_columns, inplace=True)

encoder_columns = ['OP_CARRIER', 'ORIGIN', 'DEST']

encoder = LabelEncoder()
for col in encoder_columns:
    df[col] = encoder.fit_transform(df[col])

categorical_columns = ['OP_CARRIER', 'OP_CARRIER_FL_NUM',
                       'FL_YEAR', 'FL_MONTH', 'FL_DAY', 'FL_WEEK',
                       'ORIGIN_INDEX', 'DEST_INDEX']

continuous_columns = ['CRS_DEP_TIME_MIN', 'CRS_ARR_TIME_MIN', 'CRS_ELAPSED_TIME',
                      'FLIGHTS',
                      'O_TEMP', 'O_PRCP', 'O_WSPD', 'D_TEMP', 'D_PRCP', 'D_WSPD',
                      'O_LATITUDE', 'O_LONGITUDE', 'D_LATITUDE', 'D_LONGITUDE']

target = ['DEP_DELAY', 'ARR_DELAY']

scaler = MinMaxScaler()
df[continuous_columns] = scaler.fit_transform(df[continuous_columns])

df = df[target + categorical_columns + continuous_columns]

df.to_csv('arr_delay_data.csv', index=False)

data_info = {
    "columns_info": {
        'Target': target,
        "Categorical Features": categorical_columns,
        "Continuous Features": continuous_columns
    },
    "data_summary": {
        "info": df.info(),
        "memory_usage": df.memory_usage(deep=True).to_dict()
    }
}

# 将数据集详细信息保存到YAML文件
with open('arr_delay_data_info.yaml', 'w') as yaml_file:
    yaml.dump(data_info, yaml_file, default_flow_style=False)