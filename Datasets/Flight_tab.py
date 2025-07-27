import os
import pandas as pd
import numpy as np
import yaml
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

def remove_outliers_percentile(df, column, lower=0.01, upper=0.99):
    lower_bound = df[column].quantile(lower)
    upper_bound = df[column].quantile(upper)
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def process_file(file_path):
    df = pd.read_csv(file_path, low_memory=False)

    if 'ARR_DELAY' not in df.columns:
        print(f"Skipping {file_path}: Missing ARR_DELAY column.")
        return None

    df = remove_outliers_percentile(df, 'ARR_DELAY')

    df['FL_DATE'] = pd.to_datetime(df['FL_DATE'], errors='coerce')
    df = df.dropna(subset=['FL_DATE'])

    df['FL_YEAR'] = df['FL_DATE'].dt.year
    df.rename(columns={'MONTH': 'FL_MONTH', 'DAY_OF_MONTH': 'FL_DAY', 'DAY_OF_WEEK': 'FL_WEEK'}, inplace=True)
    df.drop(columns=['FL_DATE'], inplace=True)

    time_columns = ['CRS_DEP_TIME', 'DEP_TIME', 'WHEELS_OFF', 'WHEELS_ON', 'CRS_ARR_TIME', 'ARR_TIME']
    for col in time_columns:
        df[col] = pd.to_datetime(df[col], format='%H%M', errors='coerce')
        df[col + '_MIN'] = df[col].dt.hour * 60 + df[col].dt.minute
    df.drop(columns=time_columns, inplace=True)

    df['ORIGIN_INDEX'] = df['ORIGIN']
    df['DEST_INDEX'] = df['DEST']

    categorical_columns = ['OP_CARRIER', 'OP_CARRIER_FL_NUM',
                           'FL_YEAR', 'FL_MONTH', 'FL_DAY', 'FL_WEEK',
                           'ORIGIN_INDEX', 'DEST_INDEX']

    continuous_columns = ['CRS_DEP_TIME_MIN', 'CRS_ARR_TIME_MIN', 'CRS_ELAPSED_TIME',
                          'FLIGHTS',
                          'O_TEMP', 'O_PRCP', 'O_WSPD', 'D_TEMP', 'D_PRCP', 'D_WSPD',
                          'O_LATITUDE', 'O_LONGITUDE', 'D_LATITUDE', 'D_LONGITUDE']

    target = ['DEP_DELAY', 'ARR_DELAY']

    required_columns = continuous_columns + target + categorical_columns

    df = df[target + categorical_columns + continuous_columns]

    return df

def save_year_data(df, year, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    output_csv = os.path.join(output_folder, f"Flight_tab_{year}.csv")
    yaml_file = os.path.join(output_folder, f"data_info_{year}.yaml")

    df.to_csv(output_csv, index=False)
    print(f"Data for {year} saved: {output_csv}")

    # Save field info
    data_info = {
        "columns_info": {
            'Target': ['DEP_DELAY', 'ARR_DELAY'],
            "Categorical Features": ['OP_CARRIER', 'OP_CARRIER_FL_NUM',
                                     'FL_YEAR', 'FL_MONTH', 'FL_DAY', 'FL_WEEK',
                                     'ORIGIN_INDEX', 'DEST_INDEX'],
            "Continuous Features": ['CRS_DEP_TIME_MIN', 'CRS_ARR_TIME_MIN', 'CRS_ELAPSED_TIME',
                                    'FLIGHTS',
                                    'O_TEMP', 'O_PRCP', 'O_WSPD', 'D_TEMP', 'D_PRCP', 'D_WSPD',
                                    'O_LATITUDE', 'O_LONGITUDE', 'D_LATITUDE', 'D_LONGITUDE']
        },
        "data_summary": {
            "shape": df.shape,
            "memory_usage_MB": round(df.memory_usage(deep=True).sum() / 1024**2, 2)
        }
    }

    with open(yaml_file, 'w') as f:
        yaml.dump(data_info, f, default_flow_style=False)
    print(f"Data description for {year} saved: {yaml_file}")

def main():
    input_folder = "D:\project\Delay_data\Datasets\data"
    output_folder = os.path.join(input_folder, "Tab")
    # year_files = [f"flight_with_weather_{year}.csv" for year in range(2020, 2020)]
    year_files = ["flight_with_weather_2020.csv"]
    for file_name in year_files:
        year = file_name.split("_")[-1].split(".")[0]  # Extract year
        file_path = os.path.join(input_folder, file_name)
        if not os.path.isfile(file_path):
            print(f"File does not exist: {file_path}, skipping.")
            continue

        print(f"Processing data for {year} ...")
        df = process_file(file_path)
        if df is not None:
            save_year_data(df, year, output_folder)
        else:
            print(f"Data processing failed for {year}, file not generated.")

if __name__ == "__main__":
    main()