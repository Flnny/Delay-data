import pandas as pd

file_name = "E:\\Delay_data\\flight_with_weather_2024.csv"

df = pd.read_csv(file_name)

df['FL_DATE'] = pd.to_datetime(df['FL_DATE'])

start_date = '2024-06-01'
end_date = '2024-06-15'

filtered_df = df[(df['FL_DATE'] >= start_date) & (df['FL_DATE'] <= end_date)]

filtered_df.to_csv('filtered_flight_data.csv', index=False)
