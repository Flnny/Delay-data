import pandas as pd
import yaml
from sklearn.feature_selection import mutual_info_classif

file_name = 'D:\\project\\Delay_data\\Datasets\\arr_delay_data.csv'
df = pd.read_csv(file_name)
df = df.dropna()

with open('D:\\project\\Delay_data\\Datasets\\columns_and_data_info.yaml', 'r') as yaml_file:
    data_info = yaml.load(yaml_file, Loader=yaml.FullLoader)

categorical_columns = data_info['columns_info']['Categorical Features']
continuous_columns = data_info['columns_info']['Continuous Features']

X = df[categorical_columns + continuous_columns]
y = df['DEP_DELAY']

mi = mutual_info_classif(X, y, discrete_features='auto', random_state=0)
for col, score in zip(X.columns, mi):
    print(f"{col}: MI = {score:.4f}")