import pandas as pd
import yaml
from mambular.models import AutoIntRegressor, FTTransformerRegressor, MLPRegressor, TangosRegressor, ModernNCARegressor, \
    MambularRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

file_name = 'D:\\project\\Delay_data\\Datasets\\data_processed.csv'
df = pd.read_csv(file_name)

with open('D:\\project\\Delay_data\\Datasets\\columns_and_data_info.yaml', 'r') as yaml_file:
    data_info = yaml.load(yaml_file, Loader=yaml.FullLoader)

categorical_columns = data_info['columns_info']['Categorical Features']
continuous_columns = data_info['columns_info']['Continuous Features']

continuous_columns = [col for col in continuous_columns if col != 'FLIGHTS']
categorical_columns = [col for col in categorical_columns if col != 'FL_YEAR' and col != 'FL_MONTH']

scaler = StandardScaler()
df['ARR_DELAY'] = scaler.fit_transform(df[['ARR_DELAY']])

df_train = df[df['FL_DAY'] <= 9]
df_vaild = df[(df['FL_DAY'] > 9) & (df['FL_DAY'] <= 12)]
df_test = df[df['FL_DAY'] > 12]

X_train = df_train[categorical_columns + continuous_columns]
X_vaild = df_vaild[categorical_columns + continuous_columns]
X_test = df_test[categorical_columns + continuous_columns]

y_train = df_train['ARR_DELAY']
y_vaild = df_vaild['ARR_DELAY']
y_test = df_test['ARR_DELAY']


models = {
    "AutoInt": AutoIntRegressor(d_model=64, n_layers=8),
    "FTTransformer": FTTransformerRegressor(d_model=64, n_layers=8),
    "MLP": MLPRegressor(d_model=64),
    "Tangos": TangosRegressor(d_model=64),
    "ModernNCA": ModernNCARegressor(d_model=64),
    "Mambular": MambularRegressor(d_model=64)
}

results_df = pd.DataFrame(columns=['Model', 'MSE', 'MAE'])

for model_name, model in models.items():

    model.fit(X_train, y_train, X_val=X_vaild, y_val=y_vaild, max_epochs=50, lr=1e-3, patience=5)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"{model_name} - MSE: {mse}, MAE: {mae}")

    result = pd.DataFrame({'Model': [model_name], 'MSE': [mse], 'MAE': [mae]})

    results_df = pd.concat([results_df, result], ignore_index=True)

    results_df.to_csv('Regressor_results.csv', index=False)

print('end')