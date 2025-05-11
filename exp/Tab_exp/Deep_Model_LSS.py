import numpy as np
import pandas as pd
import yaml
from mambular.models import AutoIntLSS, FTTransformerLSS, MLPLSS, TangosLSS, ModernNCALSS, MambularLSS
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
    "AutoInt": AutoIntLSS(d_model=64, n_layers=8),
    "FTTransformer": FTTransformerLSS(d_model=64, n_layers=8),
    "MLP": MLPLSS(d_model=64),
    "Tangos": TangosLSS(d_model=64),
    "ModernNCA": ModernNCALSS(d_model=64),
    "Mambular": MambularLSS(d_model=64)
}

results_df = pd.DataFrame(columns=['Model', 'NLL', 'CRPS'])

for model_name, model in models.items():
    model.fit(X_train, y_train, family='normal', X_val=X_vaild, y_val=y_vaild, max_epochs=50, lr=1e-3, patience=5)

    y_pred = model.predict(X_test)

    y_pred_mean = y_pred[:, 0]  # 提取均值（期望值）
    y_pred_std = y_pred[:, 1]

    delta = y_test - y_pred_mean
    sigma_sq = y_pred_std ** 2
    nll = 0.5 * np.mean(delta ** 2 / sigma_sq + np.log(sigma_sq) + np.log(2 * np.pi))
    print(f"NLL (calculated): {nll}")

    ######################## 计算 CRPS #########################
    from scipy.special import erf

    z = delta / y_pred_std
    phi = np.exp(-0.5 * z ** 2) / np.sqrt(2 * np.pi)  # 标准正态PDF值
    Phi = 0.5 * (1 + erf(z / np.sqrt(2)))  # 标准正态CDF值
    crps_values = y_pred_std * (z * (2 * Phi - 1) + 2 * phi - 1 / np.sqrt(np.pi))
    crps = crps_values.mean()
    print(f"CRPS: {crps}")

    print(f"{model_name} - NLL: {nll}, CRPS: {crps}")

    result = pd.DataFrame({'Model': [model_name], 'NLL': [nll], 'CRPS': [crps]})

    results_df = pd.concat([results_df, result], ignore_index=True)

    results_df.to_csv('LSS_results.csv', index=False)




print('end')