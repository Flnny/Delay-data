import pandas as pd
import yaml
from mambular.models import AutoIntRegressor, FTTransformerRegressor, MLPRegressor, TangosRegressor, ModernNCARegressor, \
    MambularRegressor, TabulaRNNRegressor, SAINTRegressor, ResNetRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

file_name = 'D:\\project\\Delay_data\\Datasets\\arr_delay_data.csv'
df = pd.read_csv(file_name)

with open('D:\\project\\Delay_data\\Datasets\\arr_delay_data_info.yaml', 'r') as yaml_file:
    data_info = yaml.load(yaml_file, Loader=yaml.FullLoader)

categorical_columns = data_info['columns_info']['Categorical Features']
continuous_columns = data_info['columns_info']['Continuous Features']

continuous_columns = [col for col in continuous_columns if col != 'FLIGHTS']
categorical_columns = [col for col in categorical_columns if col != 'FL_YEAR' and col != 'FL_MONTH']

# scaler = StandardScaler()
# df['DEP_DELAY'] = scaler.fit_transform(df[['DEP_DELAY']])

df_train = df[df['FL_DAY'] <= 9]
df_vaild = df[(df['FL_DAY'] > 9) & (df['FL_DAY'] <= 12)]
df_test = df[df['FL_DAY'] > 12]

X_train = df_train[categorical_columns + continuous_columns]
X_vaild = df_vaild[categorical_columns + continuous_columns]
X_test = df_test[categorical_columns + continuous_columns]

y_train = df_train['DEP_DELAY']
y_vaild = df_vaild['DEP_DELAY']
y_test = df_test['DEP_DELAY']


models = {
    "AutoInt": AutoIntRegressor(d_model=64, n_layers=8),
    "FTTransformer": FTTransformerRegressor(d_model=64, n_layers=8),
    "MLP": MLPRegressor(d_model=64),
    "Tangos": TangosRegressor(d_model=64),
    'TabulaRNN': TabulaRNNRegressor(d_model=64),
    'SAINT': SAINTRegressor(d_model=64),
    'ResNet': ResNetRegressor(),
}

from scipy.stats import randint, uniform
from sklearn.model_selection import RandomizedSearchCV

param_dist = {
    'd_model': randint(32, 128),
    'n_layers': randint(2, 10),
    'lr': uniform(1e-5, 1e-3)
}

results_df = pd.DataFrame(columns=['Model', 'MSE', 'MAE'])

for model_name, model in models.items():
    print(f"RandomizedSearchCV for {model_name}...")
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=10,
        cv=3,
        scoring='neg_mean_squared_error',
        random_state=42
    )
    fit_params = {"max_epochs": 100, "rebuild": True, "X_val": X_vaild, "y_val": y_vaild,
                  "patience": 5}

    random_search.fit(X_train, y_train, **fit_params)
    print("Best Parameters:", random_search.best_params_)
    print("Best Score:", random_search.best_score_)

    best_model = random_search.best_estimator_
    y_pred = best_model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"{model_name} - MSE: {mse}, MAE: {mae}")

    result = pd.DataFrame({'Model': [model_name], 'MSE': [mse], 'MAE': [mae]})

    results_df = pd.concat([results_df, result], ignore_index=True)
    results_df.to_csv('Regressor_results_DEP.csv', index=False)

print('end')
