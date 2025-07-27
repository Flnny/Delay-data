from random import randint, uniform

import numpy as np
import pandas as pd
import yaml
from mambular.models import AutoIntClassifier, FTTransformerClassifier, MLPClassifier, TangosClassifier, \
    ModernNCAClassifier, TabulaRNNClassifier, SAINTClassifier, ResNetClassifier
from mambular.models.tabr import TabRClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, roc_auc_score, accuracy_score
from mambular.models import MambularClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler

file_name = 'D:\\project\\Delay_data\\Datasets\\arr_delay_data.csv'
df = pd.read_csv(file_name)
df = df.dropna()

df['ARR_DELAY'] = df['ARR_DELAY'].apply(lambda x: 1 if abs(x) > 15.0 else 0)

df_train = df[df['FL_DAY'] <= 9]
df_vaild = df[(df['FL_DAY'] > 9) & (df['FL_DAY'] <= 12)]
df_test = df[df['FL_DAY'] > 12]

df_list = [df_train, df_vaild, df_test]

with open('D:\\project\\Delay_data\\Datasets\\arr_delay_data_info.yaml', 'r') as yaml_file:
    data_info = yaml.load(yaml_file, Loader=yaml.FullLoader)

categorical_columns = data_info['columns_info']['Categorical Features']
continuous_columns = data_info['columns_info']['Continuous Features']

continuous_columns = [col for col in continuous_columns if col != 'FLIGHTS']
categorical_columns = [col for col in categorical_columns if col != 'FL_YEAR' and col != 'FL_MONTH']

for df_1 in df_list:
    label_encoders = {}
    for col in categorical_columns:
        le = LabelEncoder()
        df_1[col] = le.fit_transform(df_1[col])
        label_encoders[col] = le
    scaler = StandardScaler()
    df_1[continuous_columns] = scaler.fit_transform(df_1[continuous_columns])

X_train = df_train[categorical_columns + continuous_columns]
X_vaild = df_vaild[categorical_columns + continuous_columns]
X_test = df_test[categorical_columns + continuous_columns]

y_train = df_train['ARR_DELAY']
y_vaild = df_vaild['ARR_DELAY']
y_test = df_test['ARR_DELAY']


results_df = pd.DataFrame(columns=['Model', 'Train AUC', 'Train ACC', 'Test AUC', 'Test ACC'])

models = {
    'MLP': MLPClassifier(d_model=128),
    'AutoInt': AutoIntClassifier(d_model=128, n_layers=4),
    'ResNet': ResNetClassifier(),
    'FTTransformer': FTTransformerClassifier(d_model=128, n_layers=4),
    'Tangos': TangosClassifier(d_model=128),
    'TabulaRNN': TabulaRNNClassifier(d_model=128),
    'SAINT': SAINTClassifier(d_model=128),
}

results_df = pd.DataFrame(columns=['Model', 'AUC', 'ACC'])

param_dist = {
    'd_model': randint(32, 128),
    'n_layers': randint(2, 10),
    'lr': uniform(1e-5, 1e-3)
}

for model_name, model in models.items():
    print(f"RandomizedSearchCV for {model_name}...")
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=10,
        cv=3,
        scoring='accuracy',True
        random_state=42
    )
    fit_params = {"max_epochs": 100, "rebuild": True, "X_val": X_vaild, "y_val": y_vaild,
                  "patience": 5}
    random_search.fit(X_train, y_train, **fit_params)
    print("Best Parameters:", random_search.best_params_)
    print("Best Score:", random_search.best_score_)

    best_model = random_search.best_estimator_
    y_pred = best_model.predict(X_test)
    auc = roc_auc_score(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    print(f"{model_name} - AUC: {auc}, ACC: {acc}")

    result = pd.DataFrame({'Model': [model_name], 'AUC': [auc], 'ACC': [acc]})
    results_df = pd.concat([results_df, result], ignore_index=True)
    results_df.to_csv('Classifier_results.csv', index=False)

print('end')
