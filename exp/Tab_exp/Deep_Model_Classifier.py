import pandas as pd
import yaml
from mambular.models import AutoIntClassifier, FTTransformerClassifier, MLPClassifier, TangosClassifier, ModernNCAClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, roc_auc_score, accuracy_score
from mambular.models import MambularClassifier

file_name = 'D:\\project\\Delay_data\\Datasets\\data_processed.csv'
df = pd.read_csv(file_name)

with open('D:\\project\\Delay_data\\Datasets\\columns_and_data_info.yaml', 'r') as yaml_file:
    data_info = yaml.load(yaml_file, Loader=yaml.FullLoader)

categorical_columns = data_info['columns_info']['Categorical Features']
continuous_columns = data_info['columns_info']['Continuous Features']

continuous_columns = [col for col in continuous_columns if col != 'FLIGHTS']
categorical_columns = [col for col in categorical_columns if col != 'FL_YEAR' and col != 'FL_MONTH']

df['ARR_DELAY'] = df['ARR_DELAY'].apply(lambda x: 1 if abs(x) > 15.0 else 0)  # 将 ARR_DELAY 转换为二分类问题

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
    "AutoInt": AutoIntClassifier(d_model=64, n_layers=8),
    "FTTransformer": FTTransformerClassifier(d_model=64, n_layers=8),
    "MLP": MLPClassifier(d_model=64),
    "Tangos": TangosClassifier(d_model=64),
    "ModernNCA": ModernNCAClassifier(d_model=64),
    "Mambular": MambularClassifier(d_model=64)
}

results_df = pd.DataFrame(columns=['Model', 'AUC', 'ACC'])

for model_name, model in models.items():
    model.fit(X_train, y_train, X_val=X_vaild, y_val=y_vaild, max_epochs=50, lr=1e-3, patience=5)

    y_pred = model.predict(X_test)

    auc = roc_auc_score(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)

    print(f"{model_name} - AUC: {auc}, ACC: {acc}")

    result = pd.DataFrame({'Model': [model_name], 'AUC': [auc], 'ACC': [acc]})

    results_df = pd.concat([results_df, result], ignore_index=True)

    results_df.to_csv('Classifier_results.csv', index=False)

print('end')
