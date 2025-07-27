import pandas as pd
import numpy as np
import shap
import yaml
from mambular.models import MLPClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

# -------- 数据加载 --------
file_name = 'D:\\project\\Delay_data\\Datasets\\arr_delay_data.csv'
df = pd.read_csv(file_name)

with open('D:\\project\\Delay_data\\Datasets\\columns_and_data_info.yaml', 'r') as yaml_file:
    data_info = yaml.load(yaml_file, Loader=yaml.FullLoader)

categorical_columns = data_info['columns_info']['Categorical Features']
continuous_columns = data_info['columns_info']['Continuous Features']

continuous_columns = [col for col in continuous_columns if col != 'FLIGHTS']
categorical_columns = [col for col in categorical_columns if col not in ['FL_YEAR', 'FL_MONTH']]

# 二分类目标（ARR_DELAY > 15分钟）
df['ARR_DELAY'] = df['ARR_DELAY'].apply(lambda x: 1 if abs(x) > 15.0 else 0)

# 划分数据集
df_train, df_temp = train_test_split(df, test_size=0.4, random_state=42, shuffle=True)
df_vaild, df_test = train_test_split(df_temp, test_size=0.5, random_state=42, shuffle=True)

X_train = df_train[categorical_columns + continuous_columns]
X_vaild = df_vaild[categorical_columns + continuous_columns]
X_test = df_test[categorical_columns + continuous_columns]

y_train = df_train['ARR_DELAY']
y_vaild = df_vaild['ARR_DELAY']
y_test = df_test['ARR_DELAY']

# -------- 模型训练 --------
model = MLPClassifier(d_model=64)
print("Training MLP...")
model.fit(X_train, y_train, X_val=X_vaild, y_val=y_vaild, max_epochs=50, lr=1e-3, patience=5)

# -------- 模型评估 --------
y_pred = model.predict(X_test)
# y_pred = (y_pred_prob >= 0.5).astype(int)

auc = roc_auc_score(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)

print(f"MLP - AUC: {auc:.4f}, ACC: {acc:.4f}")

# -------- SHAP 解释 --------
print("Calculating SHAP values...")
feature_names = categorical_columns + continuous_columns

# 子采样以加速
X_train_sample = X_train.sample(200, random_state=42)
X_test_sample = X_test.sample(100, random_state=42)

# 使用概率预测函数进行解释
explainer = shap.Explainer(model.predict, X_train_sample)
shap_values = explainer(X_test_sample)

# -------- SHAP 数值分析 --------
mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
shap_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Mean |SHAP Value|': mean_abs_shap
}).sort_values(by='Mean |SHAP Value|', ascending=False)
mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
shap_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Mean |SHAP Value|': mean_abs_shap
}).sort_values(by='Mean |SHAP Value|', ascending=False)

print("\nTop 10 SHAP features:")
print(shap_importance_df.head(10))
print("\nTop 10 SHAP features:")
print(shap_importance_df.head(10))

# 保存结果
shap_importance_df.to_csv('shap_feature_importance.csv', index=False)

# -------- SHAP 可视化 --------
plt.figure(figsize=(10, 6))
shap.plots.bar(shap_values, max_display=15)
plt.title('Top 15 Feature Importance (SHAP)', fontsize=14)
plt.tight_layout()
plt.savefig("shap_bar_top15.png")
plt.show()

shap.plots.beeswarm(shap_values, max_display=15)
