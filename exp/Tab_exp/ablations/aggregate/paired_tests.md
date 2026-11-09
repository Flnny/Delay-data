# Paired DeLong tests (same test rows)

test_hash 7cab2c005e188b58, n_test 2547088. `p_max` = largest per-seed p-value; `delta_std` = spread of ΔAUC across seeds (training noise), `se_mean` = DeLong SE (test-sampling noise).

| A | B | extraction | model | p2 | seeds | ΔAUC mean | ΔAUC std (seeds) | DeLong SE | p_max |
|---|---|---|---|---|---|---|---|---|---|
| 02_shap_only | 01b_baseline_matched_training | 01_2024_baseline | AutoInt | 0.1 | 3 | -0.0030 | 0.0061 | 0.0002 | 4.3e-17 |
| 02_shap_only | 01b_baseline_matched_training | 01_2024_baseline | FTTransformer | 0.1 | 3 | -0.0038 | 0.0013 | 0.0002 | 8.5e-22 |
| 02_shap_only | 01b_baseline_matched_training | 01_2024_baseline | MLP | 0.1 | 3 | -0.0012 | 0.0018 | 0.0001 | 2.1e-05 |
| 02_shap_only | 01b_baseline_matched_training | 01_2024_baseline | ResNet | 0.1 | 3 | -0.0062 | 0.0032 | 0.0001 | 4.4e-187 |
| 02_shap_only | 01b_baseline_matched_training | 02_2020_train | AutoInt | 0.1 | 3 | -0.0037 | 0.0029 | 0.0002 | 0.01 |
| 02_shap_only | 01b_baseline_matched_training | 02_2020_train | FTTransformer | 0.1 | 3 | -0.0022 | 0.0023 | 0.0002 | 0.11 |
| 02_shap_only | 01b_baseline_matched_training | 02_2020_train | MLP | 0.1 | 3 | +0.0144 | 0.0067 | 0.0002 | 0 |
| 02_shap_only | 01b_baseline_matched_training | 02_2020_train | ResNet | 0.1 | 3 | +0.0024 | 0.0020 | 0.0001 | 0.0012 |
| 03_shap_plus_huim | 01b_baseline_matched_training | 01_2024_baseline | AutoInt | 0.1 | 3 | -0.0166 | 0.0022 | 0.0002 | 0 |
| 03_shap_plus_huim | 01b_baseline_matched_training | 01_2024_baseline | FTTransformer | 0.1 | 3 | -0.0076 | 0.0034 | 0.0002 | 2.1e-100 |
| 03_shap_plus_huim | 01b_baseline_matched_training | 01_2024_baseline | MLP | 0.1 | 3 | -0.0019 | 0.0009 | 0.0001 | 8.8e-22 |
| 03_shap_plus_huim | 01b_baseline_matched_training | 01_2024_baseline | ResNet | 0.1 | 3 | -0.0079 | 0.0029 | 0.0002 | 0 |
| 03_shap_plus_huim | 01b_baseline_matched_training | 02_2020_train | AutoInt | 0.1 | 3 | -0.0088 | 0.0036 | 0.0002 | 2.1e-159 |
| 03_shap_plus_huim | 01b_baseline_matched_training | 02_2020_train | FTTransformer | 0.1 | 3 | -0.0050 | 0.0010 | 0.0002 | 1.4e-57 |
| 03_shap_plus_huim | 01b_baseline_matched_training | 02_2020_train | MLP | 0.1 | 3 | +0.0170 | 0.0082 | 0.0002 | 0 |
| 03_shap_plus_huim | 01b_baseline_matched_training | 02_2020_train | ResNet | 0.1 | 3 | +0.0064 | 0.0069 | 0.0002 | 7.3e-07 |
| 03_shap_plus_huim | 02_shap_only | 01_2024_baseline | AutoInt | 0.1 | 3 | -0.0136 | 0.0058 | 0.0002 | 0 |
| 03_shap_plus_huim | 02_shap_only | 01_2024_baseline | FTTransformer | 0.1 | 3 | -0.0037 | 0.0028 | 0.0002 | 2.2e-14 |
| 03_shap_plus_huim | 02_shap_only | 01_2024_baseline | MLP | 0.1 | 3 | -0.0007 | 0.0011 | 0.0001 | 0.042 |
| 03_shap_plus_huim | 02_shap_only | 01_2024_baseline | ResNet | 0.1 | 3 | -0.0017 | 0.0049 | 0.0001 | 9.8e-204 |
| 03_shap_plus_huim | 02_shap_only | 02_2020_train | AutoInt | 0.1 | 3 | -0.0051 | 0.0027 | 0.0002 | 1.1e-40 |
| 03_shap_plus_huim | 02_shap_only | 02_2020_train | FTTransformer | 0.1 | 3 | -0.0028 | 0.0016 | 0.0002 | 3.2e-19 |
| 03_shap_plus_huim | 02_shap_only | 02_2020_train | MLP | 0.1 | 3 | +0.0026 | 0.0052 | 0.0002 | 0.63 |
| 03_shap_plus_huim | 02_shap_only | 02_2020_train | ResNet | 0.1 | 3 | +0.0040 | 0.0084 | 0.0001 | 0.2 |
