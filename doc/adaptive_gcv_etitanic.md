# Adaptive GCV effect cap: etitanic

- Response formula: `survived ~ .`
- Rows: 1046 (train 732 / test 314, 70/30 holdout, seed 2024)
- earth `degree` = 2
- Task type: binary classification
- OOS bagging engine: caret::bagEarth (B = 30)
- earth built-in CV: nfold = 5, ncross = 3

## Out-of-sample: bagged earth (ordinary vs adaptive)

| setting | oos_rmse | oos_acc | oos_brier |
| --- | --- | --- | --- |
| ordinary (OFF) | 0.34406 | 0.8343949 | 0.1183773 |
| adaptive (ON) | 0.34406 | 0.8343949 | 0.1183773 |

## Cross-validation (earth built-in, independent of caret)

| setting | insample_rsq | cv_rsq | cv_class_rate |
| --- | --- | --- | --- |
| ordinary (OFF) | 0.4389834 | 0.4010213 | 0.7932107 |
| adaptive (ON) | 0.4389834 | 0.4010213 | 0.7932107 |

## Selected terms and coefficients (single model on full data)

### ordinary earth (adaptive.gcv = FALSE)
Selected 8 of 17 terms; in-sample RSq = 0.4390, GCV = 0.14045

| term | coefficient |
|------|-------------|
| `(Intercept)` | 0.96171 |
| `sexmale` | -0.570035 |
| `pclass3rd` | -0.815454 |
| `sexmale*h(16-age)` | 0.0450523 |
| `pclass2nd*sexmale` | -0.265689 |
| `pclass3rd*h(4-sibsp)` | 0.102222 |
| `pclass3rd*sexmale` | 0.193102 |
| `h(age-32)` | -0.00471938 |

### adaptive earth (adaptive.gcv = TRUE)
Selected 8 of 17 terms; in-sample RSq = 0.4390, GCV = 0.14045

| term | coefficient |
|------|-------------|
| `(Intercept)` | 0.96171 |
| `sexmale` | -0.570035 |
| `pclass3rd` | -0.815454 |
| `sexmale*h(16-age)` | 0.0450523 |
| `pclass2nd*sexmale` | -0.265689 |
| `pclass3rd*h(4-sibsp)` | 0.102222 |
| `pclass3rd*sexmale` | 0.193102 |
| `h(age-32)` | -0.00471938 |

