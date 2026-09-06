# Adaptive GCV effect cap: trees

- Response formula: `Volume ~ .`
- Rows: 31 (train 22 / test 9, 70/30 holdout, seed 2024)
- earth `degree` = 1
- Task type: regression
- OOS bagging engine: caret::bagEarth (B = 30)
- earth built-in CV: nfold = 5, ncross = 3

## Out-of-sample: bagged earth (ordinary vs adaptive)

| setting | oos_rmse | oos_r2 |
| --- | --- | --- |
| ordinary (OFF) | 2.80655 | 0.9575308 |
| adaptive (ON) | 4.27883 | 0.9012859 |

## Cross-validation (earth built-in, independent of caret)

| setting | insample_rsq | cv_rsq | cv_class_rate |
| --- | --- | --- | --- |
| ordinary (OFF) | 0.9742029 | 0.9090899 | NA |
| adaptive (ON) | 0.9127769 | 0.8225490 | NA |

## Selected terms and coefficients (single model on full data)

### ordinary earth (adaptive.gcv = FALSE)
Selected 4 of 5 terms; in-sample RSq = 0.9742, GCV = 11.254

| term | coefficient |
|------|-------------|
| `(Intercept)` | 29.06 |
| `h(Girth-14.2)` | 6.22951 |
| `h(14.2-Girth)` | -3.41981 |
| `h(Height-75)` | 0.581364 |

### adaptive earth (adaptive.gcv = TRUE)
Selected 4 of 5 terms; in-sample RSq = 0.9128, GCV = 38.052

| term | coefficient |
|------|-------------|
| `(Intercept)` | 28.9372 |
| `h(Girth-14.2)` | 4.5215 |
| `h(14.2-Girth)` | -2.51767 |
| `h(Height-75)` | 0.581364 |

