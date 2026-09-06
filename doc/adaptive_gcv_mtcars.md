# Adaptive GCV effect cap: mtcars

- Response formula: `mpg ~ .`
- Rows: 32 (train 22 / test 10, 70/30 holdout, seed 2024)
- earth `degree` = 1
- Task type: regression
- OOS bagging engine: caret::bagEarth (B = 30)
- earth built-in CV: nfold = 5, ncross = 3

## Out-of-sample: bagged earth (ordinary vs adaptive)

| setting | oos_rmse | oos_r2 |
| --- | --- | --- |
| ordinary (OFF) | 2.781329 | 0.5736188 |
| adaptive (ON) | 2.772632 | 0.5762813 |

## Cross-validation (earth built-in, independent of caret)

| setting | insample_rsq | cv_rsq | cv_class_rate |
| --- | --- | --- | --- |
| ordinary (OFF) | 0.8601598 | 0.6484781 | NA |
| adaptive (ON) | 0.8601598 | 0.6484781 | NA |

## Selected terms and coefficients (single model on full data)

### ordinary earth (adaptive.gcv = FALSE)
Selected 3 of 14 terms; in-sample RSq = 0.8602, GCV = 6.9121

| term | coefficient |
|------|-------------|
| `(Intercept)` | 20.5348 |
| `h(disp-145)` | -0.0250129 |
| `h(145-disp)` | 0.14859 |

### adaptive earth (adaptive.gcv = TRUE)
Selected 3 of 14 terms; in-sample RSq = 0.8602, GCV = 6.9121

| term | coefficient |
|------|-------------|
| `(Intercept)` | 20.5348 |
| `h(disp-145)` | -0.0250129 |
| `h(145-disp)` | 0.14859 |

