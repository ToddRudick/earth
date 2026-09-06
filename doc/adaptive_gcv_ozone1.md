# Adaptive GCV effect cap: ozone1

- Response formula: `O3 ~ .`
- Rows: 330 (train 231 / test 99, 70/30 holdout, seed 2024)
- earth `degree` = 2
- Task type: regression
- OOS bagging engine: caret::bagEarth (B = 30)
- earth built-in CV: nfold = 5, ncross = 3

## Out-of-sample: bagged earth (ordinary vs adaptive)

| setting | oos_rmse | oos_r2 |
| --- | --- | --- |
| ordinary (OFF) | 4.097401 | 0.7050523 |
| adaptive (ON) | 4.097401 | 0.7050523 |

## Cross-validation (earth built-in, independent of caret)

| setting | insample_rsq | cv_rsq | cv_class_rate |
| --- | --- | --- | --- |
| ordinary (OFF) | 0.8253857 | 0.7279271 | NA |
| adaptive (ON) | 0.8253857 | 0.7279271 | NA |

## Selected terms and coefficients (single model on full data)

### ordinary earth (adaptive.gcv = FALSE)
Selected 12 of 21 terms; in-sample RSq = 0.8254, GCV = 13.385

| term | coefficient |
|------|-------------|
| `(Intercept)` | 13.3286 |
| `h(temp-58)` | 0.372287 |
| `h(55-humidity)*h(temp-58)` | -0.0222206 |
| `h(doy-96)` | -0.0242853 |
| `h(96-doy)` | -0.123676 |
| `h(temp-58)*h(dpg-52)` | -0.0168976 |
| `h(temp-58)*h(52-dpg)` | 0.00408711 |
| `h(200-vis)` | 0.0221773 |
| `h(wind-7)*h(200-vis)` | -0.0181059 |
| `h(194-ibt)` | -0.0459797 |
| `h(1105-ibh)*h(21-dpg)` | -0.000102969 |
| `h(5740-vh)*h(temp-58)` | -0.00863238 |

### adaptive earth (adaptive.gcv = TRUE)
Selected 12 of 21 terms; in-sample RSq = 0.8254, GCV = 13.385

| term | coefficient |
|------|-------------|
| `(Intercept)` | 13.3286 |
| `h(temp-58)` | 0.372287 |
| `h(55-humidity)*h(temp-58)` | -0.0222206 |
| `h(doy-96)` | -0.0242853 |
| `h(96-doy)` | -0.123676 |
| `h(temp-58)*h(dpg-52)` | -0.0168976 |
| `h(temp-58)*h(52-dpg)` | 0.00408711 |
| `h(200-vis)` | 0.0221773 |
| `h(wind-7)*h(200-vis)` | -0.0181059 |
| `h(194-ibt)` | -0.0459797 |
| `h(1105-ibh)*h(21-dpg)` | -0.000102969 |
| `h(5740-vh)*h(temp-58)` | -0.00863238 |

