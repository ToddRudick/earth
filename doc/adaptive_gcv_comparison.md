# Adaptive GCV Effect Cap: in-sample vs out-of-sample comparison

This report compares ordinary earth (`adaptive.gcv = FALSE`, the default)
against the experimental adaptive GCV effect cap (`adaptive.gcv = TRUE`) on
several established datasets, measuring **out-of-sample** predictive power.

## Methodology

- **OOS bagging engine:** caret::bagEarth. bagEarth passes `...` through to `earth()`, so the identical bagging procedure is run with the feature OFF and ON.
- **Holdout:** a single 70/30 train/test split per dataset (`set.seed(2024)`); bagged earth is fit on train and scored on the held-out test rows.
- **Cross-validation:** earth's own k-fold CV (`nfold=5, ncross=3`, `set.seed(2024)`), fully independent of caret, reports cross-validated RSq (and classification rate for the binary response).
- **Metrics:** RMSE and R^2 for regression; RMSE, accuracy and Brier score for the binary `etitanic$survived` response.
- **Interactions:** `ozone1` and `etitanic` are fit with `degree = 2`.
- Per-dataset details (selected terms and coefficients for both settings) are in the companion files listed below.

## Datasets

| dataset | task | degree | rows | notes |
| --- | --- | --- | --- | --- |
| ozone1 | regression | 2 | 330 | canonical MARS / earth-vignette example |
| trees | regression | 1 | 31 | base R |
| mtcars | regression | 1 | 32 | base R |
| etitanic | classification | 2 | 1046 | used in earth examples; binary `survived` |

## Out-of-sample results (bagged earth, holdout test set)

### Regression (RMSE / R^2 on held-out test set)

| dataset | setting | OOS RMSE | OOS R^2 |
| --- | --- | --- | --- |
| ozone1 | ordinary (OFF) | 4.097 | 0.7051 |
| ozone1 | adaptive (ON) | 4.097 | 0.7051 |
| trees | ordinary (OFF) | 2.807 | 0.9575 |
| trees | adaptive (ON) | 4.279 | 0.9013 |
| mtcars | ordinary (OFF) | 2.781 | 0.5736 |
| mtcars | adaptive (ON) | 2.773 | 0.5763 |

### Classification (etitanic$survived, held-out test set)

| dataset | setting | OOS RMSE | OOS accuracy | OOS Brier |
| --- | --- | --- | --- | --- |
| etitanic | ordinary (OFF) | 0.3441 | 0.8344 | 0.1184 |
| etitanic | adaptive (ON) | 0.3441 | 0.8344 | 0.1184 |

## Cross-validation (earth built-in, independent of caret)

| dataset | setting | in-sample RSq | CV RSq | CV class-rate |
| --- | --- | --- | --- | --- |
| ozone1 | ordinary (OFF) | 0.8254 | 0.7279 | NA |
| ozone1 | adaptive (ON) | 0.8254 | 0.7279 | NA |
| trees | ordinary (OFF) | 0.9742 | 0.9091 | NA |
| trees | adaptive (ON) | 0.9128 | 0.8225 | NA |
| mtcars | ordinary (OFF) | 0.8602 | 0.6485 | NA |
| mtcars | adaptive (ON) | 0.8602 | 0.6485 | NA |
| etitanic | ordinary (OFF) | 0.439 | 0.401 | 0.7932 |
| etitanic | adaptive (ON) | 0.439 | 0.401 | 0.7932 |

## Interpretation

- **ozone1** (regression, degree 2): adaptive.gcv is NEUTRAL for out-of-sample RMSE (ordinary 4.097 vs adaptive 4.097, delta +1.776e-15).
- **trees** (regression, degree 1): adaptive.gcv HURTS out-of-sample RMSE (ordinary 2.807 vs adaptive 4.279, delta +1.472).
- **mtcars** (regression, degree 1): adaptive.gcv IMPROVES out-of-sample RMSE (ordinary 2.781 vs adaptive 2.773, delta -0.008697).
- **etitanic** (classification, degree 2): adaptive.gcv is NEUTRAL for out-of-sample Brier score (ordinary 0.1184 vs adaptive 0.1184, delta -2.082e-16); OOS accuracy ordinary 0.8344 vs adaptive 0.8344.

The adaptive cap is a conservative, default-off modification. Terms are selected by the
ordinary MARS forward pass, but each term's predictive effect (delta-R^2) is capped at
`effect.cap` (default 0.9, the maximum fraction of the total sum of squares any
single term may explain); the saved caps are then applied as a *constrained* final fit,
so the un-capped terms absorb the residual a dominant term is not allowed to explain.
Because a strong term is deliberately held below its unconstrained least-squares effect,
the adaptive model is more heavily regularised in-sample; whether that regularisation
helps or hurts out-of-sample generalisation is dataset dependent, as the table above
shows. Ordinary earth remains the default. The `effect.cap` argument tunes the strength:
`effect.cap >= 1` recovers stock earth, smaller values redistribute more signal.

## Companion per-dataset files

- `doc/adaptive_gcv_ozone1.md`
- `doc/adaptive_gcv_trees.md`
- `doc/adaptive_gcv_mtcars.md`
- `doc/adaptive_gcv_etitanic.md`

