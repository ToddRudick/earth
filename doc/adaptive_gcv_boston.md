# Adaptive GCV effect cap (Stage 2, large/wide): boston

- Response formula: `medv ~ .`
- Rows: 506
- Predictors: 13
- earth `degree` = 2
- Task type: regression
- Dominant predictor studied: `lstat`
- OOS engine: earth built-in cross-validation, nfold = 5, ncross = 3, seed 2024
- Automatic form competition at effect.cap = 0.5: **HELPS** OOS vs ordinary earth.

## Dominant predictor

`lstat` is the dominant continuous predictor of `medv` (|cor| = 0.738).
It is the strongest single predictor and is the one whose linear-vs-hinge
form we track in detail.

## Three-way comparison: ordinary vs automatic-adaptive vs forced-linpreds

`dominant_form` is how the dominant predictor entered the forward-pass
model (`linear` = admitted as a linpred / dirs code 2; `hinge` = knot term). Rows: (a) ordinary stock earth, (b) automatic adaptive competition (no `linpreds`), (c) adaptive with `lstat` forced linear via `linpreds`. `elapsed_s` is wall-clock seconds for the CV fit plus the forward-pass fit.

| setting | dominant_form | insample_rsq | cv_rsq | cv_classrate | nterms | elapsed_s |
| --- | --- | --- | --- | --- | --- | --- |
| (a) ordinary (adaptive OFF) | hinge | 0.9148146 | 0.6925548 | NA | 21 | 0.333 |
| (b) automatic adaptive cap=0.5 | hinge | 0.8984994 | 0.7644533 | NA | 21 | 0.321 |
| (b) automatic adaptive cap=0.9 | hinge | 0.9129706 | 0.7245876 | NA | 21 | 0.315 |
| (c) forced-linpreds cap=0.5 | linear | 0.8904235 | 0.8327601 | NA | 19 | 0.295 |
| (c) forced-linpreds cap=0.9 | linear | 0.9050407 | 0.8414278 | NA | 19 | 0.290 |

**Form verdict (effect.cap = 0.5): NO - at effect.cap=0.5 the automatic competition kept `lstat` as a hinge form (the same shape ordinary earth used: hinge); the signal is not preferred as a plain linear term here, so automatic competition does not diverge from stock for this predictor.**

**OOS (earth built-in CV RSq) at effect.cap=0.5: ordinary 0.6926, automatic-adaptive 0.7645 (+0.0719 vs ordinary), forced-linpreds 0.8328 (+0.1402 vs ordinary).**

**Per-predictor form counts under AUTOMATIC competition (effect.cap=0.5), counted across all predictors in the forward-pass `$dirs`: **linear (code 2): 0**, **hinge (+/-1): 9**, **mixed: 0**. For comparison ordinary stock earth used linear: 0, hinge: 9, mixed: 0.**

**OOS verdict at scale (effect.cap = 0.5): automatic form competition HELPS OOS vs ordinary earth for this dataset.**

