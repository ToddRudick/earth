# Adaptive GCV effect cap (Stage 2, large/wide): spam

- Response formula: `type ~ .`
- Rows: 4601
- Predictors: 57
- earth `degree` = 1
- Task type: binary classification
- Dominant predictor studied: `your`
- OOS engine: earth built-in cross-validation, nfold = 5, ncross = 3, seed 2024
- Automatic form competition at effect.cap = 0.5: **NEUTRAL** OOS vs ordinary earth.

## Dominant predictor (point-biserial)

The dominant continuous predictor is `your`, the one with the largest absolute point-biserial correlation with the 0/1 `type` response (|r| = 0.383, where nonspam = 0, spam = 1). It is computed in-script and reported here for reproducibility.

## Three-way comparison: ordinary vs automatic-adaptive vs forced-linpreds

`dominant_form` is how the dominant predictor entered the forward-pass
model (`linear` = admitted as a linpred / dirs code 2; `hinge` = knot term). Rows: (a) ordinary stock earth, (b) automatic adaptive competition (no `linpreds`), (c) adaptive with `your` forced linear via `linpreds`. `elapsed_s` is wall-clock seconds for the CV fit plus the forward-pass fit.

| setting | dominant_form | insample_rsq | cv_rsq | cv_classrate | nterms | elapsed_s |
| --- | --- | --- | --- | --- | --- | --- |
| (a) ordinary (adaptive OFF) | absent | 0.6902437 | 0.6813935 | 0.9251667 | 16 | 2.505 |
| (b) automatic adaptive cap=0.5 | absent | 0.6886657 | 0.6801018 | 0.9251667 | 16 | 2.328 |
| (b) automatic adaptive cap=0.9 | absent | 0.6900681 | 0.6814022 | 0.9251667 | 16 | 2.267 |
| (c) forced-linpreds cap=0.5 | absent | 0.6886657 | 0.6816621 | 0.9260406 | 16 | 2.166 |
| (c) forced-linpreds cap=0.9 | absent | 0.6900681 | 0.6830742 | 0.9260406 | 16 | 2.177 |

**Form verdict (effect.cap = 0.5): NO - at effect.cap=0.5 the automatic competition kept `your` as a absent form (the same shape ordinary earth used: absent); the signal is not preferred as a plain linear term here, so automatic competition does not diverge from stock for this predictor.**

**OOS (earth built-in CV RSq) at effect.cap=0.5: ordinary 0.6814, automatic-adaptive 0.6801 (-0.001292 vs ordinary), forced-linpreds 0.6817 (+0.0002686 vs ordinary).**

**Per-predictor form counts under AUTOMATIC competition (effect.cap=0.5), counted across all predictors in the forward-pass `$dirs`: **linear (code 2): 0**, **hinge (+/-1): 10**, **mixed: 0**. For comparison ordinary stock earth used linear: 0, hinge: 10, mixed: 0.**

**OOS verdict at scale (effect.cap = 0.5): automatic form competition NEUTRAL OOS vs ordinary earth for this dataset.**

