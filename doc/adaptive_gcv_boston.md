# Adaptive GCV effect cap (Stage 2, large/wide): boston

- Response formula: `medv ~ .`
- Rows: 506
- Predictors: 13
- earth `degree` = 2
- Task type: regression
- Dominant predictor studied: `lstat`
- OOS engine: earth built-in cross-validation, nfold = 5, ncross = 3, seed 2024
- Adaptive path (`adaptive.gcv = TRUE`, no `linpreds`) at effect.cap = 0.5: **HELPS** OOS vs ordinary earth (this measures form competition AND effect-cap shrinkage combined; see the attribution note below).
- Forward-pass form vs stock at effect.cap = 0.5: **UNCHANGED (OOS movement is cap shrinkage, not a form change)**.

## Dominant predictor

`lstat` is the dominant continuous predictor of `medv` (|cor| = 0.738).
It is the strongest single predictor and is the one whose linear-vs-hinge
form we track in detail.

## Three-way comparison: ordinary vs automatic-adaptive vs forced-linpreds

`dominant_form` is how the dominant predictor entered the forward-pass
model (`linear` = admitted as a linpred / dirs code 2; `hinge` = knot term). Rows: (a) ordinary stock earth, (b) automatic adaptive competition (no `linpreds`), (c) adaptive with `lstat` forced linear via `linpreds`. `elapsed_s` is wall-clock seconds for the CV fit plus the forward-pass fit.

| setting | dominant_form | insample_rsq | cv_rsq | cv_classrate | nterms | elapsed_s |
| --- | --- | --- | --- | --- | --- | --- |
| (a) ordinary (adaptive OFF) | hinge | 0.9148146 | 0.6925548 | NA | 21 | 0.323 |
| (b) automatic adaptive cap=0.5 | hinge | 0.8984994 | 0.7644533 | NA | 21 | 0.316 |
| (b) automatic adaptive cap=0.9 | hinge | 0.9129706 | 0.7245876 | NA | 21 | 0.313 |
| (c) forced-linpreds cap=0.5 | linear | 0.8904235 | 0.8327601 | NA | 19 | 0.299 |
| (c) forced-linpreds cap=0.9 | linear | 0.9050407 | 0.8414278 | NA | 19 | 0.288 |

**Form verdict (effect.cap = 0.5): NO - at effect.cap=0.5 the automatic competition kept `lstat` in the same `hinge` form that ordinary earth used (`hinge`); the signal is not preferred as a plain linear term here, so automatic competition does not diverge from stock for this predictor.**

**OOS (earth built-in CV RSq) at effect.cap=0.5: ordinary 0.6926, automatic-adaptive 0.7645 (+0.0719 vs ordinary), forced-linpreds 0.8328 (+0.1402 vs ordinary).**

**Per-predictor form counts under AUTOMATIC competition (effect.cap=0.5), counted across all predictors in the forward-pass `$dirs`: **linear (code 2): 0**, **hinge (+/-1): 9**, **mixed: 0**. For comparison ordinary stock earth used linear: 0, hinge: 9, mixed: 0.**

**Attribution: this HELPS label measures the whole adaptive.gcv=TRUE path (form competition + effect-cap coefficient shrinkage), not form competition alone. Here the OOS movement is driven by the effect cap's coefficient SHRINKAGE (the forward-pass form is IDENTICAL to stock earth - same terms, same per-predictor form counts - so the OOS movement is regularisation, NOT a form change).**

**OOS verdict at scale (effect.cap = 0.5): the adaptive.gcv=TRUE path (form competition + cap shrinkage) HELPS OOS vs ordinary earth for this dataset. Forward-pass form vs stock: UNCHANGED - the OOS movement is coefficient shrinkage from the cap, not a form change.**

