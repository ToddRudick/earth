# Adaptive GCV effect cap (Stage 2, large/wide): spam

- Response formula: `type ~ .`
- Rows: 4601
- Predictors: 57
- earth `degree` = 1
- Task type: binary classification
- Dominant predictor studied: `your`
- OOS engine: earth built-in cross-validation, nfold = 5, ncross = 3, seed 2024
- Adaptive path (`adaptive.gcv = TRUE`, no `linpreds`) at effect.cap = 0.5: **NEUTRAL** OOS vs ordinary earth (this measures form competition AND effect-cap shrinkage combined; see the attribution note below).
- Forward-pass form vs stock at effect.cap = 0.5: **UNCHANGED (OOS movement is cap shrinkage, not a form change)**.

## Dominant predictor (point-biserial)

The dominant continuous predictor is `your`, the one with the largest absolute point-biserial correlation with the 0/1 `type` response (|r| = 0.383, where nonspam = 0, spam = 1). It is computed in-script and reported here for reproducibility.

**Caveat (form study N/A for spam):** the marginally-dominant `your` does NOT survive earth's degree-1 forward pass - it is `absent` from every fitted model (ordinary, automatic and forced). Marginal correlation does not guarantee a predictor enters the model when it competes against 56 others. So the single-dominant-predictor form study is **Not Applicable** for spam. The overall three-way OOS comparison (classification metric = CV class-rate) and the per-predictor form counts across all admitted predictors remain meaningful and are reported below.

## Three-way comparison: ordinary vs automatic-adaptive vs forced-linpreds

`dominant_form` is how the dominant predictor entered the forward-pass
model (`linear` = admitted as a linpred / dirs code 2; `hinge` = knot term). Rows: (a) ordinary stock earth, (b) automatic adaptive competition (no `linpreds`), (c) adaptive with `your` forced linear via `linpreds`. `elapsed_s` is wall-clock seconds for the CV fit plus the forward-pass fit.

| setting | dominant_form | insample_rsq | cv_rsq | cv_classrate | nterms | elapsed_s |
| --- | --- | --- | --- | --- | --- | --- |
| (a) ordinary (adaptive OFF) | absent | 0.6902437 | 0.6813935 | 0.9251667 | 16 | 2.513 |
| (b) automatic adaptive cap=0.5 | absent | 0.6886657 | 0.6801018 | 0.9251667 | 16 | 2.367 |
| (b) automatic adaptive cap=0.9 | absent | 0.6900681 | 0.6814022 | 0.9251667 | 16 | 2.316 |
| (c) forced-linpreds cap=0.5 | absent | 0.6886657 | 0.6816621 | 0.9260406 | 16 | 2.269 |
| (c) forced-linpreds cap=0.9 | absent | 0.6900681 | 0.6830742 | 0.9260406 | 16 | 2.196 |

**Form verdict (effect.cap = 0.5): N/A - at effect.cap=0.5 the marginally-dominant predictor `your` did NOT enter the forward pass in ANY setting (it is `absent` from ordinary, automatic AND forced fits), so the dominant-predictor form study is Not Applicable for this dataset. The overall three-way OOS comparison and the per-predictor form counts below remain meaningful; only the single-predictor form verdict is vacuous here.**

**OOS (earth built-in CV RSq) at effect.cap=0.5: ordinary 0.6814, automatic-adaptive 0.6801 (-0.001292 vs ordinary), forced-linpreds 0.6817 (+0.0002686 vs ordinary).**

**Per-predictor form counts under AUTOMATIC competition (effect.cap=0.5), counted across all predictors in the forward-pass `$dirs`: **linear (code 2): 0**, **hinge (+/-1): 10**, **mixed: 0**. For comparison ordinary stock earth used linear: 0, hinge: 10, mixed: 0.**

**Attribution: the adaptive.gcv=TRUE path is NEUTRAL here, and the forward-pass form is IDENTICAL to stock. So neither form competition nor shrinkage moved OOS materially for this dataset.**

**OOS verdict at scale (effect.cap = 0.5): the adaptive.gcv=TRUE path (form competition + cap shrinkage) NEUTRAL OOS vs ordinary earth for this dataset. Forward-pass form vs stock: UNCHANGED - the OOS movement is coefficient shrinkage from the cap, not a form change.**

