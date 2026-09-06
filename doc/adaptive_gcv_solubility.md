# Adaptive GCV effect cap (Stage 2, large/wide): solubility

- Response formula: `y ~ .`
- Rows: 951
- Predictors: 228
- earth `degree` = 2
- Task type: regression
- Dominant predictor studied: `MolWeight`
- OOS engine: earth built-in cross-validation, nfold = 5, ncross = 3, seed 2024
- Adaptive path (`adaptive.gcv = TRUE`, no `linpreds`) at effect.cap = 0.5: **NEUTRAL** OOS vs ordinary earth (this measures form competition AND effect-cap shrinkage combined; see the attribution note below).
- Forward-pass form vs stock at effect.cap = 0.5: **UNCHANGED (OOS movement is cap shrinkage, not a form change)**.

## Dominant predictor

This is the WIDE dataset: the design matrix `solTrainX` has 228 predictors
(binary fingerprint indicators plus continuous molecular descriptors), with
response `solTrainY` attached as column `y`.

The dominant continuous predictor is `MolWeight` (largest |cor| = 0.629 with `y`),
computed in-script and reported here for reproducibility.

## Three-way comparison: ordinary vs automatic-adaptive vs forced-linpreds

`dominant_form` is how the dominant predictor entered the forward-pass
model (`linear` = admitted as a linpred / dirs code 2; `hinge` = knot term). Rows: (a) ordinary stock earth, (b) automatic adaptive competition (no `linpreds`), (c) adaptive with `MolWeight` forced linear via `linpreds`. `elapsed_s` is wall-clock seconds for the CV fit plus the forward-pass fit.

| setting | dominant_form | insample_rsq | cv_rsq | cv_classrate | nterms | elapsed_s |
| --- | --- | --- | --- | --- | --- | --- |
| (a) ordinary (adaptive OFF) | hinge | 0.9193590 | 0.8692491 | NA | 28 | 14.987 |
| (b) automatic adaptive cap=0.5 | hinge | 0.9086893 | 0.8741828 | NA | 28 | 15.028 |
| (b) automatic adaptive cap=0.9 | hinge | 0.9181631 | 0.8745961 | NA | 28 | 14.995 |
| (c) forced-linpreds cap=0.5 | linear | 0.9077834 | 0.8802301 | NA | 28 | 17.707 |
| (c) forced-linpreds cap=0.9 | linear | 0.9177300 | 0.8813241 | NA | 28 | 17.844 |

**Form verdict (effect.cap = 0.5): NO - at effect.cap=0.5 the automatic competition kept `MolWeight` in the same `hinge` form that ordinary earth used (`hinge`); the signal is not preferred as a plain linear term here, so automatic competition does not diverge from stock for this predictor.**

**OOS (earth built-in CV RSq) at effect.cap=0.5: ordinary 0.8692, automatic-adaptive 0.8742 (+0.004934 vs ordinary), forced-linpreds 0.8802 (+0.01098 vs ordinary).**

**Per-predictor form counts under AUTOMATIC competition (effect.cap=0.5), counted across all predictors in the forward-pass `$dirs`: **linear (code 2): 8**, **hinge (+/-1): 11**, **mixed: 0**. For comparison ordinary stock earth used linear: 8, hinge: 11, mixed: 0.**

**Attribution: the adaptive.gcv=TRUE path is NEUTRAL here, and the forward-pass form is IDENTICAL to stock. So neither form competition nor shrinkage moved OOS materially for this dataset.**

**OOS verdict at scale (effect.cap = 0.5): the adaptive.gcv=TRUE path (form competition + cap shrinkage) NEUTRAL OOS vs ordinary earth for this dataset. Forward-pass form vs stock: UNCHANGED - the OOS movement is coefficient shrinkage from the cap, not a form change.**

