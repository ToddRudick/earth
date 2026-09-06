# Adaptive GCV effect cap (Stage 2, large/wide): synthetic_large

- Response formula: `y ~ .`
- Rows: 4000
- Predictors: 40
- earth `degree` = 2
- Task type: regression
- Dominant predictor studied: `x1`
- OOS engine: earth built-in cross-validation, nfold = 5, ncross = 3, seed 2024
- Automatic form competition at effect.cap = 0.5: **HELPS** OOS vs ordinary earth.

## Synthetic generator (fully reproducible)

`set.seed(2024)`, n = 4000 rows, p = 40 predictors. The response is

```
y = 5.0*x1                      # x1: GENUINELY LINEAR dominant predictor
  + 4.0*pmax(x2 - 0.3, 0)       # x2: hinge / nonlinear
  + 3.0*pmax(0.5 - x3, 0)       # x3: hinge / nonlinear
  + 2.5*(x4 * x5)               # x4,x5: interaction
  + 1.2*x6                      # x6: mild linear contributor
  + N(0, 1)                     # gaussian noise
x7 = x1 + N(0, 0.25)            # CORRELATED with dominant x1
x8 = x2 + N(0, 0.25)            # CORRELATED with hinge x2
x9 .. xp                        # PURE NOISE columns (no effect on y)
```

`x1` is the dominant, genuinely-LINEAR predictor. The key Stage-2 question
for this design: does the AUTOMATIC hinge-vs-linear competition admit `x1`
as a LINEAR term (dirs code 2) on its own, at large n and wide p?

## Three-way comparison: ordinary vs automatic-adaptive vs forced-linpreds

`dominant_form` is how the dominant predictor entered the forward-pass
model (`linear` = admitted as a linpred / dirs code 2; `hinge` = knot term). Rows: (a) ordinary stock earth, (b) automatic adaptive competition (no `linpreds`), (c) adaptive with `x1` forced linear via `linpreds`. `elapsed_s` is wall-clock seconds for the CV fit plus the forward-pass fit.

| setting | dominant_form | insample_rsq | cv_rsq | cv_classrate | nterms | elapsed_s |
| --- | --- | --- | --- | --- | --- | --- |
| (a) ordinary (adaptive OFF) | hinge | 0.8320372 | 0.8588714 | NA | 15 | 5.963 |
| (b) automatic adaptive cap=0.5 | linear | 0.8251368 | 0.8648205 | NA |  8 | 4.201 |
| (b) automatic adaptive cap=0.9 | hinge | 0.8318418 | 0.8583170 | NA | 15 | 5.723 |
| (c) forced-linpreds cap=0.5 | linear | 0.8251368 | 0.8648205 | NA |  8 | 3.311 |
| (c) forced-linpreds cap=0.9 | linear | 0.8259348 | 0.8658909 | NA |  8 | 3.341 |

**Form verdict (effect.cap = 0.5): YES - at effect.cap=0.5 the AUTOMATIC competition admitted `x1` as a plain LINEAR term (dirs code 2), on its own, reproducing the Stage-1 forced-linpreds form (pure linear, no knot). Ordinary earth entered it as a hinge.**

**OOS (earth built-in CV RSq) at effect.cap=0.5: ordinary 0.8589, automatic-adaptive 0.8648 (+0.005949 vs ordinary), forced-linpreds 0.8648 (+0.005949 vs ordinary).**

**Per-predictor form counts under AUTOMATIC competition (effect.cap=0.5), counted across all predictors in the forward-pass `$dirs`: **linear (code 2): 1**, **hinge (+/-1): 4**, **mixed: 0**. For comparison ordinary stock earth used linear: 0, hinge: 7, mixed: 0.**

**OOS verdict at scale (effect.cap = 0.5): automatic form competition HELPS OOS vs ordinary earth for this dataset.**

