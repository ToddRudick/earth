# Adaptive GCV effect cap (Stage 2): ozone1

- Response formula: `O3 ~ .`
- Rows: 330
- earth `degree` = 2
- Task type: regression
- Dominant predictor studied: `temp`
- OOS engine: earth built-in cross-validation, nfold = 5, ncross = 3, seed 2024

## Stage-2 automatic form competition

Under `adaptive.gcv = TRUE` with `effect.cap < 1` (and the default
`Auto.linpreds = TRUE`), the forward pass AUTOMATICALLY competes a HINGE
form and a LINEAR form of each candidate predictor and admits the form
with the higher JUSTIFIED (capped) effect.  A linear term is charged
`deltaKnots = 0` (larger `slackFactor`, shrunk less); a hinge is charged
`deltaKnots = 1`.  So a genuinely-linear dominant predictor can now be
admitted as a plain LINEAR term (its `$dirs` entry becomes direction code
**2**, a linpred with no knot) WITHOUT the user setting `linpreds`.  In
Stage 1 that linear form could only be obtained by FORCING it via
`linpreds`.  `effect.cap >= 1` disables the competition and reproduces
stock earth byte-for-byte.

## The Stage-2 question

> Does the AUTOMATIC competition (no `linpreds`) admit `temp` as a LINEAR term on its own, reproducing the Stage-1 forced-linpreds form, and does that help or at least not hurt OOS performance versus ordinary earth?

## Three-way comparison: ordinary vs automatic-adaptive vs forced-linpreds

`dominant_form` is how the dominant predictor entered the forward-pass
model (`linear` = admitted as a linpred / dirs code 2; `hinge` = knot term). Rows: (a) ordinary stock earth, (b) automatic adaptive competition (no `linpreds`), (c) adaptive with `temp` forced linear via `linpreds`.

| setting | dominant_form | insample_rsq | cv_rsq | cv_classrate | nterms |
| --- | --- | --- | --- | --- | --- |
| (a) ordinary (adaptive OFF) | hinge | 0.8253857 | 0.7279271 | NA | 12 |
| (b) automatic adaptive cap=0.5 | hinge | 0.8081294 | 0.7333624 | NA | 12 |
| (b) automatic adaptive cap=0.9 | hinge | 0.8234371 | 0.7350913 | NA | 12 |
| (c) forced-linpreds cap=0.5 | linear | 0.8177420 | 0.7306245 | NA | 13 |
| (c) forced-linpreds cap=0.9 | linear | 0.8279385 | 0.7294938 | NA | 13 |

**Form verdict (effect.cap = 0.5): NO - at effect.cap=0.5 the automatic competition kept `temp` as a hinge form (the same shape ordinary earth used: hinge); the signal is not preferred as a plain linear term here, so automatic competition does not diverge from stock for this predictor.**

**OOS (earth built-in CV RSq) at effect.cap=0.5: ordinary 0.7279, automatic-adaptive 0.7334 (+0.005435 vs ordinary), forced-linpreds 0.7306 (+0.002697 vs ordinary).**

