# Adaptive GCV effect cap (Stage 2): trees

- Response formula: `Volume ~ .`
- Rows: 31
- earth `degree` = 1
- Task type: regression
- Dominant predictor studied: `Girth`
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

> Does the AUTOMATIC competition (no `linpreds`) admit `Girth` as a LINEAR term on its own, reproducing the Stage-1 forced-linpreds form, and does that help or at least not hurt OOS performance versus ordinary earth?

## Three-way comparison: ordinary vs automatic-adaptive vs forced-linpreds

`dominant_form` is how the dominant predictor entered the forward-pass
model (`linear` = admitted as a linpred / dirs code 2; `hinge` = knot term). Rows: (a) ordinary stock earth, (b) automatic adaptive competition (no `linpreds`), (c) adaptive with `Girth` forced linear via `linpreds`.

| setting | dominant_form | insample_rsq | cv_rsq | cv_classrate | nterms |
| --- | --- | --- | --- | --- | --- |
| (a) ordinary (adaptive OFF) | hinge | 0.9742029 | 0.9090899 | NA | 4 |
| (b) automatic adaptive cap=0.5 | mixed | 0.9138529 | 0.8104416 | NA | 4 |
| (b) automatic adaptive cap=0.9 | hinge | 0.9603086 | 0.8931114 | NA | 4 |
| (c) forced-linpreds cap=0.5 | linear | 0.8972716 | 0.7801802 | NA | 3 |
| (c) forced-linpreds cap=0.9 | linear | 0.9480327 | 0.8460695 | NA | 3 |

**Form verdict (effect.cap = 0.5): YES - at effect.cap=0.5 the AUTOMATIC competition admitted `Girth` as a LINEAR term (dirs code 2), on its own, reproducing the Stage-1 forced-linpreds form. Ordinary earth entered it as a hinge.**

**OOS (earth built-in CV RSq) at effect.cap=0.5: ordinary 0.9091, automatic-adaptive 0.8104 (-0.09865 vs ordinary), forced-linpreds 0.7802 (-0.1289 vs ordinary).**

