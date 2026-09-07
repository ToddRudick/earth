# Adaptive GCV effect cap (Stage 2): mtcars

- Response formula: `mpg ~ .`
- Rows: 32
- earth `degree` = 1
- Task type: regression
- Dominant predictor studied: `disp`
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

> Does the AUTOMATIC competition (no `linpreds`) admit `disp` as a LINEAR term on its own, reproducing the Stage-1 forced-linpreds form, and does that help or at least not hurt OOS performance versus ordinary earth?

## Three-way comparison: ordinary vs automatic-adaptive vs forced-linpreds

`dominant_form` is how the dominant predictor entered the forward-pass
model (`linear` = admitted as a linpred / dirs code 2; `hinge` = knot term). Rows: (a) ordinary stock earth, (b) automatic adaptive competition (no `linpreds`), (c) adaptive with `disp` forced linear via `linpreds`.

| setting | dominant_form | insample_rsq | cv_rsq | cv_classrate | nterms |
| --- | --- | --- | --- | --- | --- |
| (a) ordinary (adaptive OFF) | hinge | 0.8601598 | 0.6484781 | NA | 3 |
| (b) automatic adaptive cap=0.5 | hinge | 0.7634810 | 0.6593851 | NA | 3 |
| (b) automatic adaptive cap=0.9 | hinge | 0.8485889 | 0.6860748 | NA | 3 |
| (c) forced-linpreds cap=0.5 | absent | 0.8050519 | 0.6408882 | NA | 5 |
| (c) forced-linpreds cap=0.9 | absent | 0.8910223 | 0.6824443 | NA | 5 |

**Form verdict (effect.cap = 0.5): NO - at effect.cap=0.5 the automatic competition kept `disp` as a hinge form (the same shape ordinary earth used: hinge); the signal is not preferred as a plain linear term here, so automatic competition does not diverge from stock for this predictor.**

**OOS (earth built-in CV RSq) at effect.cap=0.5: ordinary 0.6485, automatic-adaptive 0.6594 (+0.01091 vs ordinary), forced-linpreds 0.6409 (-0.00759 vs ordinary).**

