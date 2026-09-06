# Adaptive GCV effect cap (Stage 2): etitanic

- Response formula: `survived ~ .`
- Rows: 1046
- earth `degree` = 2
- Task type: binary classification
- Dominant predictor studied: `age`
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

> Does the AUTOMATIC competition (no `linpreds`) admit `age` as a LINEAR term on its own, reproducing the Stage-1 forced-linpreds form, and does that help or at least not hurt OOS performance versus ordinary earth?

## Three-way comparison: ordinary vs automatic-adaptive vs forced-linpreds

`dominant_form` is how the dominant predictor entered the forward-pass
model (`linear` = admitted as a linpred / dirs code 2; `hinge` = knot term). Rows: (a) ordinary stock earth, (b) automatic adaptive competition (no `linpreds`), (c) adaptive with `age` forced linear via `linpreds`.

| setting | dominant_form | insample_rsq | cv_rsq | cv_classrate | nterms |
| --- | --- | --- | --- | --- | --- |
| (a) ordinary (adaptive OFF) | hinge | 0.4389834 | 0.4010213 | 0.7932107 | 8 |
| (b) automatic adaptive cap=0.5 | hinge | 0.4372861 | 0.4046835 | 0.7941631 | 8 |
| (b) automatic adaptive cap=0.9 | hinge | 0.4387942 | 0.4027005 | 0.7932107 | 8 |
| (c) forced-linpreds cap=0.5 | linear | 0.4229633 | 0.3960685 | 0.7871227 | 9 |
| (c) forced-linpreds cap=0.9 | linear | 0.4240744 | 0.3953375 | 0.7871227 | 9 |

**Form verdict (effect.cap = 0.5): NO - at effect.cap=0.5 the automatic competition kept `age` as a hinge form (the same shape ordinary earth used: hinge); the signal is not preferred as a plain linear term here, so automatic competition does not diverge from stock for this predictor.**

**OOS (earth built-in CV RSq) at effect.cap=0.5: ordinary 0.401, automatic-adaptive 0.4047 (+0.003662 vs ordinary), forced-linpreds 0.3961 (-0.004953 vs ordinary).**

