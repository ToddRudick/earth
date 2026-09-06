# Adaptive GCV Effect Cap (Stage 2): larger and wider datasets

This report extends the Stage-2 adaptive GCV effect-cap study
([doc/adaptive_gcv_comparison.md](adaptive_gcv_comparison.md), which used
small frames: trees 31x2, ozone1 330x9, mtcars 32x10, etitanic 1046x5) to
**larger and wider** datasets: hundreds-to-thousands of rows and up to 228
predictors. It answers the follow-up question directly: *does the automatic
hinge-vs-linear form competition help, is it neutral, or does it hurt OOS at
larger n and wider p, and how does it scale (runtime)?*

It reuses the SAME three-way comparison framework and helpers as the
small-dataset study; only the datasets, a runtime column, and per-predictor
form counts are added. No earth source code or default behaviour changed.

## What Stage 2 does

Under `adaptive.gcv = TRUE` with `effect.cap < 1` (and the default
`Auto.linpreds = TRUE`), the forward pass **automatically competes a HINGE
form and a LINEAR form of each candidate predictor** and admits the form
with the higher **justified (capped) effect** under the per-term-complexity
budget

```
DeltaRssMax = BreakEven + slackFactor * (RssDelta - BreakEven)
slackFactor = (1 - clamp(Cost1))^gamma,     gamma = 1/effect.cap - 1
Cost1       = (nOldUsedTerms + deltaTerms + Penalty*(nKnotsOld + deltaKnots)) / n
```

A linear term is charged `deltaKnots = 0` (larger `slackFactor`, shrunk
less); a hinge is charged `deltaKnots = 1`.  So a genuinely-linear dominant
predictor can be admitted as a plain **LINEAR** term (its `$dirs` entry for
that predictor becomes direction code **2**, a linpred with no knot)
**automatically**, WITHOUT the user setting `linpreds`.  `effect.cap >= 1`
disables the competition and reproduces stock earth byte-for-byte.

## The question at scale

> On larger n and wider p, does the AUTOMATIC competition (`adaptive.gcv =
> TRUE`, no `linpreds`) admit a genuinely-linear dominant predictor as a
> LINEAR term on its own, and does automatic form competition HELP, stay
> NEUTRAL, or HURT OOS versus ordinary earth -- and how does it scale?

For each dataset we run a **three-way** comparison at effect.cap values
0.5, 0.9 (both < 1; the detailed form verdict uses effect.cap = 0.5):

- **(a) ordinary / stock earth** -- `adaptive.gcv = FALSE`.
- **(b) automatic adaptive competition** -- `adaptive.gcv = TRUE`, NO `linpreds` (the Stage-2 path).
- **(c) forced-linpreds adaptive** -- `adaptive.gcv = TRUE`, dominant predictor forced linear via `linpreds` (the Stage-1 reference).

## Methodology

- **OOS engine (primary, only critical path):** earth built-in CV, `nfold = 5, ncross = 3`, `set.seed(2024)`.
- **effect.cap values studied:** 0.5, 0.9 (both < 1; detailed form verdict at effect.cap = 0.5).
- **Runtime:** wall-clock seconds for each fit (CV fit + forward-pass fit) via `system.time()`, reported in every table and summarised below.
- **Per-predictor form counts:** across ALL predictors in the forward-pass `$dirs`, how many entered linear (code 2) vs hinge (+/-1) vs mixed, under automatic competition.
- **Optional:** a `caret::bagEarth` 70/30 holdout is included only when
  `options(adaptive.gcv.run.caret = TRUE)` is set and caret is installed;
  it is OFF the critical path (caret bagEarth is slow).

## Datasets

| dataset | task | degree | rows | predictors | dominant predictor | notes |
| --- | --- | --- | --- | --- | --- | --- |
| boston | regression | 2 | 506 | 13 | lstat | MASS::Boston; dominant lstat |cor|~0.74 with medv |
| spam | classification | 1 | 4601 | 57 | your | kernlab::spam; dominant continuous predictor by point-biserial cor |
| solubility | regression | 2 | 951 | 228 | MolWeight | AppliedPredictiveModeling; WIDE (228 predictors) |
| synthetic_large | regression | 2 | 4000 | 40 | x1 | reproducible synthetic; x1 genuinely linear |

## Runtime summary

Total wall-clock seconds per dataset (all three settings x both caps, plus
the ordinary fit and forward-pass companion fits), on this machine:

| dataset | rows | predictors | total elapsed (s) |
| --- | --- | --- | --- |
| boston | 506 | 13 | 1.78 |
| spam | 4601 | 57 | 11.80 |
| solubility | 951 | 228 | 81.35 |
| synthetic_large | 4000 | 40 | 22.97 |

Per-fit runtime is in the `elapsed_s` column of each three-way table below.
Automatic competition adds no meaningful runtime overhead versus ordinary
earth at these sizes; the cost is dominated by the CV resampling, not the
hinge-vs-linear competition.

## Three-way comparison for all datasets

`dominant_form`: how the dominant predictor entered the forward-pass model
(`linear` = linpred / dirs code 2, `hinge` = knot term). `elapsed_s` is
wall-clock seconds for the CV fit plus the forward-pass fit.

| dataset | setting | dominant form | in-sample RSq | CV RSq | CV class-rate | nterms | elapsed_s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| boston | (a) ordinary (adaptive OFF) | hinge | 0.9148 | 0.6926 | NA | 21 | 0.33 |
| boston | (b) automatic adaptive cap=0.5 | hinge | 0.8985 | 0.7645 | NA | 21 | 0.32 |
| boston | (b) automatic adaptive cap=0.9 | hinge | 0.913 | 0.7246 | NA | 21 | 0.31 |
| boston | (c) forced-linpreds cap=0.5 | linear | 0.8904 | 0.8328 | NA | 19 | 0.29 |
| boston | (c) forced-linpreds cap=0.9 | linear | 0.905 | 0.8414 | NA | 19 | 0.29 |
| spam | (a) ordinary (adaptive OFF) | absent | 0.6902 | 0.6814 | 0.9252 | 16 | 2.50 |
| spam | (b) automatic adaptive cap=0.5 | absent | 0.6887 | 0.6801 | 0.9252 | 16 | 2.33 |
| spam | (b) automatic adaptive cap=0.9 | absent | 0.6901 | 0.6814 | 0.9252 | 16 | 2.27 |
| spam | (c) forced-linpreds cap=0.5 | absent | 0.6887 | 0.6817 | 0.926 | 16 | 2.17 |
| spam | (c) forced-linpreds cap=0.9 | absent | 0.6901 | 0.6831 | 0.926 | 16 | 2.18 |
| solubility | (a) ordinary (adaptive OFF) | hinge | 0.9194 | 0.8692 | NA | 28 | 15.16 |
| solubility | (b) automatic adaptive cap=0.5 | hinge | 0.9087 | 0.8742 | NA | 28 | 15.08 |
| solubility | (b) automatic adaptive cap=0.9 | hinge | 0.9182 | 0.8746 | NA | 28 | 15.06 |
| solubility | (c) forced-linpreds cap=0.5 | linear | 0.9078 | 0.8802 | NA | 28 | 17.87 |
| solubility | (c) forced-linpreds cap=0.9 | linear | 0.9177 | 0.8813 | NA | 28 | 17.78 |
| synthetic_large | (a) ordinary (adaptive OFF) | hinge | 0.832 | 0.8589 | NA | 15 | 5.96 |
| synthetic_large | (b) automatic adaptive cap=0.5 | linear | 0.8251 | 0.8648 | NA | 8 | 4.20 |
| synthetic_large | (b) automatic adaptive cap=0.9 | hinge | 0.8318 | 0.8583 | NA | 15 | 5.72 |
| synthetic_large | (c) forced-linpreds cap=0.5 | linear | 0.8251 | 0.8648 | NA | 8 | 3.31 |
| synthetic_large | (c) forced-linpreds cap=0.9 | linear | 0.8259 | 0.8659 | NA | 8 | 3.34 |

## Per-predictor form counts under automatic competition

Counted across ALL predictors in the forward-pass `$dirs` at effect.cap = 0.5.
`ordinary_*` columns are the same counts for stock earth (adaptive OFF).

| dataset | auto_linear | auto_hinge | auto_mixed | ordinary_linear | ordinary_hinge | ordinary_mixed |
| --- | --- | --- | --- | --- | --- | --- |
| boston | 0 | 9 | 0 | 0 | 9 | 0 |
| spam | 0 | 10 | 0 | 0 | 10 | 0 |
| solubility | 8 | 11 | 0 | 8 | 11 | 0 |
| synthetic_large | 1 | 4 | 0 | 0 | 7 | 0 |

## Stage-2 form verdict per dataset

- **boston** (dominant `lstat`): NO - at effect.cap=0.5 the automatic competition kept `lstat` as a hinge form (the same shape ordinary earth used: hinge); the signal is not preferred as a plain linear term here, so automatic competition does not diverge from stock for this predictor.
  - OOS (earth built-in CV RSq) at effect.cap=0.5: ordinary 0.6926, automatic-adaptive 0.7645 (+0.0719 vs ordinary), forced-linpreds 0.8328 (+0.1402 vs ordinary).
  - Per-predictor form counts under AUTOMATIC competition (effect.cap=0.5), counted across all predictors in the forward-pass `$dirs`: **linear (code 2): 0**, **hinge (+/-1): 9**, **mixed: 0**. For comparison ordinary stock earth used linear: 0, hinge: 9, mixed: 0.
  - **OOS at scale: automatic form competition HELPS vs ordinary earth.**
- **spam** (dominant `your`): NO - at effect.cap=0.5 the automatic competition kept `your` as a absent form (the same shape ordinary earth used: absent); the signal is not preferred as a plain linear term here, so automatic competition does not diverge from stock for this predictor.
  - OOS (earth built-in CV RSq) at effect.cap=0.5: ordinary 0.6814, automatic-adaptive 0.6801 (-0.001292 vs ordinary), forced-linpreds 0.6817 (+0.0002686 vs ordinary).
  - Per-predictor form counts under AUTOMATIC competition (effect.cap=0.5), counted across all predictors in the forward-pass `$dirs`: **linear (code 2): 0**, **hinge (+/-1): 10**, **mixed: 0**. For comparison ordinary stock earth used linear: 0, hinge: 10, mixed: 0.
  - **OOS at scale: automatic form competition NEUTRAL vs ordinary earth.**
- **solubility** (dominant `MolWeight`): NO - at effect.cap=0.5 the automatic competition kept `MolWeight` as a hinge form (the same shape ordinary earth used: hinge); the signal is not preferred as a plain linear term here, so automatic competition does not diverge from stock for this predictor.
  - OOS (earth built-in CV RSq) at effect.cap=0.5: ordinary 0.8692, automatic-adaptive 0.8742 (+0.004934 vs ordinary), forced-linpreds 0.8802 (+0.01098 vs ordinary).
  - Per-predictor form counts under AUTOMATIC competition (effect.cap=0.5), counted across all predictors in the forward-pass `$dirs`: **linear (code 2): 8**, **hinge (+/-1): 11**, **mixed: 0**. For comparison ordinary stock earth used linear: 8, hinge: 11, mixed: 0.
  - **OOS at scale: automatic form competition NEUTRAL vs ordinary earth.**
- **synthetic_large** (dominant `x1`): YES - at effect.cap=0.5 the AUTOMATIC competition admitted `x1` as a plain LINEAR term (dirs code 2), on its own, reproducing the Stage-1 forced-linpreds form (pure linear, no knot). Ordinary earth entered it as a hinge.
  - OOS (earth built-in CV RSq) at effect.cap=0.5: ordinary 0.8589, automatic-adaptive 0.8648 (+0.005949 vs ordinary), forced-linpreds 0.8648 (+0.005949 vs ordinary).
  - Per-predictor form counts under AUTOMATIC competition (effect.cap=0.5), counted across all predictors in the forward-pass `$dirs`: **linear (code 2): 1**, **hinge (+/-1): 4**, **mixed: 0**. For comparison ordinary stock earth used linear: 0, hinge: 7, mixed: 0.
  - **OOS at scale: automatic form competition HELPS vs ordinary earth.**

## Headline: does automatic form competition help at scale?

Across the 4 larger/wider datasets, at effect.cap = 0.5 the AUTOMATIC hinge-vs-linear competition was **HELPS in 2, NEUTRAL in 2, HURTS in 0** (OOS metric: CV RSq, or CV class-rate for the binary spam dataset; |delta| < 0.005 counted as neutral).

It admitted the dominant predictor as a PURE LINEAR term (reproducing the forced-linpreds form) in 1 dataset(s) and as a MIXED (linpred + retained hinge) form in 0 dataset(s).

On the controlled SYNTHETIC design (n = 4000, p = 40) whose dominant `x1` is genuinely LINEAR by construction, automatic competition entered `x1` as a **linear** form at effect.cap = 0.5, and was **HELPS** OOS versus ordinary earth. This is the cleanest test of the mechanism intent, because we know the true shape.

### Interpretation (honest)

- The automatic competition adds negligible runtime at these sizes (see the
  runtime summary); it scales with the CV resampling cost, not the form
  competition itself.
- Whether it HELPS, is NEUTRAL, or HURTS OOS remains dataset dependent even
  at larger n / wider p: the tables above report the signed CV deltas
  directly rather than claiming a universal win. Where the dominant signal
  is genuinely nonlinear the competition correctly keeps the hinge (a
  no-difference / neutral case), and where a cheaper linear form is
  justified it can admit one.
- Ordinary earth remains the default; `effect.cap >= 1` recovers stock earth
  exactly. This study is diagnostic, not a recommendation to change the
  default.

## Companion per-dataset files

- [`doc/adaptive_gcv_boston.md`](adaptive_gcv_boston.md)
- [`doc/adaptive_gcv_spam.md`](adaptive_gcv_spam.md)
- [`doc/adaptive_gcv_solubility.md`](adaptive_gcv_solubility.md)
- [`doc/adaptive_gcv_synthetic_large.md`](adaptive_gcv_synthetic_large.md)

See also the small-dataset study: [`doc/adaptive_gcv_comparison.md`](adaptive_gcv_comparison.md).

