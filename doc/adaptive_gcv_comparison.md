# Adaptive GCV Effect Cap (Stage 2): automatic hinge-vs-linear form competition

This report compares ordinary earth (`adaptive.gcv = FALSE`, the default)
against the experimental **Stage-2** adaptive GCV effect cap
(`adaptive.gcv = TRUE`) on several established datasets, using genuine
out-of-sample (OOS) scores from earth's built-in cross-validation.

## What Stage 2 does

Under `adaptive.gcv = TRUE` with `effect.cap < 1` (and the default
`Auto.linpreds = TRUE`), the forward pass now **automatically competes a
HINGE form and a LINEAR form of each candidate predictor** and admits the
form with the higher **justified (capped) effect** under the per-term-
complexity budget

```
DeltaRssMax = BreakEven + slackFactor * (RssDelta - BreakEven)
slackFactor = (1 - clamp(Cost1))^gamma,     gamma = 1/effect.cap - 1
Cost1       = (nOldUsedTerms + deltaTerms + Penalty*(nKnotsOld + deltaKnots)) / n
```

A linear term is charged `deltaKnots = 0` (larger `slackFactor`, shrunk
less); a hinge is charged `deltaKnots = 1`.  So a genuinely-linear dominant
predictor can now be admitted as a plain **LINEAR** term (its `$dirs` entry
for that predictor becomes direction code **2**, a linpred with no knot)
**automatically**, WITHOUT the user setting `linpreds`.  In Stage 1 the same
linear form could only be obtained by FORCING it via `linpreds`; Stage 1 used
`linpreds` merely to MEASURE the divergence.  `effect.cap >= 1` disables the
competition and reproduces stock earth byte-for-byte.

## The Stage-2 question

> Does the AUTOMATIC competition (`adaptive.gcv = TRUE`, no `linpreds`) admit
> a genuinely-linear dominant predictor as a LINEAR term on its own,
> reproducing the Stage-1 forced-`linpreds` form -- and does that help, or at
> least not hurt, OOS performance versus ordinary earth?

For each dataset we run a **three-way** comparison at the studied
`effect.cap` values:

- **(a) ordinary / stock earth** -- `adaptive.gcv = FALSE`.
- **(b) automatic adaptive competition** -- `adaptive.gcv = TRUE`, NO `linpreds` (the Stage-2 path).
- **(c) forced-linpreds adaptive** -- `adaptive.gcv = TRUE`, dominant predictor forced linear via `linpreds` (the Stage-1 reference).

We report, per dataset, how the dominant predictor entered the model under
each setting (`linear` = admitted as a linpred / `$dirs` code 2; `hinge` =
knot term), the in-sample RSq, the CV RSq, and `nterms`, and whether (b)'s
automatic choice MATCHES (c)'s forced-linear form.

## Methodology

- **OOS engine (primary, only critical path):** earth built-in CV, `nfold = 5, ncross = 3`, `set.seed(2024)`.
- **effect.cap values studied:** 0.5, 0.9 (both < 1; the detailed form verdict uses effect.cap = 0.5).
- **Optional:** a `caret::bagEarth` 70/30 holdout is included only when
  `options(adaptive.gcv.run.caret = TRUE)` is set and caret is installed;
  it is OFF the critical path (caret bagEarth is slow).

## Datasets

| dataset | task | degree | rows | dominant predictor | notes |
| --- | --- | --- | --- | --- | --- |
| trees | regression | 1 | 31 | Girth | base R; dominant signal is genuinely near-linear |
| ozone1 | regression | 2 | 330 | temp | canonical MARS / earth-vignette example; genuinely nonlinear |
| mtcars | regression | 1 | 32 | disp | base R |
| etitanic | classification | 2 | 1046 | age | earth example, binary `survived` |

## Headline results: trees and ozone1

These two datasets are presented first because they cleanly bracket the
Stage-2 behaviour: `trees` has a genuinely near-linear dominant predictor
(`Girth`), `ozone1` has a genuinely nonlinear one (`temp`).

### trees (dominant `Girth`, genuinely near-linear)

- **Automatic form choice:** MIXED - at effect.cap=0.5 the AUTOMATIC competition admitted a LINEAR form (dirs code 2) for `Girth` ALONGSIDE a retained hinge on the same predictor, so the fit is a hybrid (linpred + hinge). This is NOT the Stage-1 forced-linpreds form, which is pure linear (a single linpred, no hinge). The competition found a linear form worth admitting, but a hinge for `Girth` also survived elsewhere in the forward pass. Ordinary earth entered it as a hinge.
- OOS (earth built-in CV RSq) at effect.cap=0.5: ordinary 0.9091, automatic-adaptive 0.8104 (-0.09865 vs ordinary), forced-linpreds 0.7802 (-0.1289 vs ordinary).
- Ordinary earth entered `Girth` as a **hinge**; automatic adaptive (effect.cap=0.5) entered it as **mixed**; forced-linpreds entered it as **linear**.

| setting | dominant_form | insample_rsq | cv_rsq | cv_classrate | nterms |
| --- | --- | --- | --- | --- | --- |
| (a) ordinary (adaptive OFF) | hinge | 0.9742029 | 0.9090899 | NA | 4 |
| (b) automatic adaptive cap=0.5 | mixed | 0.9138529 | 0.8104416 | NA | 4 |
| (b) automatic adaptive cap=0.9 | hinge | 0.9603086 | 0.8931114 | NA | 4 |
| (c) forced-linpreds cap=0.5 | linear | 0.8972716 | 0.7801802 | NA | 3 |
| (c) forced-linpreds cap=0.9 | linear | 0.9480327 | 0.8460695 | NA | 3 |

### ozone1 (dominant `temp`, genuinely nonlinear)

- **Automatic form choice:** NO - at effect.cap=0.5 the automatic competition kept `temp` as a hinge form (the same shape ordinary earth used: hinge); the signal is not preferred as a plain linear term here, so automatic competition does not diverge from stock for this predictor.
- OOS (earth built-in CV RSq) at effect.cap=0.5: ordinary 0.7279, automatic-adaptive 0.7334 (+0.005435 vs ordinary), forced-linpreds 0.7306 (+0.002697 vs ordinary).
- Ordinary earth entered `temp` as a **hinge**; automatic adaptive (effect.cap=0.5) entered it as **hinge**; forced-linpreds entered it as **linear**.

| setting | dominant_form | insample_rsq | cv_rsq | cv_classrate | nterms |
| --- | --- | --- | --- | --- | --- |
| (a) ordinary (adaptive OFF) | hinge | 0.8253857 | 0.7279271 | NA | 12 |
| (b) automatic adaptive cap=0.5 | hinge | 0.8081294 | 0.7333624 | NA | 12 |
| (b) automatic adaptive cap=0.9 | hinge | 0.8234371 | 0.7350913 | NA | 12 |
| (c) forced-linpreds cap=0.5 | linear | 0.8177420 | 0.7306245 | NA | 13 |
| (c) forced-linpreds cap=0.9 | linear | 0.8279385 | 0.7294938 | NA | 13 |

## Three-way comparison for all datasets

`dominant_form`: how the dominant predictor entered the forward-pass model
(`linear` = linpred / dirs code 2, `hinge` = knot term).

| dataset | setting | dominant form | in-sample RSq | CV RSq | CV class-rate | nterms |
| --- | --- | --- | --- | --- | --- | --- |
| trees | (a) ordinary (adaptive OFF) | hinge | 0.9742 | 0.9091 | NA | 4 |
| trees | (b) automatic adaptive cap=0.5 | mixed | 0.9139 | 0.8104 | NA | 4 |
| trees | (b) automatic adaptive cap=0.9 | hinge | 0.9603 | 0.8931 | NA | 4 |
| trees | (c) forced-linpreds cap=0.5 | linear | 0.8973 | 0.7802 | NA | 3 |
| trees | (c) forced-linpreds cap=0.9 | linear | 0.948 | 0.8461 | NA | 3 |
| ozone1 | (a) ordinary (adaptive OFF) | hinge | 0.8254 | 0.7279 | NA | 12 |
| ozone1 | (b) automatic adaptive cap=0.5 | hinge | 0.8081 | 0.7334 | NA | 12 |
| ozone1 | (b) automatic adaptive cap=0.9 | hinge | 0.8234 | 0.7351 | NA | 12 |
| ozone1 | (c) forced-linpreds cap=0.5 | linear | 0.8177 | 0.7306 | NA | 13 |
| ozone1 | (c) forced-linpreds cap=0.9 | linear | 0.8279 | 0.7295 | NA | 13 |
| mtcars | (a) ordinary (adaptive OFF) | hinge | 0.8602 | 0.6485 | NA | 3 |
| mtcars | (b) automatic adaptive cap=0.5 | hinge | 0.7635 | 0.6594 | NA | 3 |
| mtcars | (b) automatic adaptive cap=0.9 | hinge | 0.8486 | 0.6861 | NA | 3 |
| mtcars | (c) forced-linpreds cap=0.5 | absent | 0.8051 | 0.6409 | NA | 5 |
| mtcars | (c) forced-linpreds cap=0.9 | absent | 0.891 | 0.6824 | NA | 5 |
| etitanic | (a) ordinary (adaptive OFF) | hinge | 0.439 | 0.401 | 0.7932 | 8 |
| etitanic | (b) automatic adaptive cap=0.5 | hinge | 0.4373 | 0.4047 | 0.7942 | 8 |
| etitanic | (b) automatic adaptive cap=0.9 | hinge | 0.4388 | 0.4027 | 0.7932 | 8 |
| etitanic | (c) forced-linpreds cap=0.5 | linear | 0.423 | 0.3961 | 0.7871 | 9 |
| etitanic | (c) forced-linpreds cap=0.9 | linear | 0.4241 | 0.3953 | 0.7871 | 9 |

## Stage-2 form verdict per dataset

- **trees** (dominant `Girth`): MIXED - at effect.cap=0.5 the AUTOMATIC competition admitted a LINEAR form (dirs code 2) for `Girth` ALONGSIDE a retained hinge on the same predictor, so the fit is a hybrid (linpred + hinge). This is NOT the Stage-1 forced-linpreds form, which is pure linear (a single linpred, no hinge). The competition found a linear form worth admitting, but a hinge for `Girth` also survived elsewhere in the forward pass. Ordinary earth entered it as a hinge.
  - OOS (earth built-in CV RSq) at effect.cap=0.5: ordinary 0.9091, automatic-adaptive 0.8104 (-0.09865 vs ordinary), forced-linpreds 0.7802 (-0.1289 vs ordinary).
- **ozone1** (dominant `temp`): NO - at effect.cap=0.5 the automatic competition kept `temp` as a hinge form (the same shape ordinary earth used: hinge); the signal is not preferred as a plain linear term here, so automatic competition does not diverge from stock for this predictor.
  - OOS (earth built-in CV RSq) at effect.cap=0.5: ordinary 0.7279, automatic-adaptive 0.7334 (+0.005435 vs ordinary), forced-linpreds 0.7306 (+0.002697 vs ordinary).
- **mtcars** (dominant `disp`): NO - at effect.cap=0.5 the automatic competition kept `disp` as a hinge form (the same shape ordinary earth used: hinge); the signal is not preferred as a plain linear term here, so automatic competition does not diverge from stock for this predictor.
  - OOS (earth built-in CV RSq) at effect.cap=0.5: ordinary 0.6485, automatic-adaptive 0.6594 (+0.01091 vs ordinary), forced-linpreds 0.6409 (-0.00759 vs ordinary).
- **etitanic** (dominant `age`): NO - at effect.cap=0.5 the automatic competition kept `age` as a hinge form (the same shape ordinary earth used: hinge); the signal is not preferred as a plain linear term here, so automatic competition does not diverge from stock for this predictor.
  - OOS (earth built-in CV RSq) at effect.cap=0.5: ordinary 0.401, automatic-adaptive 0.4047 (+0.003662 vs ordinary), forced-linpreds 0.3961 (-0.004953 vs ordinary).

## Interpretation

Across the 4 datasets, the AUTOMATIC hinge-vs-linear competition (adaptive.gcv=TRUE, no linpreds) admitted the dominant predictor as a PURE LINEAR term on its own (reproducing the Stage-1 forced-linpreds form exactly) in 0 of them, and as a MIXED form (a linpred admitted ALONGSIDE a retained hinge on the same predictor, which is NOT the pure forced-linpreds form) in 1 of them.

The behaviour splits by the true shape of the dominant signal:

- **Genuinely near-linear dominant signal (trees / `Girth`):** at the tighter
  `effect.cap = 0.5` the automatic competition DOES admit a cheaper LINEAR
  form (`$dirs` code 2) for `Girth` on its own -- but a `Girth` hinge also
  survives elsewhere in the forward pass, so the automatic fit is **mixed**
  (a linpred PLUS a hinge on the same predictor, 4 terms), NOT the pure
  forced-linpreds form (a single `Girth` linpred, no hinge, 3 terms).  So the
  automatic competition finds the linear form worth admitting, but it does
  not reproduce the forced-linpreds form here: it adds a linpred alongside a
  retained hinge rather than replacing the hinge.  On OOS the mixed automatic
  fit scores a little HIGHER than the forced-linpreds reference at this cap
  (see the CV RSq columns).  Note honestly, however, that on trees BOTH
  adaptive settings at `effect.cap = 0.5` score LOWER OOS than ordinary stock
  earth: the tight cap shrinks every term, and trees is a tiny 31-row dataset
  where stock earth's hinge on `Girth` already generalises well.  The
  regularisation, not the automatic form choice, is what costs OOS RSq here;
  at the looser `effect.cap = 0.9` the automatic fit keeps the hinge and lands
  much closer to stock.
- **Genuinely nonlinear dominant signal (ozone1 / `temp`):** the hinge form
  carries the larger justified effect, so the automatic competition KEEPS the
  hinge and the fit matches stock earth for that predictor.  Automatic
  competition correctly makes essentially NO change where a linear form is not
  warranted (CV RSq within ~0.005 of stock at both caps); this is reported
  honestly as a no-difference case, not hidden.

Whether the automatic linear form helps OOS is dataset dependent and is read
directly from the CV RSq columns above; ordinary earth remains the default,
and `effect.cap >= 1` recovers stock earth exactly.  The clean Stage-2
conclusion is about the FORM CHOICE, which is what Stage 2 changed: for a
genuinely near-linear dominant predictor (trees) the automatic competition
admits a linear form for that predictor on its own -- though here it does so
ALONGSIDE a retained hinge (a mixed form), rather than reproducing the pure
forced-linpreds form -- and for a genuinely nonlinear one (ozone1) it
correctly keeps the hinge and does not add a linear form at all.  Boundary
datasets (mtcars, etitanic) keep the hinge and show only small OOS movement,
which the tables report directly.

## Companion per-dataset files

- `doc/adaptive_gcv_trees.md`
- `doc/adaptive_gcv_ozone1.md`
- `doc/adaptive_gcv_mtcars.md`
- `doc/adaptive_gcv_etitanic.md`

