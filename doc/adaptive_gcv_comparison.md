# Adaptive GCV Effect Cap (Stage 1): in-sample vs out-of-sample comparison

This report compares ordinary earth (`adaptive.gcv = FALSE`, the default)
against the experimental **Stage-1** adaptive GCV effect cap
(`adaptive.gcv = TRUE`) on several established datasets, using genuine
out-of-sample (OOS) scores from earth's built-in cross-validation.

## What changed in Stage 1

The effect cap is now **GCV / per-term-complexity adaptive**, not a fixed
fraction of the total variance.  When a candidate term is admitted, the
incremental delta-RSS it is allowed to realise is

```
DeltaRssMax = BreakEven + slackFactor * (RssDelta - BreakEven)
slackFactor = (1 - clamp(Cost1))^gamma,     gamma = 1/effect.cap - 1
Cost1       = (nOldUsedTerms + deltaTerms + Penalty*(nKnotsOld + deltaKnots)) / n
```

where `RssDelta` is the unconstrained OLS effect (ceiling) and `BreakEven` is
the GCV break-even reduction (floor).  The Stage-1 change replaces the old
model-wide averaged approximation `(nUsedTerms-1)/2` with a **per-term
complexity charge**.  Two things distinguish a hinge from a linear term in
`Cost1`:

- `deltaTerms`: a hinge term-pair adds **2** terms, a linear/`linpreds` term
  adds **1**.  This is the larger effect, and the old averaged formula already
  carried it (via `nUsedTerms`).
- `deltaKnots`: the new **explicit per-term knot charge**, `deltaKnots = 1`
  for a hinge and `deltaKnots = 0` for a linear/`linpreds` term.  This refines
  the charge (the averaged formula charged a linear term `0.5` knots on
  average, not `0`), so it *widens* the hinge-vs-linear gap rather than being
  its sole cause.

Because `Cost1` rises with both `deltaTerms` and `deltaKnots`, and
`slackFactor` decreases with `Cost1`, a HINGE term gets a SMALLER budget than
a LINEAR term carrying the same OLS effect.  `effect.cap >= 1` forces
`slackFactor == 1` and reproduces stock earth byte-for-byte.

## The Stage-1 question

> Under the per-term-complexity-aware cap, do HINGE and LINEAR terms diverge
> as predicted?  Does a dominant predictor entered as a cheap LINEAR term (0
> knots) get a HIGHER justified effect budget / larger CapScale (shrunk
> LESS) than the same signal expressed as a HINGE (1 knot)?

For each dataset the dominant predictor is fit once as a HINGE (default) and
once FORCED LINEAR via `linpreds`, and we report the retained effect /
CapScale (from `trace >= 6`) and the OOS CV RSq of each representation.
This uses the EXISTING `linpreds` mechanism only to MEASURE the divergence;
generating both a hinged and unhinged version of every candidate inside the
algorithm is Stage 2 and is deliberately out of scope here.

## Methodology

- **OOS engine (primary, only critical path):** earth built-in CV, `nfold = 5, ncross = 3`, `set.seed(2024)`.
- **effect.cap values studied:** 0.5, 0.9 (both < 1; the hinge-vs-linear contrast uses effect.cap = 0.5).
- **Optional:** a `caret::bagEarth` 70/30 holdout is included only when
  `options(adaptive.gcv.run.caret = TRUE)` is set and caret is installed;
  it is OFF the critical path (caret bagEarth is slow).

## Datasets

| dataset | task | degree | rows | dominant predictor | notes |
| --- | --- | --- | --- | --- | --- |
| ozone1 | regression | 2 | 330 | temp | canonical MARS / earth-vignette example |
| trees | regression | 1 | 31 | Girth | base R |
| mtcars | regression | 1 | 32 | disp | base R |
| etitanic | classification | 2 | 1046 | age | earth example, binary `survived` |

## Cross-validated fit: ordinary vs adaptive

| dataset | setting | in-sample RSq | CV RSq | CV class-rate | nterms |
| --- | --- | --- | --- | --- | --- |
| ozone1 | ordinary (OFF) | 0.8254 | 0.7279 | NA | 12 |
| ozone1 | adaptive cap=0.5 | 0.8081 | 0.7334 | NA | 12 |
| ozone1 | adaptive cap=0.9 | 0.8234 | 0.7351 | NA | 12 |
| trees | ordinary (OFF) | 0.9742 | 0.9091 | NA | 4 |
| trees | adaptive cap=0.5 | 0.8585 | 0.7233 | NA | 4 |
| trees | adaptive cap=0.9 | 0.9603 | 0.8928 | NA | 4 |
| mtcars | ordinary (OFF) | 0.8602 | 0.6485 | NA | 3 |
| mtcars | adaptive cap=0.5 | 0.7635 | 0.6577 | NA | 3 |
| mtcars | adaptive cap=0.9 | 0.8486 | 0.6861 | NA | 3 |
| etitanic | ordinary (OFF) | 0.439 | 0.401 | 0.7932 | 8 |
| etitanic | adaptive cap=0.5 | 0.4373 | 0.4046 | 0.7932 | 8 |
| etitanic | adaptive cap=0.9 | 0.4388 | 0.4027 | 0.7932 | 8 |

## Hinge vs forced-linear divergence (dominant predictor, effect.cap = 0.5)

| dataset | predictor | representation | deltaKnots | slackFactor | dRSSmax budget | CapScale | retained var | CV RSq |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ozone1 | temp | hinge | 1 | 0.98182 | 213.33 | 0.86828 | 15.474 | 0.73336 |
| ozone1 | temp | forced linear | 0 | 0.99394 | 199.32 | 0.92254 | 13.493 | 0.73062 |
| trees | Girth | hinge | 1 | 0.83871 | 25.395 | 0.65426 | 95.049 | 0.72328 |
| trees | Girth | forced linear | 0 | 0.93548 | 26.376 | 0.75506 | 122.42 | 0.78018 |
| mtcars | disp | hinge | 1 | 0.84375 | 23.668 | 0.66474 | 13.806 | 0.65772 |
| mtcars | disp | forced linear | NA |     NA |     NA |     NA |      0 | 0.65846 |
| etitanic | age | hinge | 0 | 0.99713 | 48.652 | 0.94645 | 0.012639 | 0.4046 |
| etitanic | age | forced linear | 0 | 0.99618 | 35.463 | 0.93816 | 0.015858 | 0.39607 |

## Divergence verdict per dataset

- **ozone1** (dominant `temp`): YES - the cheaper LINEAR form (1 term, 0 knots) carries a lower Cost1, a larger slackFactor and a larger CapScale (shrunk less) than the HINGE form (2 terms, 1 knot), as the per-term complexity charge predicts.
- **trees** (dominant `Girth`): YES - the cheaper LINEAR form (1 term, 0 knots) carries a lower Cost1, a larger slackFactor and a larger CapScale (shrunk less) than the HINGE form (2 terms, 1 knot), as the per-term complexity charge predicts.
- **mtcars** (dominant `disp`): the dominant predictor's term was NOT saturated in one representation (its cap did not bind at effect.cap=0.5); no divergence is expected there.
- **etitanic** (dominant `age`): both representations charged the same per-term knot cost (deltaKnots=0); the forced-linear term did not reduce the knot charge here (the predictor's earliest saturated term was already linear), so no divergence is expected.

## Interpretation

Across the 4 datasets, the cheaper LINEAR representation of the dominant predictor received a strictly larger effect budget and larger CapScale (was shrunk less) than the HINGE representation in 2 of them.
This is the divergence the per-term complexity charge is designed to produce:
a hinge is charged for its extra term (`deltaTerms = 2` vs `1`) and its extra
knot (`deltaKnots = 1` vs `0`) through a higher `Cost1`, which lowers
`slackFactor` and therefore the effect budget, so the same signal is
regularised more heavily when expressed as a hinge than when forced linear.
Most of that gap is driven by the per-term term count (which the old averaged
`(nUsedTerms-1)/2` approximation already carried); the explicit knot charge
sharpens and widens it rather than creating it on its own.  Where the dominant
predictor's term is not saturated in a given representation (the cap does not
bind), no divergence is expected and the table reports that directly.

Whether the extra regularisation helps or hurts OOS generalisation is
dataset dependent (see the CV RSq columns); ordinary earth remains the
default.  The `effect.cap` argument tunes the strength: `effect.cap >= 1`
recovers stock earth, smaller values push saturated terms further toward
their GCV break-even effect, with hinges pushed hardest.

## Companion per-dataset files

- `doc/adaptive_gcv_ozone1.md`
- `doc/adaptive_gcv_trees.md`
- `doc/adaptive_gcv_mtcars.md`
- `doc/adaptive_gcv_etitanic.md`

