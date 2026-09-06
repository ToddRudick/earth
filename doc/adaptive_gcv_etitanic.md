# Adaptive GCV effect cap (Stage 1): etitanic

- Response formula: `survived ~ .`
- Rows: 1046
- earth `degree` = 2
- Task type: binary classification
- Dominant predictor studied (hinge vs forced linear): `age`
- OOS engine: earth built-in cross-validation, nfold = 5, ncross = 3, seed 2024

## Stage-1 cap semantics

The cap is GCV / per-term-complexity adaptive (not a fixed fraction of
variance).  A term's realised delta-RSS is limited to
`DeltaRssMax = BreakEven + slackFactor*(RssDelta - BreakEven)` where
`slackFactor = (1 - clamp(Cost1))^(1/effect.cap - 1)` and `Cost1` uses an
EXPLICIT per-term knot charge (0 knots for a linear/linpreds term, 1 knot
for a hinge term).  `effect.cap >= 1` reproduces stock earth exactly.

## Ordinary vs adaptive: in-sample and cross-validated fit

| setting | insample_rsq | cv_rsq | cv_classrate | nterms |
| --- | --- | --- | --- | --- |
| ordinary (OFF) | 0.4389834 | 0.4010213 | 0.7932107 | 8 |
| adaptive cap=0.5 | 0.4372861 | 0.4046046 | 0.7932107 | 8 |
| adaptive cap=0.9 | 0.4387942 | 0.4027005 | 0.7932107 | 8 |

## Hinge vs forced-linear contrast for `age` (effect.cap = 0.5)

For the dominant predictor we compare its natural HINGE representation
against the SAME predictor FORCED LINEAR via `linpreds`.  `deltaKnots`,
`Cost1`, `slackFactor`, the effect budget `dRSSmax`, and the applied
`CapScale` are read from the `trace >= 6` cap diagnostics for the
predictor's earliest saturated term; `retained_var` is the variance of
the fitted contribution attributable to the predictor; `cv_rsq` is the
earth built-in CV RSq of the whole model under each representation.

| representation | deltaKnots | Cost1 | slackFactor | dRSSmax_budget | CapScale | retained_var | cv_rsq |
| --- | --- | --- | --- | --- | --- | --- | --- |
| hinge | 0 | 0.0028681 | 0.99713 | 48.652 | 0.94645 | 0.01263872 | 0.4046046 |
| forced linear | 0 | 0.0038241 | 0.99618 | 35.463 | 0.93816 | 0.01585841 | 0.3960685 |

**Divergence verdict: both representations charged the same per-term knot cost (deltaKnots=0); the forced-linear term did not reduce the knot charge here (the predictor's earliest saturated term was already linear), so no divergence is expected.**

