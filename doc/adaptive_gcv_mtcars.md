# Adaptive GCV effect cap (Stage 1): mtcars

- Response formula: `mpg ~ .`
- Rows: 32
- earth `degree` = 1
- Task type: regression
- Dominant predictor studied (hinge vs forced linear): `disp`
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
| ordinary (OFF) | 0.8601598 | 0.6484781 | NA | 3 |
| adaptive cap=0.5 | 0.7634810 | 0.6577238 | NA | 3 |
| adaptive cap=0.9 | 0.8485889 | 0.6860748 | NA | 3 |

## Hinge vs forced-linear contrast for `disp` (effect.cap = 0.5)

For the dominant predictor we compare its natural HINGE representation
against the SAME predictor FORCED LINEAR via `linpreds`.  `deltaKnots`,
`Cost1`, `slackFactor`, the effect budget `dRSSmax`, and the applied
`CapScale` are read from the `trace >= 6` cap diagnostics for the
predictor's earliest saturated term; `retained_var` is the variance of
the fitted contribution attributable to the predictor; `cv_rsq` is the
earth built-in CV RSq of the whole model under each representation.

| representation | deltaKnots | Cost1 | slackFactor | dRSSmax_budget | CapScale | retained_var | cv_rsq |
| --- | --- | --- | --- | --- | --- | --- | --- |
| hinge |  1 | 0.15625 | 0.84375 | 23.668 | 0.66474 | 13.80649 | 0.6577238 |
| forced linear | NA | NA | NA | NA | NA |  0.00000 | 0.6584623 |

**Divergence verdict: the dominant predictor's term was NOT saturated in one representation (its cap did not bind at effect.cap=0.5); no divergence is expected there.**

