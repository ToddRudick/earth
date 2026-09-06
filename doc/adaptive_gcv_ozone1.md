# Adaptive GCV effect cap (Stage 1): ozone1

- Response formula: `O3 ~ .`
- Rows: 330
- earth `degree` = 2
- Task type: regression
- Dominant predictor studied (hinge vs forced linear): `temp`
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
| ordinary (OFF) | 0.8253857 | 0.7279271 | NA | 12 |
| adaptive cap=0.5 | 0.8081294 | 0.7333624 | NA | 12 |
| adaptive cap=0.9 | 0.8234371 | 0.7350913 | NA | 12 |

## Hinge vs forced-linear contrast for `temp` (effect.cap = 0.5)

For the dominant predictor we compare its natural HINGE representation
against the SAME predictor FORCED LINEAR via `linpreds`.  `deltaKnots`,
`Cost1`, `slackFactor`, the effect budget `dRSSmax`, and the applied
`CapScale` are read from the `trace >= 6` cap diagnostics for the
predictor's earliest saturated term; `retained_var` is the variance of
the fitted contribution attributable to the predictor; `cv_rsq` is the
earth built-in CV RSq of the whole model under each representation.

| representation | deltaKnots | Cost1 | slackFactor | dRSSmax_budget | CapScale | retained_var | cv_rsq |
| --- | --- | --- | --- | --- | --- | --- | --- |
| hinge | 1 | 0.0181820 | 0.98182 | 213.33 | 0.86828 | 15.47426 | 0.7333624 |
| forced linear | 0 | 0.0060606 | 0.99394 | 199.32 | 0.92254 | 13.49262 | 0.7306245 |

**Divergence verdict: YES - the cheaper LINEAR form (0 knots) carries a lower Cost1, a larger slackFactor and a larger CapScale (shrunk less) than the HINGE form (1 knot), exactly as the per-term knot charge predicts.**

