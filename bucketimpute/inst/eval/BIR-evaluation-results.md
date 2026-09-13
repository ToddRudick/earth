# Bucketed Imputation Regression (BIR): out-of-sample evaluation

This document reports an **out-of-sample** evaluation of the full BIR pipeline
(the `bucketimpute` package) against a fair baseline on four datasets, plus a
sanity check of BIR's predictive uncertainty (`se`) and honest verdicts on where
BIR wins and loses. All numbers below are real measurements produced by the
scripts in this directory; none are placeholders.

Reproduce with (from the repo root, each dataset a separate timed invocation):

```sh
timeout  600 Rscript bucketimpute/inst/eval/eval_boston.R
timeout 1200 Rscript bucketimpute/inst/eval/eval_synthetic.R
timeout  300 Rscript bucketimpute/inst/eval/eval_spam.R
timeout  180 Rscript bucketimpute/inst/eval/eval_solubility.R smoke
timeout 1500 Rscript bucketimpute/inst/eval/eval_solubility.R full 1400
```

Fixed seed `set.seed(2024)` is used everywhere (splitting, generation, fitting).

## 1. Method summary (and the lars substitution)

BIR (see `bucketed-imputation-regression-spec.md`) works as follows:

1. Split the continuous response `y` into `n` equal-frequency quantile buckets
   (default `n = 10`).
2. In each bucket `k`, fit a degree-1 `earth` (MARS) model for every predictor
   column `j`, reconstructing column `j` from the other `p - 1` columns. Record
   the residual scale `s_{k,j} = sqrt(mean(resid^2))`.
3. For each row, per bucket, compute an *unusualness* score
   `u_k = sum_j |x_j - predict(model_{k,j})| / s_{k,j}` and convert it to an
   *affinity* `a_k = 1 / (1 + u_k)` in `(0, 1]`. This gives an `N x n` affinity
   matrix.
4. Fit a cross-validated lasso of `y` on the affinity features.
5. Predictive scale: normalize a row's affinities to weights `w_k`, then
   `se = sqrt(sum(w_k * bucket_vars[k]))`, where `bucket_vars[k]` is the
   within-bucket variance of the training `y`.

**Deliberate deviation from the spec: `lars` instead of `glmnet`.** The spec
calls for `cv.glmnet(..., alpha = 1)` with `lambda.min`. This implementation
instead fits `lars::lars(affinities, y, type = "lasso")` and selects the penalty
with `lars::cv.lars(..., mode = "fraction")`, choosing the fraction at the
**minimum** cross-validated MSE (`s_opt = cvobj$index[which.min(cvobj$cv)]`).
This is the `lars` analogue of glmnet's `lambda.min`: the net behaviour is a
CV-selected lasso matching the spec's intent; only the fitting engine changes.
Prediction uses `predict(fit$lasso, newx = affinities, s = fit$lasso_s,
mode = "fraction", type = "fit")$fit`.

Cost note: BIR fits `n * p` earth models **per fit**, so wide data is
expensive (see solubility below).

## 2. Evaluation protocol

- **Out-of-sample only.** The spec warns that training affinities are mildly
  optimistic, so BIR is always trained on a held-out training split and scored
  on a disjoint test split. In-sample scoring is never used.
- **Split.** 70/30 train/test, indices drawn with `set.seed(2024)`
  (`train_test_split()` in `eval_common.R`).
- **BIR config.** `fit_bir(x_train, y_train, n = 10)` (package defaults),
  predicted with `type = "all"` to obtain `fit` and `se`.
- **Baseline.** Plain degree-2 `earth` (MARS), `earth(x_train, y_train,
  degree = 2)` — a fair, widely used, strong off-the-shelf regressor, applied
  consistently across the regression datasets.
- **Metric.** Out-of-sample RMSE on the test split (spam is scored by RMSE on
  the numeric 0/1 encoding, per the approved treatment).
- **se calibration.** Spearman correlation of predicted `se` against actual
  absolute residual `|y - fit|` on the test split, plus mean `|resid|` per `se`
  quartile.

### synthetic_large generator (documented exactly)

`seed 2024, n = 4000, p = 40`:

```r
set.seed(2024)
X <- matrix(runif(n * p), n, p)          # all columns start as Uniform(0,1)
x1..x6 <- X[,1:6]                         # signal predictors, Uniform(0,1)
y <- 5*x1 + 4*pmax(x2 - 0.3, 0) + 3*pmax(0.5 - x3, 0) +
     2.5*x4*x5 + 1.2*x6 + rnorm(n, 0, 1)
X[,7] <- x1 + rnorm(n, 0, 0.25)          # redundant with x1
X[,8] <- x2 + rnorm(n, 0, 0.25)          # redundant with x2
# columns x9..x40 remain independent Uniform(0,1) NOISE (unrelated to y)
```

`pmax(., 0)` is the elementwise hinge. The generator lives in
`eval_synthetic.R::make_synthetic_large()`.

## 3. Out-of-sample RMSE: BIR vs baseline

| Dataset | Rows (train/test) | Predictors | BIR OOS RMSE | Baseline | Baseline OOS RMSE | Winner |
|---|---|---|---|---|---|---|
| Boston (`MASS::Boston`, `medv`) | 354 / 152 | 13 | **5.577** | earth (deg 2) | **3.802** | baseline |
| synthetic_large | 2800 / 1200 | 40 | **1.532** | earth (deg 2) | **1.057** | baseline |
| solubility (full, n=10) | 951 / 316 | 228 | **2.076** | earth (deg 2) | 55.13 | BIR (robustness only, see note) |
| solubility (smoke, 20 preds, n=4) — cost/pipeline probe, not a wide-data verdict | 951 / 316 | 20 (of 228) | **1.636** | earth (deg 2) | **1.630** | ~tie (probe only) |
| spam (`kernlab`, `type` as 0/1) | 3220 / — | 57 | did not run | earth (deg 2) | — | n/a (expected error) |

> **Reading the "Winner" column.** The solubility (full) "win" is **robustness,
> not accuracy**: BIR is not better than a working baseline, it is merely
> bounded while the naive baselines extrapolate catastrophically on this wide,
> collinear design (degree-2 earth RMSE ~55, a CV-lasso on the raw predictors
> ~293). See section 6 for the full explanation. The solubility (smoke) row is a
> **cost/pipeline probe** that subsets to 20 of the 228 predictors with a tiny
> bucket count; its 1.636-vs-1.630 near-tie is *not* a fair wide-data verdict
> and should not be read as one.

## 4. se calibration finding

Does a larger predicted `se` go with a larger actual absolute residual?

| Dataset | Spearman(se, \|resid\|) | Mean \|resid\| by se quartile (low -> high) |
|---|---|---|
| Boston | **+0.472** | 1.90, 2.89, 3.38, 7.62 |
| synthetic_large | +0.057 | 1.08, 1.30, 1.19, 1.33 |
| solubility (full) | +0.203 | 1.21, 1.94, 1.47, 1.97 |
| solubility (smoke) | +0.149 | 1.05, 1.05, 1.45, 1.53 |

**Verdict on uncertainty (secondary objective):** the `se` is *usefully but
weakly* calibrated. On Boston it is genuinely informative — the mean absolute
error rises monotonically from 1.9 in the lowest-`se` quartile to 7.6 in the
highest (Spearman +0.47). On solubility the top `se` quartile also carries the
largest errors. On synthetic_large the signal is essentially flat (+0.06): the
affinity-weighted bucket-variance `se` barely varies across rows because the
buckets have similar spread, so it carries little row-level discriminative
information there. Overall the `se` points in the right direction and never
inverts, but it should be read as a coarse, optimistic spread indicator rather
than a well-calibrated predictive interval.

## 5. Wall-clock cost

| Dataset / run | BIR fit+predict | Baseline | Notes |
|---|---|---|---|
| Boston | 0.5 s | 0.0 s | trivial |
| synthetic_large | 5.9 s | 0.3 s | 40 predictors x 10 buckets = 400 earth models |
| solubility smoke (20 preds, n=4) | 0.6 s | 0.0 s | 80 earth models |
| solubility full (228 preds, n=10) | **124.6 s** | 0.8 s | ~2280 earth models + affinity passes + cv.lars |
| spam | ~0.0 s | — | errors immediately at bucketing |

The full solubility fit was run under a hard `timeout` and an in-R
`setTimeLimit(elapsed = ...)` budget (see `eval_solubility.R`); it **completed**
in 124.6 s. A cost probe showed the full-width degree-1 earth models cost
~0.05 s each (~113 s for all 2280 models), so model fitting dominates the wall
clock; the affinity recompute and `cv.lars` add the remainder.

## 6. Honest verdicts

- **Boston — BIR loses.** BIR RMSE 5.58 vs earth 3.80. On a low-dimensional,
  well-behaved regression problem, plain MARS is clearly better; BIR's affinity
  compression discards predictive signal that degree-2 earth uses directly.
- **synthetic_large — BIR loses.** BIR RMSE 1.53 vs earth 1.06 (the irreducible
  noise floor is `sd = 1`). earth recovers the hinge/interaction structure and
  gets close to the noise floor; BIR's `n`-affinity representation cannot
  reconstruct the `2.5*x4*x5` interaction and the hinges as accurately, so it
  sits well above the floor. A clean loss on a problem MARS is built for.
- **solubility (WIDE) — BIR "wins", with an important caveat.** BIR is stable
  (RMSE ~2.08) while off-the-shelf degree-2 earth blows up to RMSE ~55 (its
  predictions extrapolate to the hundreds on the transformed, collinear 228-column
  design; a CV-lasso on the raw predictors extrapolates similarly badly, RMSE
  ~293). So BIR *beats the naive baselines here*, but the honest reading is that
  **the naive baselines are broken on this data, not that BIR is excellent**:
  BIR's affinities are bounded in `(0, 1]`, so its lasso cannot extrapolate
  wildly, which makes it robust rather than accurate. On a fair thin-slice
  (`smoke`, 20 predictors) BIR and earth are a near-tie (1.636 vs 1.630). Wide
  data is also where BIR is most expensive (125 s vs <1 s).
- **spam — did not run (expected).** `type` encoded as numeric 0/1 has only two
  distinct values, so equal-frequency quantile bucketing at `n >= 2` cannot form
  `n` distinct non-empty buckets. `fit_bir` raises its distinct-buckets error by
  design; the eval script captures it with `tryCatch` and reports it as an
  **expected, reportable outcome, not a bug**. Verbatim message:

  > Equal-frequency quantile bucketing of `y` did not yield 10 distinct
  > non-empty buckets (got 2). This happens when `y` has heavy ties or is
  > near-binary. Reduce `n` or supply a response with more distinct values;
  > buckets are not silently merged.

### Overall

On the two clean regression benchmarks where a fair, strong baseline exists
(Boston, synthetic_large), **BIR loses to plain degree-2 earth** on accuracy.
Its apparent solubility "win" is really robustness to a design that breaks the
naive baselines. BIR's secondary contribution — a predictive `se` — is weakly
but correctly ordered (best on Boston). BIR is also markedly more expensive than
MARS because it fits `n * p` earth models per fit, which makes wide data costly.
The method is most defensible as a *bounded, uncertainty-aware* regressor for
ill-scaled wide data, not as a general-purpose accuracy improvement over MARS.

## 7. Soft mixture-of-experts (moe_soft): four-way OOS comparison

This section augments (does not replace) the results above. It evaluates the
optional `moe_soft` prediction mode added to `bucketimpute` and answers three
questions per dataset:

- **(a)** Does `moe_soft` beat the old affinity-lasso BIR (the default method)?
- **(b)** Does it beat, or close the gap to, plain degree-2 earth?
- **(c)** Does it beat the affinity-weighted **bucket-mean floor** — i.e. is the
  per-bucket *regression* pulling its weight, or is the affinity gate doing all
  the work?

### Methods compared

1. **earth(deg2)** — plain degree-2 `earth` (MARS), the existing baseline
   (`run_earth`).
2. **affinity_lasso(old)** — the original BIR prediction,
   `predict(fit, x_test)` (default `method = "affinity_lasso"`).
3. **moe_soft** — `predict(fit, x_test, method = "moe_soft")`: the
   affinity-weighted blend of the per-bucket degree-2 earth experts,
   `yhat = sum_k w_k(x) * expert_k(x)`.
4. **bucket_mean_floor** — the affinity-weighted per-bucket **mean** of the
   *training* `y`, using the **identical** normalized affinity gate as
   `moe_soft`.

**Single-fit protocol.** For each dataset BIR is fit **once** with
`fit_bir(..., experts = TRUE)` (seed 2024, 70/30 split, `n = 10`), and methods
(2), (3) and (4) are all read off that one fitted model on the test split
(`run_bir_experts()` in `eval_common.R`). Because the fit and the
default-method prediction are produced under the same `set.seed(2024)` as the
no-experts `run_bir`, the `affinity_lasso(old)` numbers here are byte-for-byte
identical to section 3 (e.g. Boston 5.5766, synthetic_large 1.5321,
solubility-full 2.0764).

**How the floor reuses the identical gate.** `moe_soft` forms per-row weights
`w_k = a_k / sum_j a_j` from the affinity matrix (`predict(., type = "affinity")`),
forcing an all-zero-affinity row to a uniform `1/n` vector first (the
`.bir_moe_predict` all-zero guard). The floor recomputes weights from *that
same* affinity matrix with *that same* guard and normalization, then blends the
per-bucket training means `mbar_k = mean(y_train in bucket k)` (bucketing via
the model's own `fit$cuts`), giving `yhat_floor = sum_k w_k * mbar_k`. The
weights are therefore byte-identical to `moe_soft`; the **only** difference is
the per-bucket value blended (fitted earth expert vs. constant bucket mean). So
`moe_soft - floor` isolates exactly the contribution of the per-bucket
regression.

### OOS-R2 and RMSE (real measured numbers, seed 2024)

`R2 = 1 - SSE/SST` on the test set, using the test-set mean for SST. Higher R2
is better; lower RMSE is better.

| Dataset | Metric | earth(deg2) | affinity_lasso(old) | moe_soft | bucket_mean_floor |
|---|---|---|---|---|---|
| Boston (medv) | OOS-R2 | **0.8015** | 0.5730 | 0.5417 | 0.5061 |
| Boston (medv) | RMSE | **3.8020** | 5.5766 | 5.7777 | 5.9981 |
| synthetic_large | OOS-R2 | **0.7476** | 0.4697 | 0.1469 | 0.0698 |
| synthetic_large | RMSE | **1.0570** | 1.5321 | 1.9432 | 2.0291 |
| solubility (full, n=10, 228 preds) | OOS-R2 | -704.7335 | -0.0012 | **0.1444** | **0.1444** |
| solubility (full, n=10, 228 preds) | RMSE | 55.1285 | 2.0764 | **1.9195** | **1.9195** |
| solubility (smoke, 20 preds, n=4) — probe only | OOS-R2 | **0.3831** | 0.3789 | 0.2300 | 0.1760 |
| solubility (smoke, 20 preds, n=4) — probe only | RMSE | **1.6299** | 1.6355 | 1.8209 | 1.8837 |
| spam (`type` as 0/1) | — | did not run — expected distinct-buckets error (see below) | | | |

Wall-clock (BIR fit with experts + all three derived predictions): Boston
~0.6 s, synthetic_large ~6.8 s, solubility-smoke ~0.7 s, solubility-full
~127 s (vs ~120 s for the no-experts fit; the `n` extra degree-2 earth experts
add little here because on the wide design every expert falls back to a
constant — see below). Run under a hard `timeout` and in-R `setTimeLimit`.

### Per-dataset verdicts

- **Boston — moe_soft does NOT help.** (a) It is slightly *worse* than the old
  affinity-lasso BIR (R2 0.5417 vs 0.5730; RMSE 5.78 vs 5.58). (b) It gets
  nowhere near plain earth (R2 0.80). (c) It *does* beat the bucket-mean floor
  (0.5417 vs 0.5061), so the per-bucket regression adds a little over blending
  bucket means — but not enough to overtake the old lasso gate, which remains
  the best BIR variant here.
- **synthetic_large — moe_soft does NOT help; it is the worst BIR variant.**
  (a) Clearly worse than the old affinity-lasso BIR (R2 0.147 vs 0.470). (b)
  Far below plain earth (R2 0.748), which recovers the hinge/interaction
  structure near the noise floor. (c) It beats the bucket-mean floor only
  narrowly (0.147 vs 0.070), so the per-bucket regression buys a little, but the
  whole soft-blend approach discards most of the structured signal that both the
  old lasso and earth capture. A clean loss.
- **solubility (full, WIDE) — moe_soft helps vs the broken baselines and vs the
  old BIR, but the per-bucket regression contributes NOTHING.** (a) It beats the
  old affinity-lasso BIR (R2 0.1444 vs -0.0012; RMSE 1.92 vs 2.08). (b) It
  trivially beats plain earth, which extrapolates catastrophically on this
  collinear 228-column design (R2 -704, RMSE 55) — this is baseline breakage,
  not moe_soft excellence. (c) **moe_soft and the bucket-mean floor are
  identical to 4 decimals (0.1444 / 1.9195).** With 228 predictors and ~66-95
  rows per bucket, every bucket falls below the expert threshold
  (`expert_min_rows = 2*p = 456`), so *every* per-bucket earth expert falls back
  to its constant bucket mean. On wide data `moe_soft` therefore *is* the
  bucket-mean floor: the affinity gate is doing 100% of the work and the
  per-bucket regression pulls zero weight.
- **solubility (smoke, 20 preds, n=4) — probe only; moe_soft does NOT help.**
  (a) Worse than the old BIR (R2 0.230 vs 0.379). (b) Below plain earth (0.383).
  (c) It does beat the bucket-mean floor (0.230 vs 0.176), so here — where the
  bucket rows are large enough for real experts — the per-bucket regression does
  add value over bucket means, though still not enough to match the old lasso
  gate. This is a cost/pipeline probe on 20 of 228 predictors, not a wide-data
  verdict.
- **spam — did not run (expected).** `type` encoded as numeric 0/1 has only two
  distinct values, so equal-frequency quantile bucketing at `n >= 2` cannot form
  `n` distinct non-empty buckets; `fit_bir` raises its distinct-buckets error by
  design. **Enabling experts does not change this**: the error is raised at the
  bucketing step, before any expert would be fit. Reported as an expected,
  reportable outcome, not a bug.

### moe_soft overall

Across every dataset with a fair, working baseline (Boston, synthetic_large,
solubility-smoke), **`moe_soft` does not beat the old affinity-lasso BIR and
does not close the gap to plain degree-2 earth** — it is consistently the weaker
BIR variant. Its only "win" is on solubility-full, where it edges past the old
BIR *and* the broken baselines, but there it is provably identical to the
bucket-mean floor because the wide design forces every expert to its constant
fallback, so the affinity gate — not the per-bucket regression — is doing all
the work. Where buckets are wide enough for genuine experts (Boston,
solubility-smoke, and marginally synthetic_large), the per-bucket regression
*does* beat the bucket-mean floor, confirming the experts carry some signal, but
never enough to overtake the existing lasso gate. Net: `moe_soft` is a correct,
cleanly optional addition that is most useful as a robustness/diagnostic mode,
not a general accuracy improvement over the default BIR or over MARS.

## Files

- `eval_common.R` — shared helpers (split, RMSE, OOS R2, BIR/earth runners, the
  four-way `run_bir_experts` moe_soft runner, se calibration).
- `eval_boston.R`, `eval_synthetic.R`, `eval_spam.R`, `eval_solubility.R` —
  per-dataset drivers, each runnable as a separate timed invocation.
