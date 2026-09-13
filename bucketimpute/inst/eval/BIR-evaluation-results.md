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

## Files

- `eval_common.R` — shared helpers (split, RMSE, BIR/earth runners, se calibration).
- `eval_boston.R`, `eval_synthetic.R`, `eval_spam.R`, `eval_solubility.R` —
  per-dataset drivers, each runnable as a separate timed invocation.
