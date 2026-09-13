# Spec: Bucketed Imputation Regression

## Goal
Implement a regression method that:
1. Divides the continuous response `y` into `n` equal-frequency quantile buckets.
2. Inside each bucket fits a degree-1 earth model for every predictor (imputing each column from all others).
3. For any observation computes an “unusualness” score per bucket (standardized absolute reconstruction error).
4. Converts those scores into affinity features.
5. Fits a lasso regression of `y` on the affinity features.
6. At prediction time returns both a point prediction and a simple heteroscedastic scale (predictive spread).

Primary objective is predictive accuracy; secondary objective is to also supply a rough predictive uncertainty.

## Packages
- `earth` (degree-1 MARS)
- `glmnet` (lasso with cross-validation)
- `caret` only if needed for outer evaluation; the core method does not depend on caret
- Base R for quantiles, matrix operations, etc.

No other heavy dependencies.

## Data assumptions
- `x` is a numeric matrix or data.frame (all columns numeric, no factors).
- `y` is a numeric vector (continuous).
- No missing values (caller is responsible for imputation if needed).
- `n` (number of buckets) defaults to 10 and must be ≥ 2.

## Training algorithm (`fit_bir`)

```r
fit_bir <- function(x, y, n = 10, ...)
```

### Steps

1. **Bucket assignment**
   - Compute `n+1` quantile cut-points of `y` (type = 7, the R default).
   - Store the cut-points (they will be needed for prediction).
   - Assign every training row to a bucket `k = 1 … n` using `findInterval` / `cut`.
   - Guarantees roughly equal number of rows per bucket.

2. **Per-bucket earth models**
   - For each bucket `k = 1 … n`:
     - Extract the sub-matrix `x_k` of rows belonging to bucket `k`.
     - For every column `j = 1 … p`:
       - Fit `earth(x_k[,-j], x_k[,j], degree = 1, ...)`  
         (linear MARS; pass any extra `earth` arguments via `...`).
       - Record the residual scale of the fit:  
         `s_{k,j} = sqrt(mean(residuals(fit)^2))`  
         (ordinary residual standard error; simple and consistent with the model).
       - Store the fitted earth object.
   - Result: a list of length `n`, each element a list of `p` earth models + a vector of `p` residual scales.

3. **Training affinities**
   - For every training row `i`:
     - For each bucket `k` compute
       ```
       unusualness_{i,k} = sum_j |x_{i,j} - predict(earth_{k,j}, x_i[,-j])| / s_{k,j}
       ```
     - Convert to affinity:
       ```
       affinity_{i,k} = 1 / (1 + unusualness_{i,k})
       ```
   - Produce an `N × n` matrix of affinities.

4. **Final lasso**
   - Fit `cv.glmnet(affinities, y, alpha = 1)` (lasso).
   - Store the whole `cv.glmnet` object (so the optimal λ is available).

5. **Bucket variances (for uncertainty)**
   - For each bucket `k` compute the empirical variance of the `y` values that fell into that bucket:
     ```
     v_k = var(y[bucket == k])
     ```
   - Store the length-`n` vector `v`.

6. **Return value**
   An S3 object of class `"bir"` containing at least:
   - `cuts`          – the quantile cut-points
   - `models`        – the nested list of earth models
   - `scales`        – the nested list / matrix of residual scales `s_{k,j}`
   - `lasso`         – the `cv.glmnet` object
   - `bucket_vars`   – vector of length `n`
   - `n`, `p`, `formula` or column names, etc. for convenience

## Prediction (`predict.bir`)

```r
predict.bir <- function(object, newdata, type = c("response", "affinity", "all"), ...)
```

- `type = "response"` (default) → point prediction only.
- `type = "affinity"` → the `n` affinity values.
- `type = "all"` → a data.frame / list with:
  - `fit`          – point prediction
  - `se`           – predictive scale (square-root of weighted variance)
  - `affinities`   – the raw affinity vector (optional)

### Steps for a new matrix `newdata` (rows = observations)

1. For each row and each bucket `k` compute the unusualness score exactly as in training (using the stored earth models and scales).
2. Convert to affinities: `a_k = 1 / (1 + u_k)`.
3. Point prediction:
   ```
   ŷ = predict(object$lasso, newx = affinities, s = "lambda.min")
   ```
4. Predictive scale:
   - Normalize the affinities so they sum to 1 → soft weights `w_k`.
   - `var_hat = sum(w_k * object$bucket_vars[k])`
   - `se = sqrt(var_hat)`
5. Return according to `type`.

## Helper / convenience methods
- `print.bir`, `summary.bir` – show number of buckets, λ chosen, etc.
- Optional: `plot.bir` showing the affinity → y relationship or coefficient path (nice-to-have, not required for first version).

## Evaluation recommendation (for the user of the method)
Because the training affinities are mildly optimistic, always evaluate the whole pipeline with an outer cross-validation or a held-out test set (caret’s `train` can wrap `fit_bir` / `predict.bir` if desired).

## Design choices deliberately kept simple
- Residual scale = ordinary residual SE (not MAD).
- Affinity = `1 / (1 + unusualness)` (no temperature parameter).
- Final model = pure lasso (`alpha = 1`) with `lambda.min`.
- Uncertainty = affinity-weighted average of the per-bucket empirical variances of `y`.
- Earth models are degree-1 only.
- No missing-value handling, no factor support, no parallelisation in the first version (can be added later).

## File layout expectation
A single self-contained R script or an R package skeleton that exports:
- `fit_bir()`
- `predict.bir()`
- the S3 class `"bir"`

Documentation (roxygen or plain comments) should make the algorithm and the returned object crystal-clear.
