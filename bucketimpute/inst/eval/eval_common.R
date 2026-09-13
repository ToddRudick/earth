## eval_common.R
## Shared helpers for the out-of-sample evaluation of Bucketed Imputation
## Regression (BIR) vs fair baselines. Sourced by the per-dataset scripts.
##
## All evaluation here is STRICTLY out-of-sample: models are trained on a
## held-out training split and scored on a disjoint test split. The BIR spec
## notes that training affinities are mildly optimistic, so in-sample scoring
## is deliberately never used.
##
## Reproducibility: every script sets set.seed(2024) before splitting/fitting.

suppressMessages({
  library(bucketimpute)
  library(earth)
})

SEED <- 2024

## Pin the RNG algorithm so results are reproducible across R installations,
## not merely within one machine's default. BIR's penalty selection uses
## lars::cv.lars, whose cross-validation folds are drawn from the GLOBAL RNG;
## if the ambient sampler differs (e.g. an older R with the pre-3.6
## "Rounding" sample.kind, or a non-default RNGkind), the exact fold draws and
## therefore the fitted lasso penalty shift, moving every downstream number.
## Fixing the kind to the R >= 3.6 default here means set.seed(SEED) selects
## the SAME stream everywhere, so the study reproduces from a clean install
## regardless of the host R's default sampler.
suppressWarnings(
  RNGkind("Mersenne-Twister", "Inversion", "Rejection")
)

## Root-mean-squared error.
rmse <- function(actual, predicted) {
  sqrt(mean((as.numeric(actual) - as.numeric(predicted))^2))
}

## Out-of-sample coefficient of determination:
##   R2 = 1 - SSE/SST
## where SSE = sum((actual - predicted)^2) and SST = sum((actual - mean)^2),
## using the TEST-set mean of `actual` for SST (the standard OOS R2). A model
## that only predicts the test-set mean scores 0; worse-than-mean scores are
## negative (which happens when a baseline extrapolates catastrophically).
r2 <- function(actual, predicted) {
  actual <- as.numeric(actual)
  predicted <- as.numeric(predicted)
  sse <- sum((actual - predicted)^2)
  sst <- sum((actual - mean(actual))^2)
  1 - sse / sst
}

## 70/30 train/test split of row indices, reproducible given a seed.
train_test_split <- function(n_rows, prop_train = 0.70, seed = SEED) {
  set.seed(seed)
  idx <- sample.int(n_rows)
  n_train <- floor(prop_train * n_rows)
  list(train = sort(idx[seq_len(n_train)]),
       test  = sort(idx[(n_train + 1L):n_rows]))
}

## Fit BIR on the training split and predict (type = "all") on the test split.
## Returns a list with fit, se, affinities, plus timing, or an error record.
run_bir <- function(x_train, y_train, x_test, n = 10, ...) {
  set.seed(SEED)
  t0 <- proc.time()[["elapsed"]]
  fit <- tryCatch(
    fit_bir(x_train, y_train, n = n, ...),
    error = function(e) e
  )
  if (inherits(fit, "error")) {
    return(list(ok = FALSE, error = conditionMessage(fit),
                seconds = proc.time()[["elapsed"]] - t0))
  }
  pr <- predict(fit, x_test, type = "all")
  list(ok = TRUE, fit = pr$fit, se = pr$se, affinities = pr$affinities,
       model = fit, seconds = proc.time()[["elapsed"]] - t0)
}

## Baseline: plain degree-2 earth (a fair, widely used MARS baseline).
run_earth <- function(x_train, y_train, x_test, degree = 2, ...) {
  set.seed(SEED)
  t0 <- proc.time()[["elapsed"]]
  fit <- earth(x = as.matrix(x_train), y = as.numeric(y_train),
               degree = degree, ...)
  pred <- as.numeric(predict(fit, as.matrix(x_test)))
  list(pred = pred, model = fit, seconds = proc.time()[["elapsed"]] - t0)
}

## ---------------------------------------------------------------------------
## Out-of-fold (cross-fitted) BIR prediction column for STACKING (FEAT-003).
##
## To feed BIR's default affinity_lasso point prediction to the "original
## model" (plain degree-2 earth) as an EXTRA input column WITHOUT leakage, the
## training-row bir_pred must be produced out-of-fold: for each of K folds, fit
## BIR on the other K-1 folds and predict the held-out fold. No training row's
## bir_pred ever used a BIR that saw that row. The TEST-row bir_pred is produced
## by a SINGLE BIR fit on the FULL training split.
##
## Reproducibility / seed scheme (deterministic from a CLEAN install):
##   * The RNG algorithm is pinned once at the top of this file
##     (RNGkind("Mersenne-Twister", "Inversion", "Rejection")), so set.seed()
##     selects the identical stream on any R >= 3.6 host.
##   * Fold assignment uses set.seed(SEED) then sample() of fold labels, so the
##     partition is fixed given SEED and K, and does NOT depend on any RNG
##     state left behind by earlier code in the session.
##   * BIR's CV-lasso penalty selection (cv.lars) consumes the global RNG, so
##     EACH fold fit AND the full-train fit call set.seed(SEED) immediately
##     before fit_bir(). Because every fit re-seeds from a fixed constant (not
##     from the accumulated stream), the number of BIR fits / RNG draws that
##     precede a given fit cannot change its result: every fit is independently
##     reproducible. This is what makes `Rscript eval_stacking.R <ds>` produce
##     identical numbers on every fresh session.
##
## Returns list(oof_train = <numeric length nrow(x_train)>,
##              test      = <numeric length nrow(x_test)>,
##              full_model = <bir fitted on full train>,
##              K = K, seconds = elapsed, fold_seconds = <per-fold vector>).
## On a BIR failure in ANY fold or the full fit, returns list(ok = FALSE, ...).
bir_oof_feature <- function(x_train, y_train, x_test, n = 10, K = 5,
                            seed = SEED, ...) {
  x_train <- as.matrix(x_train)
  x_test  <- as.matrix(x_test)
  n_tr <- nrow(x_train)
  t0 <- proc.time()[["elapsed"]]

  ## fixed, reproducible fold labels
  set.seed(seed)
  folds <- sample(rep_len(seq_len(K), n_tr))

  oof <- rep(NA_real_, n_tr)
  fold_seconds <- numeric(K)
  for (f in seq_len(K)) {
    in_test  <- which(folds == f)
    in_train <- which(folds != f)
    tf0 <- proc.time()[["elapsed"]]
    set.seed(seed)                       # reproducible cv.lars per fold
    fit_f <- tryCatch(
      fit_bir(x_train[in_train, , drop = FALSE], y_train[in_train],
              n = n, ...),
      error = function(e) e
    )
    if (inherits(fit_f, "error")) {
      return(list(ok = FALSE,
                  error = sprintf("fold %d/%d: %s", f, K,
                                  conditionMessage(fit_f)),
                  seconds = proc.time()[["elapsed"]] - t0))
    }
    oof[in_test] <- as.numeric(
      predict(fit_f, x_train[in_test, , drop = FALSE], type = "response")
    )
    fold_seconds[f] <- proc.time()[["elapsed"]] - tf0
  }

  ## TEST column: single BIR fit on the FULL training split.
  set.seed(seed)
  full_fit <- tryCatch(
    fit_bir(x_train, y_train, n = n, ...),
    error = function(e) e
  )
  if (inherits(full_fit, "error")) {
    return(list(ok = FALSE,
                error = sprintf("full-train fit: %s",
                                conditionMessage(full_fit)),
                seconds = proc.time()[["elapsed"]] - t0))
  }
  test_pred <- as.numeric(predict(full_fit, x_test, type = "response"))

  list(ok = TRUE, oof_train = oof, test = test_pred, full_model = full_fit,
       K = K, seconds = proc.time()[["elapsed"]] - t0,
       fold_seconds = fold_seconds)
}

## se calibration sanity check: does a larger predicted se go with a larger
## actual absolute residual? We report Spearman correlation plus mean |resid|
## per se quartile.
se_calibration <- function(se, actual, predicted, n_bins = 4) {
  abs_err <- abs(as.numeric(actual) - as.numeric(predicted))
  spearman <- suppressWarnings(
    stats::cor(se, abs_err, method = "spearman")
  )
  ## guard against degenerate (constant) se
  if (length(unique(se)) < n_bins) {
    bins <- NULL
  } else {
    q <- stats::quantile(se, probs = seq(0, 1, length.out = n_bins + 1L),
                         type = 7)
    grp <- cut(se, breaks = unique(q), include.lowest = TRUE)
    bins <- tapply(abs_err, grp, mean)
  }
  list(spearman = spearman, mean_abs_err_by_se_quantile = bins)
}

## Pretty-print a single dataset's result block to stdout (captured to a log).
report_block <- function(name, bir_rmse, base_rmse, base_name,
                         seconds_bir, seconds_base, calib = NULL,
                         extra = NULL) {
  cat(sprintf("\n===== %s =====\n", name))
  cat(sprintf("  BIR   OOS RMSE : %s\n", format(bir_rmse)))
  cat(sprintf("  %-5s OOS RMSE : %s\n", base_name, format(base_rmse)))
  cat(sprintf("  BIR wall-clock : %.1f s\n", seconds_bir))
  cat(sprintf("  base wall-clock: %.1f s\n", seconds_base))
  if (!is.null(calib)) {
    cat(sprintf("  se~|resid| Spearman: %s\n", format(calib$spearman)))
    if (!is.null(calib$mean_abs_err_by_se_quantile)) {
      cat("  mean |resid| by se quartile:\n")
      print(calib$mean_abs_err_by_se_quantile)
    }
  }
  if (!is.null(extra)) cat(extra, "\n")
}

## ---------------------------------------------------------------------------
## Four-way soft mixture-of-experts (moe_soft) comparison runner.
##
## Fits BIR ONCE on the training split WITH per-bucket experts enabled
## (fit_bir(..., experts = TRUE)) and, from that SINGLE fitted model, derives
## three of the four methods on the test split:
##
##   (2) old affinity-lasso BIR  = predict(fit, x_test)          [default method]
##   (3) moe_soft                = predict(fit, x_test, method = "moe_soft")
##   (4) bucket-mean FLOOR       = affinity-weighted per-bucket MEAN of the
##                                 TRAINING y, using the IDENTICAL normalized
##                                 affinity gate as moe_soft.
##
## The FLOOR isolates whether the per-bucket REGRESSION (the earth experts)
## adds value over merely blending bucket means under the same gate.
##
## How the floor reuses the identical affinity gate: moe_soft computes, for
## each test row, weights w_k = a_k / sum_j a_j from the affinity matrix
## returned by predict(fit, ., type = "affinity"), forcing an all-zero row to
## a uniform 1/n vector first (the .bir_moe_predict all-zero guard). The floor
## recomputes weights from THAT SAME affinity matrix with THAT SAME guard and
## normalization, then takes yhat_floor = sum_k w_k * mean(y_train in bucket k).
## moe_soft is yhat_moe = sum_k w_k * expert_k(x). The weights are therefore
## byte-identical between the two; the ONLY difference is the per-bucket value
## blended (fitted earth expert vs. constant bucket mean), so the moe_soft-minus
## -floor difference is attributable purely to the per-bucket regression.
##
## Method (1), plain degree-2 earth, is fit separately via run_earth().
##
## Returns a list with ok/error plus, on success, per-method prediction vectors
## (bir_old, moe_soft, floor), the fitted model, and timing. run_bir (old
## behaviour, no experts) is left intact for back-compat and other callers.
run_bir_experts <- function(x_train, y_train, x_test, n = 10, ...) {
  set.seed(SEED)
  t0 <- proc.time()[["elapsed"]]
  fit <- tryCatch(
    fit_bir(x_train, y_train, n = n, experts = TRUE, ...),
    error = function(e) e
  )
  if (inherits(fit, "error")) {
    return(list(ok = FALSE, error = conditionMessage(fit),
                seconds = proc.time()[["elapsed"]] - t0))
  }

  ## (2) old affinity-lasso BIR (default method), plus se via type = "all"
  pr_old <- predict(fit, x_test, type = "all")
  ## (3) moe_soft blend of the per-bucket earth experts
  pred_moe <- as.numeric(predict(fit, x_test, method = "moe_soft"))

  ## (4) bucket-mean floor: reuse the SAME affinity matrix and the SAME
  ## normalization/guard as moe_soft, but blend per-bucket TRAINING means.
  affin <- pr_old$affinities                 # identical to moe_soft's gate input
  zero_rows <- rowSums(affin) == 0
  if (any(zero_rows)) affin[zero_rows, ] <- 1   # .bir_moe_predict all-zero guard
  weights <- affin / rowSums(affin)
  ## per-bucket TRAINING means, using the model's own equal-frequency cuts so
  ## the bucket assignment matches fit_bir exactly.
  bucket_tr <- findInterval(as.numeric(y_train), fit$cuts,
                            rightmost.closed = TRUE, all.inside = TRUE)
  bucket_means <- vapply(seq_len(n), function(k) mean(y_train[bucket_tr == k]),
                         numeric(1))
  pred_floor <- as.numeric(weights %*% bucket_means)

  list(ok = TRUE,
       bir_old  = as.numeric(pr_old$fit),
       moe_soft = pred_moe,
       floor    = pred_floor,
       se = pr_old$se, affinities = pr_old$affinities,
       model = fit, seconds = proc.time()[["elapsed"]] - t0)
}

## Pretty-print the four-way OOS-R2 + RMSE comparison for one dataset and emit
## machine-readable RESULT rows. `preds` is a named list of prediction vectors
## keyed by method label; order is preserved for the printed block.
report_moe_block <- function(name, y_test, earth_pred, bir_old, moe_soft,
                             floor, seconds_bir = NA, seconds_earth = NA) {
  methods <- list(
    "earth(deg2)"        = earth_pred,
    "affinity_lasso(old)" = bir_old,
    "moe_soft"           = moe_soft,
    "bucket_mean_floor"  = floor
  )
  cat(sprintf("\n===== %s : four-way OOS comparison =====\n", name))
  cat(sprintf("  %-22s %10s %10s\n", "method", "OOS-R2", "RMSE"))
  for (m in names(methods)) {
    p <- methods[[m]]
    cat(sprintf("  %-22s %10.4f %10.4f\n", m, r2(y_test, p), rmse(y_test, p)))
  }
  if (!is.na(seconds_bir))
    cat(sprintf("  BIR(experts) wall-clock: %.1f s\n", seconds_bir))
  if (!is.na(seconds_earth))
    cat(sprintf("  earth wall-clock       : %.1f s\n", seconds_earth))
  for (m in names(methods)) {
    p <- methods[[m]]
    cat(sprintf("MOE_ROW\t%s\t%s\t%.4f\t%.4f\n",
                name, m, r2(y_test, p), rmse(y_test, p)))
  }
}
