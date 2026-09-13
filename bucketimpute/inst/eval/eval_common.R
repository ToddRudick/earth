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

## Root-mean-squared error.
rmse <- function(actual, predicted) {
  sqrt(mean((as.numeric(actual) - as.numeric(predicted))^2))
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
