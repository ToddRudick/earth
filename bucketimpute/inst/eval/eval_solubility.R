## eval_solubility.R
## AppliedPredictiveModeling::solubility -- WIDE data (~951 x 228 predictors).
## BIR fits n*p earth models per fit (n=10, p=228 => ~2280 earth models, each
## regressing one column on the other 227 across a thin bucket of ~66 rows,
## i.e. p >> rows-in-bucket). This is expensive and thin, so this script:
##   1. runs a SMALL SMOKE first (subset of predictors, small n) to confirm the
##      pipeline works and to get a cost signal, then
##   2. attempts a bounded "real" run controlled by env vars, and records
##      wall-clock cost and whether it completed.
##
## Because /this/ script must not hang, invoke it under a hard bash timeout and
## keep the internal work modest. Example invocations:
##   timeout 120  Rscript bucketimpute/inst/eval/eval_solubility.R smoke
##   timeout 1800 Rscript bucketimpute/inst/eval/eval_solubility.R full
##
## The `mode` arg ("smoke" default, or "full") selects the configuration. An
## optional second arg is a wall-clock budget in seconds enforced INSIDE R via
## setTimeLimit(); if the fit exceeds it, the run is reported as
## DID_NOT_COMPLETE rather than hanging. Default budget: 1200 s for full.

here <- dirname(sub("--file=", "",
                    grep("--file=", commandArgs(FALSE), value = TRUE)[1]))
if (is.na(here) || !nzchar(here)) here <- "bucketimpute/inst/eval"
source(file.path(here, "eval_common.R"))

args <- commandArgs(trailingOnly = TRUE)
mode <- if (length(args) >= 1) args[1] else "smoke"
budget_s <- if (length(args) >= 2) as.numeric(args[2]) else 1200

suppressMessages(library(AppliedPredictiveModeling))
data(solubility)

## solTrainXtrans/solTrainY = training; solTestX/solTestY = held-out test.
x_train_full <- as.matrix(solTrainXtrans)
y_train_full <- as.numeric(solTrainY)
x_test_full  <- as.matrix(solTestX)
y_test_full  <- as.numeric(solTestY)

cat(sprintf("solubility: %d train / %d test rows, %d predictors (mode=%s)\n",
            nrow(x_train_full), nrow(x_test_full), ncol(x_train_full), mode))

## ---- configuration per mode --------------------------------------------
## smoke: tiny subset to prove the pipeline + measure per-model cost cheaply.
## full : whole predictor set with n = 10 as in the primary protocol.
if (mode == "smoke") {
  set.seed(SEED)
  keep <- sort(sample.int(ncol(x_train_full), 20))  # 20 predictors
  n_buckets <- 4
} else {
  keep <- seq_len(ncol(x_train_full))               # all predictors
  n_buckets <- 10
}

x_train <- x_train_full[, keep, drop = FALSE]
x_test  <- x_test_full[,  keep, drop = FALSE]

cat(sprintf("  using %d predictors, n = %d buckets => ~%d earth models/fit\n",
            length(keep), n_buckets, length(keep) * n_buckets))

## Enforce a hard wall-clock budget inside R so a slow fit reports honestly
## instead of hanging. run_bir already wraps fit_bir in tryCatch, so an elapsed
## time-limit signal surfaces as bir$ok == FALSE with a "reached ... time limit"
## message.
cat(sprintf("  wall-clock budget: %.0f s (enforced via setTimeLimit)\n",
            budget_s))
setTimeLimit(elapsed = budget_s, transient = TRUE)
bir <- run_bir(x_train, y_train_full, x_test, n = n_buckets)
setTimeLimit(elapsed = Inf)

if (!bir$ok) {
  cat("\nBIR fit did NOT complete within budget: ", bir$error, "\n", sep = "")
  cat(sprintf("\nRESULT_ROW\tsolubility(%s)\tDID_NOT_COMPLETE\t-\t-\t%.1f\t-\t-\n",
              mode, bir$seconds))
} else {
  bir_rmse <- rmse(y_test_full, bir$fit)
  base <- run_earth(x_train, y_train_full, x_test, degree = 2)
  base_rmse <- rmse(y_test_full, base$pred)
  calib <- se_calibration(bir$se, y_test_full, bir$fit)
  report_block(sprintf("solubility (%s)", mode), bir_rmse, base_rmse, "earth",
               bir$seconds, base$seconds, calib)
  cat(sprintf("\nRESULT_ROW\tsolubility(%s)\t%.4f\t%.4f\tearth(deg2)\t%.1f\t%.1f\t%.3f\n",
              mode, bir_rmse, base_rmse, bir$seconds, base$seconds,
              calib$spearman))

  ## ---- four-way moe_soft comparison (FEAT-002) -------------------------
  ## Fit BIR ONCE with experts enabled under the SAME wall-clock budget. The
  ## experts add n degree-2 earth fits on bucket rows, so honor the budget and
  ## report DID_NOT_COMPLETE rather than hanging if it is exceeded.
  cat(sprintf("\n  fitting BIR with experts (budget %.0f s)...\n", budget_s))
  setTimeLimit(elapsed = budget_s, transient = TRUE)
  moe <- run_bir_experts(x_train, y_train_full, x_test, n = n_buckets)
  setTimeLimit(elapsed = Inf)
  if (!moe$ok) {
    cat("  BIR(experts) did NOT complete within budget: ", moe$error, "\n",
        sep = "")
    cat(sprintf("\nMOE_ROW\tsolubility(%s)\tDID_NOT_COMPLETE\t-\t-\n", mode))
  } else {
    report_moe_block(sprintf("solubility(%s)", mode), y_test_full, base$pred,
                     moe$bir_old, moe$moe_soft, moe$floor, moe$seconds,
                     base$seconds)
  }
}
