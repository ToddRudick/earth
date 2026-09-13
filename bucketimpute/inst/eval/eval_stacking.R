## eval_stacking.R
## STACKING EXPERIMENT (FEAT-003): does giving the "original model" (plain
## degree-2 earth) BIR's default affinity_lasso prediction as an EXTRA input
## column improve its out-of-sample accuracy?
##
## Design. Build a stacked feature matrix
##     x_augmented = [ x  ||  bir_pred ]
## (original predictors plus one new column, "bir_pred") and fit the original
## model on it: earth(x_augmented, y, degree = 2).
##
## The bir_pred feature is generated OUT-OF-FOLD to avoid leakage (see
## bir_oof_feature() in eval_common.R):
##   * TRAIN column: K-fold cross-fit. For each fold, fit BIR on the other
##     K-1 folds and predict the held-out fold.
##   * TEST  column: fit BIR ONCE on the FULL training split, predict test.
## BIR uses its DEFAULT method (affinity_lasso); moe_soft is NOT used here.
##
## Protocol matches the rest of the harness: SEED = 2024, 70/30 outer holdout,
## n = 10 buckets for BIR, degree = 2 earth. Per dataset we report OOS-R2 and
## RMSE for:
##   (a) plain earth on x           -- the existing baseline / "original model"
##   (b) earth on [x || bir_pred]   -- the NEW stacked model
##   (c) BIR alone (affinity_lasso) -- for reference
## and we inspect the fitted stacked earth model (evimp) to report whether the
## bir_pred term was actually selected and how important it was.
##
## Runtime. The out-of-fold scheme fits BIR K + 1 times per dataset. Boston and
## synthetic are cheap; solubility (228 predictors) is expensive, so run each
## dataset as a SEPARATE timed invocation and reduce K for solubility. Invoke:
##   timeout  600 Rscript bucketimpute/inst/eval/eval_stacking.R boston
##   timeout 1500 Rscript bucketimpute/inst/eval/eval_stacking.R synthetic
##   timeout 1700 Rscript bucketimpute/inst/eval/eval_stacking.R solubility 3 1500
##   timeout  180 Rscript bucketimpute/inst/eval/eval_stacking.R smoke
## Args: <dataset> [K] [budget_s]. dataset in {boston, synthetic, solubility,
## smoke}. smoke runs a tiny solubility-subset pipeline to validate plumbing.

here <- dirname(sub("--file=", "",
                    grep("--file=", commandArgs(FALSE), value = TRUE)[1]))
if (is.na(here) || !nzchar(here)) here <- "bucketimpute/inst/eval"
source(file.path(here, "eval_common.R"))

args    <- commandArgs(trailingOnly = TRUE)
dataset <- if (length(args) >= 1) args[1] else "boston"
K_arg   <- if (length(args) >= 2) as.integer(args[2]) else NA_integer_
budget_s <- if (length(args) >= 3) as.numeric(args[3]) else Inf

## ---- dataset loaders: return list(x, y, name, default_K) ------------------
load_dataset <- function(dataset) {
  if (dataset == "boston") {
    suppressMessages(library(MASS)); data(Boston)
    x <- as.matrix(Boston[, setdiff(names(Boston), "medv")])
    y <- Boston[["medv"]]
    sp <- train_test_split(nrow(x), prop_train = 0.70, seed = SEED)
    return(list(name = "Boston", default_K = 5,
                x_train = x[sp$train, , drop = FALSE], y_train = y[sp$train],
                x_test  = x[sp$test,  , drop = FALSE], y_test  = y[sp$test]))
  }
  if (dataset == "synthetic") {
    make_synthetic_large <- function(seed = 2024, n = 4000, p = 40) {
      set.seed(seed)
      X <- matrix(runif(n * p), nrow = n, ncol = p)
      colnames(X) <- paste0("x", seq_len(p))
      x1 <- X[, 1]; x2 <- X[, 2]; x3 <- X[, 3]
      x4 <- X[, 4]; x5 <- X[, 5]; x6 <- X[, 6]
      y <- 5 * x1 + 4 * pmax(x2 - 0.3, 0) + 3 * pmax(0.5 - x3, 0) +
        2.5 * x4 * x5 + 1.2 * x6 + rnorm(n, 0, 1)
      X[, 7] <- x1 + rnorm(n, 0, 0.25)
      X[, 8] <- x2 + rnorm(n, 0, 0.25)
      list(x = X, y = y)
    }
    d <- make_synthetic_large()
    sp <- train_test_split(nrow(d$x), prop_train = 0.70, seed = SEED)
    return(list(name = "synthetic_large", default_K = 5,
                x_train = d$x[sp$train, , drop = FALSE], y_train = d$y[sp$train],
                x_test  = d$x[sp$test,  , drop = FALSE], y_test  = d$y[sp$test]))
  }
  if (dataset %in% c("solubility", "smoke")) {
    suppressMessages(library(AppliedPredictiveModeling)); data(solubility)
    x_train <- as.matrix(solTrainXtrans); y_train <- as.numeric(solTrainY)
    x_test  <- as.matrix(solTestX);       y_test  <- as.numeric(solTestY)
    if (dataset == "smoke") {
      set.seed(SEED)
      keep <- sort(sample.int(ncol(x_train), 20))
      x_train <- x_train[, keep, drop = FALSE]
      x_test  <- x_test[,  keep, drop = FALSE]
      return(list(name = "solubility(smoke)", default_K = 3,
                  x_train = x_train, y_train = y_train,
                  x_test = x_test, y_test = y_test, n_buckets = 4))
    }
    return(list(name = "solubility(full)", default_K = 3,
                x_train = x_train, y_train = y_train,
                x_test = x_test, y_test = y_test))
  }
  stop("unknown dataset: ", dataset)
}

d <- load_dataset(dataset)
K <- if (!is.na(K_arg)) K_arg else d$default_K
n_buckets <- if (!is.null(d$n_buckets)) d$n_buckets else 10L

cat(sprintf("%s: %d train / %d test rows, %d predictors | K=%d, n=%d buckets\n",
            d$name, nrow(d$x_train), nrow(d$x_test), ncol(d$x_train),
            K, n_buckets))
if (is.finite(budget_s))
  cat(sprintf("  wall-clock budget: %.0f s (enforced via setTimeLimit)\n",
              budget_s))

## ---- (a) plain earth on x -- the original model / baseline ---------------
base <- run_earth(d$x_train, d$y_train, d$x_test, degree = 2)
r2_a  <- r2(d$y_test, base$pred);  rmse_a <- rmse(d$y_test, base$pred)

## ---- out-of-fold bir_pred feature ----------------------------------------
if (is.finite(budget_s)) setTimeLimit(elapsed = budget_s, transient = TRUE)
oof <- tryCatch(
  bir_oof_feature(d$x_train, d$y_train, d$x_test, n = n_buckets, K = K),
  error = function(e) list(ok = FALSE, error = conditionMessage(e),
                           seconds = NA_real_)
)
if (is.finite(budget_s)) setTimeLimit(elapsed = Inf)

if (!isTRUE(oof$ok)) {
  cat("\nOut-of-fold BIR feature did NOT complete: ", oof$error, "\n", sep = "")
  cat(sprintf("\nSTACK_ROW\t%s\tplain_earth\t%.4f\t%.4f\n",
              d$name, r2_a, rmse_a))
  cat(sprintf("STACK_ROW\t%s\tstacked_earth\tNA\tNA\n", d$name))
  cat(sprintf("STACK_ROW\t%s\tBIR_alone\tNA\tNA\n", d$name))
  cat(sprintf("STACK_NOTE\t%s\tK=%d\tstatus=DID_NOT_COMPLETE (%s)\n",
              d$name, K, oof$error))
  quit(save = "no", status = 0)
}

## ---- (c) BIR alone: predict test from the full-train BIR model -----------
bir_pred_test <- oof$test
r2_c  <- r2(d$y_test, bir_pred_test);  rmse_c <- rmse(d$y_test, bir_pred_test)

## ---- (b) stacked earth: earth([x || bir_pred], y) ------------------------
## Preserve column names/order into predict; name the new column "bir_pred".
x_train_aug <- cbind(d$x_train, bir_pred = oof$oof_train)
x_test_aug  <- cbind(d$x_test,  bir_pred = oof$test)
stopifnot(identical(colnames(x_train_aug), colnames(x_test_aug)),
          "bir_pred" %in% colnames(x_train_aug))

set.seed(SEED)
stacked_fit  <- earth(x = x_train_aug, y = as.numeric(d$y_train), degree = 2)
stacked_pred <- as.numeric(predict(stacked_fit, x_test_aug))
r2_b  <- r2(d$y_test, stacked_pred);  rmse_b <- rmse(d$y_test, stacked_pred)

## ---- did MARS actually SELECT/USE the bir_pred term? ---------------------
## $dirs is a (terms x variables) matrix; columns are variables. A term uses
## bir_pred iff its bir_pred column entry is non-zero. Count only among the
## MARS-selected terms.
sel <- if (!is.null(stacked_fit$selected.terms)) stacked_fit$selected.terms else integer(0)
terms_with_bir <- if ("bir_pred" %in% colnames(stacked_fit$dirs))
  sum(stacked_fit$dirs[sel, "bir_pred"] != 0) else 0L
used_in_terms <- terms_with_bir > 0L
imp <- tryCatch(evimp(stacked_fit, trim = FALSE), error = function(e) NULL)

## ---- report --------------------------------------------------------------
cat(sprintf("\n===== %s : stacking OOS comparison =====\n", d$name))
cat(sprintf("  %-24s %10s %10s\n", "model", "OOS-R2", "RMSE"))
cat(sprintf("  %-24s %10.4f %10.4f\n", "(a) plain earth [x]", r2_a, rmse_a))
cat(sprintf("  %-24s %10.4f %10.4f\n", "(b) stacked earth [x|bir]", r2_b, rmse_b))
cat(sprintf("  %-24s %10.4f %10.4f\n", "(c) BIR alone", r2_c, rmse_c))
cat(sprintf("  delta (b - a) OOS-R2 : %+.4f\n", r2_b - r2_a))
cat(sprintf("  out-of-fold BIR wall-clock: %.1f s (K=%d folds + 1 full fit)\n",
            oof$seconds, K))
cat(sprintf("  per-fold seconds: %s\n",
            paste(sprintf("%.1f", oof$fold_seconds), collapse = ", ")))

cat("\n  --- bir_pred term selected by MARS? ---\n")
cat(sprintf("  bir_pred appears in any selected term : %s\n", used_in_terms))
cat(sprintf("  # selected terms referencing bir_pred : %d\n", terms_with_bir))
if (!is.null(imp)) {
  cat("  evimp() variable importance (nsubsets / gcv / rss):\n")
  print(imp)
} else {
  cat("  evimp() unavailable for this fit\n")
}

## machine-readable rows
cat(sprintf("\nSTACK_ROW\t%s\tplain_earth\t%.4f\t%.4f\n", d$name, r2_a, rmse_a))
cat(sprintf("STACK_ROW\t%s\tstacked_earth\t%.4f\t%.4f\n", d$name, r2_b, rmse_b))
cat(sprintf("STACK_ROW\t%s\tBIR_alone\t%.4f\t%.4f\n", d$name, r2_c, rmse_c))
cat(sprintf("STACK_USE\t%s\tbir_pred_selected=%s\tterms=%d\tK=%d\tsec=%.1f\n",
            d$name, used_in_terms, terms_with_bir, K, oof$seconds))
