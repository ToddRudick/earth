## eval_synthetic.R
## Out-of-sample BIR vs baseline on the reproducible "synthetic_large" dataset.
## Run as its own timed invocation, e.g.:
##   timeout 1200 Rscript bucketimpute/inst/eval/eval_synthetic.R
##
## synthetic_large generator (documented exactly; seed 2024, n = 4000, p = 40):
##   x1..x6 : independent Uniform(0, 1) signal predictors
##   y = 5*x1 + 4*pmax(x2 - 0.3, 0) + 3*pmax(0.5 - x3, 0)
##       + 2.5*x4*x5 + 1.2*x6 + rnorm(n, 0, 1)
##   x7 = x1 + rnorm(n, 0, 0.25)   (correlated / redundant with x1)
##   x8 = x2 + rnorm(n, 0, 0.25)   (correlated / redundant with x2)
##   x9..x40 : pure Uniform(0, 1) noise, unrelated to y
##
## Protocol: 70/30 train/test split (seed 2024), BIR n = 10 buckets.
## Baseline: plain degree-2 earth (MARS).

here <- dirname(sub("--file=", "",
                    grep("--file=", commandArgs(FALSE), value = TRUE)[1]))
if (is.na(here) || !nzchar(here)) here <- "bucketimpute/inst/eval"
source(file.path(here, "eval_common.R"))

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
  ## columns 9..40 remain independent Uniform(0,1) noise (from the initial fill)
  list(x = X, y = y)
}

d <- make_synthetic_large()
x <- d$x; y <- d$y

sp <- train_test_split(nrow(x), prop_train = 0.70, seed = SEED)
x_train <- x[sp$train, , drop = FALSE]; y_train <- y[sp$train]
x_test  <- x[sp$test,  , drop = FALSE]; y_test  <- y[sp$test]

cat(sprintf("synthetic_large: %d train / %d test rows, %d predictors\n",
            nrow(x_train), nrow(x_test), ncol(x_train)))

bir <- run_bir(x_train, y_train, x_test, n = 10)
if (!bir$ok) stop("BIR failed on synthetic_large: ", bir$error)
bir_rmse <- rmse(y_test, bir$fit)

base <- run_earth(x_train, y_train, x_test, degree = 2)
base_rmse <- rmse(y_test, base$pred)

calib <- se_calibration(bir$se, y_test, bir$fit)

report_block("synthetic_large", bir_rmse, base_rmse, "earth",
             bir$seconds, base$seconds, calib)

cat(sprintf("\nRESULT_ROW\tsynthetic_large\t%.4f\t%.4f\tearth(deg2)\t%.1f\t%.1f\t%.3f\n",
            bir_rmse, base_rmse, bir$seconds, base$seconds, calib$spearman))
