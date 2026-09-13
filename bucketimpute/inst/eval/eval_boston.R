## eval_boston.R
## Out-of-sample BIR vs baseline on MASS::Boston (response medv, regression).
## Run as its own timed invocation, e.g.:
##   timeout 600 Rscript bucketimpute/inst/eval/eval_boston.R
##
## Protocol: 70/30 train/test split, seed 2024. BIR with n = 10 buckets.
## Baseline: plain degree-2 earth (MARS), a fair state-of-the-art regressor.

here <- dirname(sub("--file=", "",
                    grep("--file=", commandArgs(FALSE), value = TRUE)[1]))
if (is.na(here) || !nzchar(here)) here <- "bucketimpute/inst/eval"
source(file.path(here, "eval_common.R"))

suppressMessages(library(MASS))
data(Boston)

y_col <- "medv"
x <- as.matrix(Boston[, setdiff(names(Boston), y_col)])
y <- Boston[[y_col]]

sp <- train_test_split(nrow(x), prop_train = 0.70, seed = SEED)
x_train <- x[sp$train, , drop = FALSE]; y_train <- y[sp$train]
x_test  <- x[sp$test,  , drop = FALSE]; y_test  <- y[sp$test]

cat(sprintf("Boston: %d train / %d test rows, %d predictors\n",
            nrow(x_train), nrow(x_test), ncol(x_train)))

bir <- run_bir(x_train, y_train, x_test, n = 10)
if (!bir$ok) stop("BIR failed on Boston: ", bir$error)
bir_rmse <- rmse(y_test, bir$fit)

base <- run_earth(x_train, y_train, x_test, degree = 2)
base_rmse <- rmse(y_test, base$pred)

calib <- se_calibration(bir$se, y_test, bir$fit)

report_block("Boston (medv)", bir_rmse, base_rmse, "earth",
             bir$seconds, base$seconds, calib)

cat(sprintf("\nRESULT_ROW\tBoston\t%.4f\t%.4f\tearth(deg2)\t%.1f\t%.1f\t%.3f\n",
            bir_rmse, base_rmse, bir$seconds, base$seconds, calib$spearman))

## ---- four-way moe_soft comparison (FEAT-002) ---------------------------
## Fit BIR ONCE with experts enabled and derive the old affinity-lasso BIR,
## moe_soft, and the affinity-weighted bucket-mean floor from that one model.
moe <- run_bir_experts(x_train, y_train, x_test, n = 10)
if (!moe$ok) stop("BIR(experts) failed on Boston: ", moe$error)
report_moe_block("Boston", y_test, base$pred, moe$bir_old, moe$moe_soft,
                 moe$floor, moe$seconds, base$seconds)
