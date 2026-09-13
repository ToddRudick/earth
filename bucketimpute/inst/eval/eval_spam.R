## eval_spam.R
## kernlab::spam with the binary `type` treated as numeric 0/1 (user approved),
## scored by RMSE. This is EXPECTED to fail at the fit_bir bucketing step:
## equal-frequency quantile bucketing of a two-valued y cannot produce n >= 2
## distinct non-empty buckets, so fit_bir raises the distinct-buckets error.
## We capture that error via tryCatch and report it as an expected, reportable
## outcome (NOT a bug). Run as:
##   timeout 300 Rscript bucketimpute/inst/eval/eval_spam.R

here <- dirname(sub("--file=", "",
                    grep("--file=", commandArgs(FALSE), value = TRUE)[1]))
if (is.na(here) || !nzchar(here)) here <- "bucketimpute/inst/eval"
source(file.path(here, "eval_common.R"))

suppressMessages(library(kernlab))
data(spam)

## type is a factor with levels "nonspam"/"spam"; encode as numeric 0/1.
y <- as.numeric(spam$type == "spam")
x <- as.matrix(spam[, setdiff(names(spam), "type")])

sp <- train_test_split(nrow(x), prop_train = 0.70, seed = SEED)
x_train <- x[sp$train, , drop = FALSE]; y_train <- y[sp$train]

cat(sprintf("spam: %d train rows, %d predictors; y in {0,1} (numeric)\n",
            nrow(x_train), ncol(x_train)))
cat(sprintf("distinct y values in training: %d\n", length(unique(y_train))))

bir <- run_bir(x_train, y_train, x_train[1:5, , drop = FALSE], n = 10)

if (bir$ok) {
  cat("UNEXPECTED: BIR fit succeeded on binary spam y (review this).\n")
  cat(sprintf("\nRESULT_ROW\tspam\tUNEXPECTED_OK\t-\t-\t%.1f\t-\t-\n",
              bir$seconds))
} else {
  cat("\nEXPECTED distinct-buckets error captured from fit_bir:\n")
  cat("  ", bir$error, "\n", sep = "")
  cat(sprintf("\nRESULT_ROW\tspam\tEXPECTED_ERROR\t-\tn/a\t%.2f\t-\t-\n",
              bir$seconds))
}
