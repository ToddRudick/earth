library(bucketimpute)

set.seed(42)

## ---- small synthetic dataset ------------------------------------------
N <- 150
p <- 3
x <- matrix(rnorm(N * p), N, p)
colnames(x) <- c("a", "b", "c")
y <- x[, 1] - 0.5 * x[, 2] + 0.2 * x[, 3] + rnorm(N, sd = 0.3)

fit <- fit_bir(x, y, n = 4)

## (1) fit + predict round-trip -----------------------------------------
expect_inherits(fit, "bir")
expect_equal(fit$n, 4L)
expect_equal(fit$p, 3L)
expect_equal(fit$colnames, c("a", "b", "c"))
expect_equal(dim(fit$scales), c(4L, 3L))
expect_true(all(fit$scales >= fit$scale_floor))
expect_equal(length(fit$bucket_vars), 4L)

## stored object contains all spec-required fields
for (nm in c("cuts", "models", "scales", "lasso", "bucket_vars",
             "n", "p", "colnames")) {
  expect_true(nm %in% names(fit), info = nm)
}

## response -> finite numeric vector of correct length
resp <- predict(fit, x, type = "response")
expect_equal(length(resp), N)
expect_true(all(is.finite(resp)))

## affinity -> N x n matrix in [0, 1]
aff <- predict(fit, x, type = "affinity")
expect_equal(dim(aff), c(N, 4L))
expect_true(all(aff >= 0 & aff <= 1))

## all -> fit + se + affinities
allp <- predict(fit, x, type = "all")
expect_true(all(c("fit", "se", "affinities") %in% names(allp)))
expect_equal(length(allp$fit), N)
expect_true(all(is.finite(allp$fit)))

## (3) se output shape / sign -------------------------------------------
expect_equal(length(allp$se), N)
expect_true(all(is.finite(allp$se)))
expect_true(all(allp$se >= 0))
expect_equal(dim(allp$affinities), c(N, 4L))

## default type is "response"
expect_equal(predict(fit, x), resp)

## column reordering at predict time is handled
x_shuffled <- x[, c("c", "a", "b")]
expect_equal(predict(fit, x_shuffled, type = "response"), resp)

## missing column is an error
expect_error(predict(fit, x[, c("a", "b")]), "missing required column")

## reproducibility: same caller-set seed -> identical lasso_s and predictions
set.seed(123)
fit_a <- fit_bir(x, y, n = 4)
set.seed(123)
fit_b <- fit_bir(x, y, n = 4)
expect_equal(fit_a$lasso_s, fit_b$lasso_s)
expect_equal(predict(fit_a, x, type = "response"),
             predict(fit_b, x, type = "response"))

## (2) near-binary y -> distinct-buckets error --------------------------
y_bin <- c(rep(0, N / 2), rep(1, N / 2))
expect_error(fit_bir(x, y_bin, n = 4), "distinct")

## (4) n < 2 errors ------------------------------------------------------
expect_error(fit_bir(x, y, n = 1), "n.*>= 2|>= 2")

## scale_floor validation
expect_error(fit_bir(x, y, n = 4, scale_floor = -1), "scale_floor")

## print / summary run without error
expect_silent(invisible(capture.output(print(fit))))
expect_silent(invisible(capture.output(summary(fit))))
