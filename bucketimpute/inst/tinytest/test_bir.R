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

## ---- soft mixture-of-experts (moe_soft, Option A) ---------------------

## (a) default (no experts): experts field is NULL, default method unchanged,
## and requesting moe_soft on such a model errors clearly.
expect_true(is.null(fit$experts))
expect_equal(predict(fit, x, type = "response"),
             predict(fit, x, type = "response", method = "affinity_lasso"))
expect_error(predict(fit, x, method = "moe_soft"),
             "moe_soft|experts")

## fit a no-experts and a with-experts model under an IDENTICAL caller seed,
## so the lasso CV folds line up and the affinity_lasso path is comparable.
set.seed(7)
fit_noe <- fit_bir(x, y, n = 4)
set.seed(7)
fit_e <- fit_bir(x, y, n = 4, experts = TRUE)
resp_noe <- predict(fit_noe, x, type = "response")

## (e) experts must NOT alter the affinity_lasso path: default-method output is
## byte-for-byte identical to the no-experts model (same seed, same data).
expect_equal(fit_e$lasso_s, fit_noe$lasso_s)
expect_equal(predict(fit_e, x, type = "response"), resp_noe)
expect_equal(predict(fit_e, x, type = "response", method = "affinity_lasso"),
             resp_noe)

## the experts field is a length-n list
expect_equal(length(fit_e$experts), 4L)
expect_equal(fit_e$expert_type, "earth_degree2")

## (b) moe_soft round-trip: finite numeric vector of correct length
moe <- predict(fit_e, x, method = "moe_soft")
expect_equal(length(moe), N)
expect_true(is.numeric(moe))
expect_true(all(is.finite(moe)))

## type = "all" under moe_soft returns fit / se / affinities, with fit == moe
moe_all <- predict(fit_e, x, type = "all", method = "moe_soft")
expect_true(all(c("fit", "se", "affinities") %in% names(moe_all)))
expect_equal(moe_all$fit, moe)
expect_equal(length(moe_all$se), N)
expect_true(all(is.finite(moe_all$se) & moe_all$se >= 0))
expect_equal(dim(moe_all$affinities), c(N, 4L))

## type = "affinity" is identical regardless of method
expect_equal(predict(fit_e, x, type = "affinity", method = "moe_soft"),
             predict(fit_e, x, type = "affinity"))

## (c) requesting moe_soft without experts errors clearly
expect_error(predict(fit, x, method = "moe_soft"), "moe_soft|experts")

## (d) weights normalize / all-zero-affinity fallback -> unweighted expert
## average, finite and free of divide-by-zero. Simulate the guard directly:
## with an all-zero affinity row, weights collapse to 1/n and the blend is the
## plain mean of the per-bucket expert predictions.
expert_vals <- vapply(seq_len(fit_e$n), function(k) {
  ek <- fit_e$experts[[k]]
  if (inherits(ek, "bir_const_expert")) ek$value
  else as.numeric(predict(ek, x[1, , drop = FALSE]))
}, numeric(1))
zero_aff <- matrix(0, nrow = 1L, ncol = fit_e$n)
blend_zero <- bucketimpute:::.bir_moe_predict(fit_e, x[1, , drop = FALSE],
                                              zero_aff)
expect_true(is.finite(blend_zero))
expect_equal(blend_zero, mean(expert_vals))

## invalid experts argument errors
expect_error(fit_bir(x, y, n = 4, experts = "nope"), "experts")

## print reports experts presence without error
expect_silent(invisible(capture.output(print(fit_e))))
expect_silent(invisible(capture.output(summary(fit_e))))
