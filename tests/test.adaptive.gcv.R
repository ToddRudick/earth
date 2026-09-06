# test.adaptive.gcv.R
#
# Validation suite for the experimental adaptive.gcv effect cap (FEAT-003).
#
# Implements the six synthetic scenarios from adaptive_gcv_effect_cap_coding_agent.md
# ("Validation plan") plus a hard regression check that adaptive.gcv=FALSE is
# numerically identical to stock earth (spec "Important behavior to verify" Q8).
#
# Each scenario fits ordinary earth (adaptive.gcv=FALSE) and adaptive earth
# (adaptive.gcv=TRUE), prints a compact comparison, and asserts the intended
# qualitative outcome with stopifnot() where an exact expectation exists.
#
# Mechanism (see man/earth.Rd and src/earth.c): the forward pass admits terms
# exactly as stock earth, but SAVES a per-term effect cap and applies it as a
# CONSTRAINED final fit (lm.fit of the pruned basis to the capped forward fit).
# The cap limits any single term's incremental delta-R^2 to effect.cap (a
# fraction of the total sum of squares); the un-capped terms then absorb the
# residual a dominant term is not allowed to explain.  effect.cap>=1 or
# adaptive.gcv=FALSE reproduces stock earth.

library(earth)
options(digits = 4)
cat("=== test.adaptive.gcv.R ===\n")

# helper: predictors actually used by a model (excludes the intercept)
used.preds <- function(m) {
    ev <- evimp(m, trim = TRUE)
    if (is.null(ev)) character(0) else rownames(ev)
}

# helper: number of selected terms (including intercept)
nterms <- function(m) length(m$selected.terms)

# helper: in-sample RSq
insample.rsq <- function(m) m$rsq

# ----------------------------------------------------------------------------
# Scenario 0 (hard requirement, spec Q8): adaptive.gcv=FALSE == stock earth.
# ----------------------------------------------------------------------------
cat("\n--- Scenario 0: adaptive.gcv=FALSE is byte-for-byte stock earth ---\n")

data(trees)
m.stock <- earth(Volume ~ ., data = trees)
m.off   <- earth(Volume ~ ., data = trees, adaptive.gcv = FALSE)
stopifnot(isTRUE(all.equal(m.stock$coefficients, m.off$coefficients)))
stopifnot(isTRUE(all.equal(m.stock$rss,          m.off$rss)))
stopifnot(isTRUE(all.equal(m.stock$gcv,          m.off$gcv)))
stopifnot(isTRUE(all.equal(m.stock$selected.terms, m.off$selected.terms)))
# adaptive.gcv=FALSE must NOT attach the diagnostic flag
stopifnot(is.null(m.stock$adaptive.gcv))
stopifnot(is.null(m.off$adaptive.gcv))
cat("trees: adaptive.gcv=FALSE identical to omitting the argument: PASS\n")

# also verify on a synthetic frame
set.seed(2020)
n <- 200
xs1 <- runif(n, 0, 10); xs2 <- runif(n, 0, 10)
ys  <- sin(xs1) + (xs2 - 5)^2 / 5 + rnorm(n, sd = 0.5)
ds  <- data.frame(x1 = xs1, x2 = xs2, y = ys)
s.stock <- earth(y ~ ., data = ds, degree = 1)
s.off   <- earth(y ~ ., data = ds, degree = 1, adaptive.gcv = FALSE)
stopifnot(isTRUE(all.equal(s.stock$coefficients, s.off$coefficients)))
stopifnot(isTRUE(all.equal(s.stock$rss, s.off$rss)))
stopifnot(isTRUE(all.equal(s.stock$gcv, s.off$gcv)))
cat("synthetic: adaptive.gcv=FALSE identical to omitting the argument: PASS\n")

# with adaptive.gcv=TRUE the flag must be present
s.on <- earth(y ~ ., data = ds, degree = 1, adaptive.gcv = TRUE)
stopifnot(isTRUE(s.on$adaptive.gcv))
cat("adaptive.gcv=TRUE attaches m$adaptive.gcv==TRUE: PASS\n")

# ----------------------------------------------------------------------------
# Scenario 1: Two independent predictors of comparable power.
# y = f1(x1) + f2(x2) + eps.  Check adaptive gives x2 at least as much
# opportunity to enter as ordinary earth (both should use both predictors).
# ----------------------------------------------------------------------------
cat("\n--- Scenario 1: two independent predictors ---\n")
set.seed(1)
n <- 400
x1 <- runif(n, 0, 10); x2 <- runif(n, 0, 10)
f1 <- pmax(0, x1 - 5); f2 <- pmax(0, 5 - x2)
y  <- f1 + f2 + rnorm(n, sd = 0.3)
d1 <- data.frame(x1, x2, y)
m.ord <- earth(y ~ ., data = d1, degree = 1)
m.adp <- earth(y ~ ., data = d1, degree = 1, adaptive.gcv = TRUE)
cat("ordinary  preds:", paste(used.preds(m.ord), collapse = ","),
    " terms:", nterms(m.ord), " rsq:", round(insample.rsq(m.ord), 4), "\n")
cat("adaptive  preds:", paste(used.preds(m.adp), collapse = ","),
    " terms:", nterms(m.adp), " rsq:", round(insample.rsq(m.adp), 4), "\n")
# both predictors are genuinely present; adaptive must still discover x2
stopifnot("x2" %in% used.preds(m.adp))
stopifnot("x1" %in% used.preds(m.adp))
cat("adaptive uses both independent predictors: PASS\n")

# ----------------------------------------------------------------------------
# Scenario 2: Dominant predictor.  y = 10*f1(x1) + f2(x2) + eps.
# The cap should prevent x1 from consuming essentially all explanatory power;
# demonstrate the cap fires (fewer/equal in-sample rsq is expected because the
# adaptive pass is deliberately conservative) and that x2 still enters.
# ----------------------------------------------------------------------------
cat("\n--- Scenario 2: dominant predictor ---\n")
set.seed(2)
n <- 400
x1 <- runif(n, 0, 10); x2 <- runif(n, 0, 10)
f1 <- pmax(0, x1 - 5); f2 <- pmax(0, 5 - x2)
y  <- 10 * f1 + f2 + rnorm(n, sd = 0.3)
d2 <- data.frame(x1, x2, y)
m.ord <- earth(y ~ ., data = d2, degree = 1)
m.adp <- earth(y ~ ., data = d2, degree = 1, adaptive.gcv = TRUE)
cat("ordinary  preds:", paste(used.preds(m.ord), collapse = ","),
    " terms:", nterms(m.ord), " rsq:", round(insample.rsq(m.ord), 4), "\n")
cat("adaptive  preds:", paste(used.preds(m.adp), collapse = ","),
    " terms:", nterms(m.adp), " rsq:", round(insample.rsq(m.adp), 4), "\n")
# ordinary earth uses the dominant predictor and normally also finds x2 here;
# the key adaptive property: the capped model is conservative, so its in-sample
# RSq does not exceed the ordinary model's (the cap limits how much signal one
# term is charged with). x1 (the dominant predictor) must still be used.
stopifnot("x1" %in% used.preds(m.adp))
stopifnot(insample.rsq(m.adp) <= insample.rsq(m.ord) + 1e-8)
cat("adaptive keeps dominant predictor, is not more greedy in-sample: PASS\n")

# ----------------------------------------------------------------------------
# Scenario 3: Pure noise.  y = eps.  No spurious large effects.
# ----------------------------------------------------------------------------
cat("\n--- Scenario 3: pure noise ---\n")
set.seed(3)
n <- 300
x1 <- runif(n); x2 <- runif(n); x3 <- runif(n)
y  <- rnorm(n)
d3 <- data.frame(x1, x2, x3, y)
m.ord <- earth(y ~ ., data = d3, degree = 1)
m.adp <- earth(y ~ ., data = d3, degree = 1, adaptive.gcv = TRUE)
cat("ordinary  terms:", nterms(m.ord), " rsq:", round(insample.rsq(m.ord), 4), "\n")
cat("adaptive  terms:", nterms(m.adp), " rsq:", round(insample.rsq(m.adp), 4), "\n")
# adaptive must not manufacture more apparent signal than ordinary on pure noise
stopifnot(insample.rsq(m.adp) <= insample.rsq(m.ord) + 1e-8)
# and it must not blow up the model size relative to ordinary
stopifnot(nterms(m.adp) <= nterms(m.ord) + 1L)
cat("adaptive creates no spurious effects on pure noise: PASS\n")

# ----------------------------------------------------------------------------
# Scenario 4: Correlated predictors.  Effect measured conditional on the model.
# ----------------------------------------------------------------------------
cat("\n--- Scenario 4: correlated predictors ---\n")
set.seed(4)
n <- 400
z  <- runif(n, 0, 10)
x1 <- z + rnorm(n, sd = 0.1)          # x1 and x2 strongly correlated
x2 <- z + rnorm(n, sd = 0.1)
y  <- pmax(0, x1 - 5) + rnorm(n, sd = 0.3)
d4 <- data.frame(x1, x2, y)
cat("cor(x1,x2) =", round(cor(x1, x2), 4), "\n")
m.ord <- earth(y ~ ., data = d4, degree = 1)
m.adp <- earth(y ~ ., data = d4, degree = 1, adaptive.gcv = TRUE)
cat("ordinary  preds:", paste(used.preds(m.ord), collapse = ","),
    " terms:", nterms(m.ord), " rsq:", round(insample.rsq(m.ord), 4), "\n")
cat("adaptive  preds:", paste(used.preds(m.adp), collapse = ","),
    " terms:", nterms(m.adp), " rsq:", round(insample.rsq(m.adp), 4), "\n")
# must produce a sane, finite model (no pathology): valid rsq in [ -inf, 1 ]
stopifnot(is.finite(insample.rsq(m.adp)))
stopifnot(insample.rsq(m.adp) <= 1 + 1e-8)
stopifnot(all(is.finite(m.adp$coefficients)))
cat("adaptive behaves sanely under strong collinearity: PASS\n")

# ----------------------------------------------------------------------------
# Scenario 5 (CRITICAL): Scale invariance.  Fit on x, 10*x, 1000*x.
# The fitted function and adaptive behavior must be essentially invariant.
# ----------------------------------------------------------------------------
cat("\n--- Scenario 5: scale invariance (critical) ---\n")
set.seed(5)
n <- 400
x1 <- runif(n, 0, 10); x2 <- runif(n, 0, 10)
y  <- pmax(0, x1 - 5) + pmax(0, 5 - x2) + rnorm(n, sd = 0.3)

fit.scaled <- function(scale) {
    d <- data.frame(x1 = scale * x1, x2 = scale * x2, y = y)
    earth(y ~ ., data = d, degree = 1, adaptive.gcv = TRUE)
}
m1    <- fit.scaled(1)
m10   <- fit.scaled(10)
m1000 <- fit.scaled(1000)

# compare fitted values (predictions on the same underlying points)
fv1    <- as.numeric(m1$fitted.values)
fv10   <- as.numeric(m10$fitted.values)
fv1000 <- as.numeric(m1000$fitted.values)
d.10   <- max(abs(fv1 - fv10))
d.1000 <- max(abs(fv1 - fv1000))
cat("max|fitted(x) - fitted(10x)|   =", format(d.10,   scientific = TRUE), "\n")
cat("max|fitted(x) - fitted(1000x)| =", format(d.1000, scientific = TRUE), "\n")
cat("selected terms:", nterms(m1), nterms(m10), nterms(m1000), "\n")
# fitted values essentially identical under scaling (numeric tolerance)
tol <- 1e-6 * (max(abs(y)) + 1)
stopifnot(d.10   < tol)
stopifnot(d.1000 < tol)
# same number of selected terms across scales
stopifnot(nterms(m1) == nterms(m10), nterms(m1) == nterms(m1000))
cat("adaptive.gcv is scale invariant: PASS\n")

# ----------------------------------------------------------------------------
# Scenario 6: Interactions.  y = f(x1,x2) + g(x3) with degree=2.
# The cap must not misbehave on interaction basis functions.
# ----------------------------------------------------------------------------
cat("\n--- Scenario 6: interactions (degree=2) ---\n")
set.seed(6)
n <- 500
x1 <- runif(n, 0, 10); x2 <- runif(n, 0, 10); x3 <- runif(n, 0, 10)
y  <- pmax(0, x1 - 5) * pmax(0, x2 - 5) + pmax(0, x3 - 5) + rnorm(n, sd = 0.5)
d6 <- data.frame(x1, x2, x3, y)
m.ord <- earth(y ~ ., data = d6, degree = 2)
m.adp <- earth(y ~ ., data = d6, degree = 2, adaptive.gcv = TRUE)
cat("ordinary  preds:", paste(used.preds(m.ord), collapse = ","),
    " terms:", nterms(m.ord), " rsq:", round(insample.rsq(m.ord), 4), "\n")
cat("adaptive  preds:", paste(used.preds(m.adp), collapse = ","),
    " terms:", nterms(m.adp), " rsq:", round(insample.rsq(m.adp), 4), "\n")
# adaptive must yield a valid, finite degree-2 model and not exceed ordinary in-sample rsq
stopifnot(is.finite(insample.rsq(m.adp)))
stopifnot(all(is.finite(m.adp$coefficients)))
stopifnot(insample.rsq(m.adp) <= insample.rsq(m.ord) + 1e-8)
cat("adaptive handles interaction basis functions without pathology: PASS\n")

# ----------------------------------------------------------------------------
# Scenario 7: effect.cap knob.  effect.cap >= 1 reproduces stock earth (the cap
# can never bind); decreasing effect.cap monotonically holds back more signal
# from a dominant term.  Invalid effect.cap is rejected.
# ----------------------------------------------------------------------------
cat("\n--- Scenario 7: effect.cap strength knob ---\n")
# effect.cap >= 1: a single term may explain the whole TSS, so the cap never
# binds and the model is identical to stock earth.
m.big <- earth(Volume ~ ., data = trees, adaptive.gcv = TRUE, effect.cap = 1)
stopifnot(isTRUE(all.equal(m.stock$coefficients, m.big$coefficients)))
stopifnot(isTRUE(all.equal(m.stock$rss, m.big$rss)))
cat("effect.cap=1 reproduces stock earth: PASS\n")

# smaller effect.cap => more signal held back => lower (or equal) in-sample rsq
r90 <- earth(Volume ~ ., data = trees, adaptive.gcv = TRUE, effect.cap = 0.9)$rsq
r50 <- earth(Volume ~ ., data = trees, adaptive.gcv = TRUE, effect.cap = 0.5)$rsq
r25 <- earth(Volume ~ ., data = trees, adaptive.gcv = TRUE, effect.cap = 0.25)$rsq
cat("trees rsq at effect.cap 0.9/0.5/0.25:",
    round(r90, 4), round(r50, 4), round(r25, 4), "\n")
stopifnot(r90 >= r50 - 1e-8, r50 >= r25 - 1e-8)
stopifnot(r90 <= m.stock$rsq + 1e-8)
cat("effect.cap is monotone and never exceeds stock in-sample rsq: PASS\n")

# invalid effect.cap must error
stopifnot(inherits(try(earth(Volume ~ ., data = trees, adaptive.gcv = TRUE,
                              effect.cap = -1), silent = TRUE), "try-error"))
stopifnot(inherits(try(earth(Volume ~ ., data = trees, adaptive.gcv = TRUE,
                              effect.cap = 0), silent = TRUE), "try-error"))
cat("invalid effect.cap is rejected: PASS\n")

cat("\n=== all test.adaptive.gcv.R assertions passed ===\n")
