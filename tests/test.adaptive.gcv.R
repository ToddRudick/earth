# test.adaptive.gcv.R
#
# Validation suite for the experimental adaptive.gcv effect cap.
#
# STAGE 1 (FEAT-004): the effect cap is now GCV / per-term-complexity ADAPTIVE.
# It is no longer a fixed fraction of the total variance.  The budget for the
# incremental delta-RSS a newly admitted term may realise is
#
#   DeltaRssMax = BreakEven + slackFactor * (RssDelta - BreakEven)
#
# where BreakEven is the GCV break-even reduction computed with an EXPLICIT
# per-term knot charge (deltaKnots = 0 for a linear/linpreds term, deltaKnots =
# 1 for a hinge term), RssDelta is the unconstrained OLS effect (ceiling), and
# slackFactor = (1 - clamp(Cost1))^gamma with gamma = 1/effect.cap - 1 and
# Cost1 = (nOldUsedTerms + deltaTerms + Penalty*(nKnotsOld + deltaKnots))/nCases.
# effect.cap >= 1 (or adaptive.gcv = FALSE) forces slackFactor = 1 =>
# DeltaRssMax = RssDelta => CapScale = 1 => byte-for-byte stock earth.
#
# The Stage-1 hypothesis this suite must pin down: for effect.cap < 1 a HINGE
# term (1 knot, higher Cost1) gets a SMALLER budget / CapScale than a LINEAR
# term (0 knots, lower Cost1) carrying the same OLS effect, so the linear form
# is shrunk LESS.  This is form-aware: it FAILS if the per-term knot charge is
# reverted to the model-wide averaged (nUsedTerms-1)/2 approximation.
#
# This file implements the six synthetic scenarios from
# adaptive_gcv_effect_cap_coding_agent.md ("Validation plan") plus hard
# invariant checks (adaptive.gcv=FALSE and effect.cap>=1 both == stock earth)
# and the hinge-vs-linear divergence scenario.  Each scenario asserts the
# intended qualitative outcome with stopifnot().  There is deliberately NO
# committed .Rout.save: the suite is self-checking.
#
# Mechanism (see man/earth.Rd and src/earth.c): the forward pass admits terms
# exactly as stock earth, but SAVES the per-term capped contribution and applies
# it as a CONSTRAINED final fit (lm.fit of the pruned basis to the capped
# forward fit yHatCap).  Term SELECTION is unchanged; only the realised effect
# of a saturated term is held below its unconstrained OLS effect, so the
# un-capped terms absorb the residual a dominant term is not allowed to explain.

library(earth)
options(digits = 4)
cat("=== test.adaptive.gcv.R ===\n")

# ----------------------------------------------------------------------------
# Helper: extract the per-term cap diagnostics from a trace>=6 fit.
# Returns a data.frame with one row per admitted term (from the "effectcap CAP"
# and "effectcap diag" lines emitted by ForwardPass in src/earth.c).  The CAP
# line is only printed when a term is actually saturated (RssDelta > DeltaRssMax)
# and can be prefixed by other trace output on the same line, so we grep for the
# marker substring and parse the numeric fields by name.
# ----------------------------------------------------------------------------
parse.field <- function(line, key) {
    # match "<key> <number>" allowing scientific notation and sign
    m <- regmatches(line, regexpr(
        paste0(key, "\\s+[-+0-9.eE]+"), line))
    if (length(m) == 0) return(NA_real_)
    as.numeric(sub(paste0("^", key, "\\s+"), "", m))
}

cap.diag <- function(..., effect.cap = 0.5) {
    out <- capture.output(fit <- earth(..., adaptive.gcv = TRUE,
                                       effect.cap = effect.cap, trace = 6))
    cap.lines <- grep("effectcap CAP:", out, value = TRUE)
    # split each concatenated line at the marker so leading trace text is dropped
    cap.lines <- sub(".*effectcap CAP:", "effectcap CAP:", cap.lines)
    if (length(cap.lines) == 0)
        return(list(fit = fit, cap = data.frame()))
    df <- data.frame(
        iTerm      = as.integer(sapply(cap.lines, parse.field, key = "iTerm")),
        dRSSols    = sapply(cap.lines, parse.field, key = "dRSS\\(ols\\)"),
        dRSSmax    = sapply(cap.lines, parse.field, key = "dRSSmax"),
        scale      = sapply(cap.lines, parse.field, key = "scale"),
        deltaKnots = as.integer(sapply(cap.lines, parse.field, key = "deltaKnots")),
        Cost1      = sapply(cap.lines, parse.field, key = "Cost1"),
        BreakEven  = sapply(cap.lines, parse.field, key = "BreakEven"),
        slackFactor= sapply(cap.lines, parse.field, key = "slackFactor"),
        row.names  = NULL, stringsAsFactors = FALSE)
    list(fit = fit, cap = df)
}

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
# Scenario 0 (hard requirement, spec Q8): adaptive.gcv=FALSE == stock earth,
# AND adaptive.gcv=TRUE with effect.cap>=1 == stock earth (the cap can never
# bind under the new adaptive formula: gamma<=0 => slackFactor==1 =>
# DeltaRssMax==RssDelta => CapScale==1).
# ----------------------------------------------------------------------------
cat("\n--- Scenario 0: stock-earth invariants (OFF, and ON with effect.cap>=1) ---\n")

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

# adaptive.gcv=TRUE with effect.cap>=1 must ALSO be byte-for-byte stock earth
# (the adaptive budget's early fast-path forces DeltaRssMax==RssDelta).
m.cap15 <- earth(Volume ~ ., data = trees, adaptive.gcv = TRUE, effect.cap = 1.5)
m.cap1  <- earth(Volume ~ ., data = trees, adaptive.gcv = TRUE, effect.cap = 1)
stopifnot(isTRUE(all.equal(unname(m.stock$coefficients),
                           unname(m.cap15$coefficients), tol = 1e-9)))
stopifnot(isTRUE(all.equal(m.stock$rss, m.cap15$rss, tol = 1e-9)))
stopifnot(isTRUE(all.equal(m.stock$gcv, m.cap15$gcv, tol = 1e-9)))
stopifnot(isTRUE(all.equal(unname(m.stock$coefficients),
                           unname(m.cap1$coefficients), tol = 1e-9)))
cat("trees: adaptive.gcv=TRUE + effect.cap>=1 identical to stock earth: PASS\n")

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

# synthetic frame: effect.cap>=1 with adaptive.gcv=TRUE must also match stock
s.cap15 <- earth(y ~ ., data = ds, degree = 1, adaptive.gcv = TRUE, effect.cap = 1.5)
stopifnot(isTRUE(all.equal(unname(s.stock$coefficients),
                           unname(s.cap15$coefficients), tol = 1e-9)))
stopifnot(isTRUE(all.equal(s.stock$rss, s.cap15$rss, tol = 1e-9)))
cat("synthetic: adaptive.gcv=TRUE + effect.cap>=1 identical to stock earth: PASS\n")

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

# ----------------------------------------------------------------------------
# Scenario 8 (CORE STAGE-1 hypothesis): hinge vs linear per-term knot charge.
#
# Construct data with a DOMINANT, essentially linear predictor x1 plus a weaker
# secondary predictor x2.  Fit the adaptive cap (effect.cap<1) TWICE:
#   (a) x1 free to enter as a HINGE (default), and
#   (b) x1 FORCED LINEAR via linpreds.
# The per-term knot charge is deltaKnots=1 for the hinge and deltaKnots=0 for
# the linear form.  Because Cost1 rises with deltaKnots and slackFactor =
# (1-clamp(Cost1))^gamma DECREASES with Cost1, the linear form must receive a
# LARGER budget (DeltaRssMax) and a LARGER CapScale on x1's dominant term, i.e.
# it is shrunk LESS.  This assertion FAILS if the per-term knot charge is
# reverted to the model-wide averaged (nUsedTerms-1)/2 approximation, which is
# form-blind (same charge for hinge and linear).
# ----------------------------------------------------------------------------
cat("\n--- Scenario 8: hinge vs linear per-term knot charge (Stage-1 core) ---\n")
set.seed(101)
n  <- 60
x1 <- runif(n, 0, 10)                 # dominant, linear signal
x2 <- runif(n, 0, 10)                 # weaker, hinge-shaped secondary signal
y  <- 3 * x1 + 0.5 * pmax(0, x2 - 5) + rnorm(n, sd = 0.5)
d8 <- data.frame(x1, x2, y)

res.hinge <- cap.diag(y ~ ., data = d8, degree = 1, effect.cap = 0.5)
res.lin   <- cap.diag(y ~ ., data = d8, degree = 1, effect.cap = 0.5,
                      linpreds = "x1")

# The dominant predictor is the FIRST admitted term (iTerm 1) in both fits.
cap.h <- res.hinge$cap
cap.l <- res.lin$cap
stopifnot(nrow(cap.h) >= 1L, nrow(cap.l) >= 1L)
row.h <- cap.h[cap.h$iTerm == 1L, ][1, ]
row.l <- cap.l[cap.l$iTerm == 1L, ][1, ]
stopifnot(!is.na(row.h$scale), !is.na(row.l$scale))

cat(sprintf("x1 as HINGE : deltaKnots %d  Cost1 %.5g  slackFactor %.5g  dRSSmax %.5g  CapScale %.5g\n",
            row.h$deltaKnots, row.h$Cost1, row.h$slackFactor, row.h$dRSSmax, row.h$scale))
cat(sprintf("x1 as LINEAR: deltaKnots %d  Cost1 %.5g  slackFactor %.5g  dRSSmax %.5g  CapScale %.5g\n",
            row.l$deltaKnots, row.l$Cost1, row.l$slackFactor, row.l$dRSSmax, row.l$scale))

# per-term knot charge: hinge charges 1 knot, forced-linear charges 0
stopifnot(row.h$deltaKnots == 1L)
stopifnot(row.l$deltaKnots == 0L)
# the cheaper (linear, 0-knot) form must carry a LOWER complexity cost, a
# LARGER slack, a LARGER effect budget, and be shrunk LESS than the hinge form
stopifnot(row.l$Cost1       <  row.h$Cost1)
stopifnot(row.l$slackFactor >  row.h$slackFactor)
stopifnot(row.l$dRSSmax     >  row.h$dRSSmax)
stopifnot(row.l$scale       >  row.h$scale)
cat("linear (0-knot) x1 gets larger budget/CapScale than hinge (1-knot) x1: PASS\n")

# --------------------------------------------------------------------------
# THE regression guard for the EXPLICIT per-term knot charge (review Issue 1).
#
# The inequality assertions above are NOT sensitive to the specific Stage-1
# change (linear-term knot charge 0.5 -> 0).  They are dominated by deltaTerms
# (a hinge term-pair adds 2 terms vs 1 for a linear term), which the OLD
# model-wide averaged (nUsedTerms-1)/2 formula ALREADY encoded via nUsedTerms;
# so linear > hinge holds under BOTH the explicit and the averaged charge and
# the inequalities cannot detect a revert.  To actually PIN the explicit
# per-term knot charge we assert the forced-LINEAR dominant term's printed
# Cost1 equals the EXPLICIT value and NOT the averaged value.
#
# For the first admitted term the pre-admission complexity is nOldUsedTerms = 1
# (intercept only).  The forced-linear form has deltaTerms = 1 and
# deltaKnots = 0, so
#     EXPLICIT  Cost1 = (nOldUsedTerms + deltaTerms + Penalty*(0 + deltaKnots))/n
#                     = (1 + 1 + Penalty*0)/n = 2/n            (Penalty drops out)
# whereas the OLD averaged formula with nUsedTerms = 2 would give
#     AVERAGED  Cost1 = (nUsedTerms + Penalty*(nUsedTerms-1)/2)/n
#                     = (2 + Penalty*0.5)/n.
# Because deltaKnots = 0 for the linear form, Penalty*deltaKnots = 0 and the
# explicit value is 2/n for ANY penalty; the averaged charge, in contrast,
# would add Penalty*0.5.  Asserting Cost1 == 2/n (tight tol) therefore FAILS
# under the revert to the averaged approximation -- this is the real regression
# guard for the budget-affecting Stage-1 change.
n8      <- nrow(d8)
# earth's default penalty: 2 for degree=1, 3 for degree>1 (this scenario is
# degree=1, so penalty = 2).
penalty8 <- 2
cost1.explicit <- 2 / n8                             # (1 + 1 + penalty*0)/n
cost1.averaged <- (2 + penalty8 * 0.5) / n8          # (nUsedTerms + P*(nUsed-1)/2)/n
cat(sprintf("linear x1 Cost1: observed %.6g  explicit-expected %.6g  averaged(revert) %.6g\n",
            row.l$Cost1, cost1.explicit, cost1.averaged))
# the two formulas must be genuinely distinguishable (self-documenting intent)
stopifnot(abs(cost1.explicit - cost1.averaged) > 1e-6)
# and the shipped code must print the EXPLICIT value, not the averaged one
stopifnot(isTRUE(all.equal(row.l$Cost1, cost1.explicit, tolerance = 1e-4)))
stopifnot(abs(row.l$Cost1 - cost1.averaged) > 1e-4)
cat("linear x1 Cost1 pins the EXPLICIT per-term knot charge (2/n), not the averaged value: PASS\n")

# Cross-check the divergence via the retained effect on x1 in the fitted model:
# the forced-linear representation should retain MORE of x1's variance than the
# hinge representation once each is shrunk by its own CapScale.
eff.on.x1 <- function(fit) {
    # variance of the fitted contribution attributable to the x1 terms
    bx <- fit$bx
    keep <- fit$selected.terms
    nm   <- rownames(fit$dirs)[keep]
    dirs <- fit$dirs[keep, , drop = FALSE]
    x1col <- which(colnames(dirs) == "x1")
    uses.x1 <- dirs[, x1col] != 0
    co <- fit$coefficients[, 1]
    contrib <- bx[, uses.x1, drop = FALSE] %*% co[uses.x1]
    var(as.numeric(contrib))
}
v.h <- eff.on.x1(res.hinge$fit)
v.l <- eff.on.x1(res.lin$fit)
cat(sprintf("retained var of x1 contribution: hinge %.5g  linear %.5g\n", v.h, v.l))
stopifnot(v.l > v.h)
cat("forced-linear x1 retains more effect than hinge x1 under the cap: PASS\n")

cat("\n=== all test.adaptive.gcv.R assertions passed ===\n")
