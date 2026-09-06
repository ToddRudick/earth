# adaptive.gcv.comparison.R
#
# Empirical in-sample vs out-of-sample (OOS) comparison of the experimental
# adaptive.gcv effect cap (adaptive.gcv=TRUE) against ordinary earth
# (adaptive.gcv=FALSE, the default), for FEAT-003.
#
# Two independent OOS engines are used so the result does not depend on any one
# resampling method:
#
#   (1) caret::bagEarth  - bagged (bootstrap-aggregated) earth. bagEarth passes
#       ... through to earth(), so we fit bagged earth both WITHOUT adaptive.gcv
#       (default) and WITH adaptive.gcv=TRUE, on a fixed train/test split, and
#       score on the held-out test set. If caret / bagEarth is unavailable, a
#       documented direct-earth bootstrap-aggregating fallback is used instead
#       (never skipped).
#
#   (2) earth's built-in cross-validation (nfold/ncross) - earth performs its
#       own k-fold CV and reports cross-validated RSq/metrics. This is fully
#       independent of caret.
#
# Datasets (shipped with earth or base R, established MARS / earth-vignette
# examples):
#   - ozone1    : canonical MARS / earth-vignette regression example (O3 ~ 9 preds)
#   - trees     : base R regression (Volume ~ Girth + Height)
#   - mtcars    : base R regression (mpg ~ .)
#   - etitanic  : used in earth examples; BINARY response (survived) -> classification
#
# For regression datasets we report RMSE and R^2 (in-sample and OOS).
# For etitanic (binary) we additionally report accuracy and Brier score.
# At least one degree=2 run is included (ozone1 and etitanic use degree=2).
#
# Output: writes per-dataset markdown files and a combined summary report under
# the repository (paths printed at the end). Reproducible via fixed seeds.

suppressWarnings(suppressMessages(library(earth)))
options(warn = 1)

set.seed(2024)

have.caret <- requireNamespace("caret", quietly = TRUE) &&
    exists("bagEarth", where = asNamespace("caret"))

## ---------------------------------------------------------------------------
## Output location
## ---------------------------------------------------------------------------
# Resolve repo root robustly whether run from repo root or inst/slowtests.
find.repo.root <- function() {
    cands <- c(".", "..", "../..", "/projects/sandbox/earth")
    for (d in cands)
        if (file.exists(file.path(d, "DESCRIPTION")) &&
            file.exists(file.path(d, "NEWS.md")))
            return(normalizePath(d))
    normalizePath(".")
}
repo.root <- find.repo.root()
doc.dir   <- file.path(repo.root, "doc")
dir.create(doc.dir, showWarnings = FALSE, recursive = TRUE)
cat("repo.root =", repo.root, "\n")
cat("caret bagEarth available:", have.caret, "\n\n")

## ---------------------------------------------------------------------------
## Metrics
## ---------------------------------------------------------------------------
rmse <- function(actual, pred) sqrt(mean((actual - pred)^2))
r2   <- function(actual, pred) {
    ss.res <- sum((actual - pred)^2)
    ss.tot <- sum((actual - mean(actual))^2)
    1 - ss.res / ss.tot
}
accuracy   <- function(actual01, prob) mean((prob >= 0.5) == (actual01 == 1))
brier      <- function(actual01, prob) mean((prob - actual01)^2)

## ---------------------------------------------------------------------------
## Bagged earth on a pre-built NUMERIC design matrix (factors already expanded)
## so bootstrap samples and the test set always share identical columns.
## caret::bagEarth if available, else a documented direct fallback.
## Returns a predictor function: numeric design matrix -> numeric prediction.
## ---------------------------------------------------------------------------
bagged.earth.fit <- function(xmat, yv, B = 30, adaptive.gcv = FALSE, degree = 1) {
    xmat <- as.data.frame(xmat)
    if (have.caret) {
        # bagEarth passes ... through to earth(); pass adaptive.gcv + degree
        fit <- caret::bagEarth(x = xmat, y = yv, B = B,
                               degree = degree, adaptive.gcv = adaptive.gcv)
        pred.fun <- function(newx) as.numeric(predict(fit, as.data.frame(newx)))
        attr(pred.fun, "engine") <- "caret::bagEarth"
        attr(pred.fun, "fit")    <- fit
        return(pred.fun)
    }
    # ---- fallback: direct bootstrap-aggregated earth ----
    n    <- nrow(xmat)
    fits <- vector("list", B)
    for (b in seq_len(B)) {
        idx <- sample.int(n, n, replace = TRUE)
        fits[[b]] <- earth(x = xmat[idx, , drop = FALSE], y = yv[idx],
                           degree = degree, adaptive.gcv = adaptive.gcv)
    }
    pred.fun <- function(newx) {
        newx  <- as.data.frame(newx)
        preds <- vapply(fits, function(f) as.numeric(predict(f, newx)),
                        numeric(nrow(newx)))
        if (is.null(dim(preds))) mean(preds) else rowMeans(preds)
    }
    attr(pred.fun, "engine") <- "direct-earth bagging (fallback)"
    attr(pred.fun, "fits")   <- fits
    return(pred.fun)
}

## ---------------------------------------------------------------------------
## Single-model summary (terms + coefficients) for the per-dataset .md file.
## ---------------------------------------------------------------------------
coef.table.md <- function(m) {
    co <- m$coefficients
    nm <- rownames(co)
    val <- co[, 1]
    lines <- c("| term | coefficient |", "|------|-------------|")
    for (i in seq_along(nm))
        lines <- c(lines, sprintf("| `%s` | %.6g |", nm[i], val[i]))
    paste(lines, collapse = "\n")
}

## ---------------------------------------------------------------------------
## Run one dataset: holdout OOS via bagged earth (ON/OFF) + earth built-in CV,
## plus single-model term/coef listings. Writes a per-dataset .md file.
## ---------------------------------------------------------------------------
run.dataset <- function(name, form, data, degree = 1,
                        is.binary = FALSE, B = 30, nfold = 5, ncross = 3) {
    cat("==== dataset:", name, "(degree =", degree, ") ====\n")

    # Build a single numeric design matrix (factors expanded once) so every
    # bootstrap fit and the held-out test set share identical columns.
    mf    <- model.frame(form, data)
    y.all <- model.response(mf)
    x.all <- model.matrix(form, mf)
    x.all <- x.all[, colnames(x.all) != "(Intercept)", drop = FALSE]
    if (is.binary) y01.all <- as.integer(y.all) - min(as.integer(y.all))

    set.seed(2024)
    n     <- nrow(x.all)
    tr    <- sample.int(n, round(0.7 * n))
    x.tr  <- x.all[tr, , drop = FALSE];  x.te <- x.all[-tr, , drop = FALSE]
    y.tr  <- y.all[tr];                  y.te <- y.all[-tr]
    if (is.binary) { y01.tr <- y01.all[tr]; y01.te <- y01.all[-tr] }

    ## --- Engine 1: bagged earth OFF vs ON, holdout OOS ---
    yv.tr <- if (is.binary) y01.tr else y.tr
    set.seed(2024)
    pf.off <- bagged.earth.fit(x.tr, yv.tr, B = B, adaptive.gcv = FALSE, degree = degree)
    set.seed(2024)
    pf.on  <- bagged.earth.fit(x.tr, yv.tr, B = B, adaptive.gcv = TRUE,  degree = degree)
    engine <- attr(pf.off, "engine")

    p.off <- pf.off(x.te)
    p.on  <- pf.on(x.te)

    if (is.binary) {
        p.off <- pmin(1, pmax(0, p.off)); p.on <- pmin(1, pmax(0, p.on))
        bag <- data.frame(
            setting  = c("ordinary (OFF)", "adaptive (ON)"),
            oos_rmse = c(rmse(y01.te, p.off),     rmse(y01.te, p.on)),
            oos_acc  = c(accuracy(y01.te, p.off), accuracy(y01.te, p.on)),
            oos_brier= c(brier(y01.te, p.off),    brier(y01.te, p.on)))
    } else {
        bag <- data.frame(
            setting  = c("ordinary (OFF)", "adaptive (ON)"),
            oos_rmse = c(rmse(y.te, p.off), rmse(y.te, p.on)),
            oos_r2   = c(r2(y.te, p.off),   r2(y.te, p.on)))
    }

    ## --- Engine 2: earth built-in CV (independent of caret) ---
    cv.metric <- function(adaptive) {
        set.seed(2024)
        m <- earth(form, data = data, degree = degree,
                   nfold = nfold, ncross = ncross,
                   adaptive.gcv = adaptive,
                   glm = if (is.binary) list(family = binomial) else NULL)
        # cross-validated stats live in m$cv.oof.rsq.tab / m$cv.list summaries
        list(model = m,
             insample.rsq = m$rsq,
             cv.rsq = if (!is.null(m$cv.rsq.tab))
                          m$cv.rsq.tab[nrow(m$cv.rsq.tab), "mean"] else NA_real_,
             cv.class.rate = if (!is.null(m$cv.class.rate.tab))
                          m$cv.class.rate.tab[nrow(m$cv.class.rate.tab), "mean"] else NA_real_)
    }
    cv.off <- cv.metric(FALSE)
    cv.on  <- cv.metric(TRUE)

    ## --- single (non-bagged) models on full data for terms/coefs listing ---
    m.off.full <- earth(form, data = data, degree = degree, adaptive.gcv = FALSE,
                        glm = if (is.binary) list(family = binomial) else NULL)
    m.on.full  <- earth(form, data = data, degree = degree, adaptive.gcv = TRUE,
                        glm = if (is.binary) list(family = binomial) else NULL)

    ## --- write per-dataset markdown ---
    md <- c(
        sprintf("# Adaptive GCV effect cap: %s", name),
        "",
        sprintf("- Response formula: `%s`", deparse(form)),
        sprintf("- Rows: %d (train %d / test %d, 70/30 holdout, seed 2024)",
                n, nrow(x.tr), nrow(x.te)),
        sprintf("- earth `degree` = %d", degree),
        sprintf("- Task type: %s", if (is.binary) "binary classification" else "regression"),
        sprintf("- OOS bagging engine: %s (B = %d)", engine, B),
        sprintf("- earth built-in CV: nfold = %d, ncross = %d", nfold, ncross),
        "",
        "## Out-of-sample: bagged earth (ordinary vs adaptive)",
        "",
        knitr.free.table(bag),
        "",
        "## Cross-validation (earth built-in, independent of caret)",
        "",
        knitr.free.table(data.frame(
            setting        = c("ordinary (OFF)", "adaptive (ON)"),
            insample_rsq   = c(cv.off$insample.rsq, cv.on$insample.rsq),
            cv_rsq         = c(cv.off$cv.rsq,       cv.on$cv.rsq),
            cv_class_rate  = c(cv.off$cv.class.rate, cv.on$cv.class.rate))),
        "",
        "## Selected terms and coefficients (single model on full data)",
        "",
        "### ordinary earth (adaptive.gcv = FALSE)",
        sprintf("Selected %d of %d terms; in-sample RSq = %.4f, GCV = %.5g",
                length(m.off.full$selected.terms),
                nrow(m.off.full$dirs), m.off.full$rsq, m.off.full$gcv),
        "",
        coef.table.md(m.off.full),
        "",
        "### adaptive earth (adaptive.gcv = TRUE)",
        sprintf("Selected %d of %d terms; in-sample RSq = %.4f, GCV = %.5g",
                length(m.on.full$selected.terms),
                nrow(m.on.full$dirs), m.on.full$rsq, m.on.full$gcv),
        "",
        coef.table.md(m.on.full),
        "")
    outfile <- file.path(doc.dir, sprintf("adaptive_gcv_%s.md", name))
    writeLines(md, outfile)
    cat("wrote", outfile, "\n\n")

    # return a compact summary row list for the combined report
    list(name = name, degree = degree, is.binary = is.binary,
         engine = engine, bag = bag, cv.off = cv.off, cv.on = cv.on,
         outfile = outfile)
}

## Small helper to render a data.frame as a github-markdown table.
knitr.free.table <- function(df) {
    fmt <- function(x) {
        if (is.numeric(x)) ifelse(is.na(x), "NA", formatC(x, digits = 5, format = "g"))
        else as.character(x)
    }
    cols <- names(df)
    header <- paste0("| ", paste(cols, collapse = " | "), " |")
    sep    <- paste0("| ", paste(rep("---", length(cols)), collapse = " | "), " |")
    rows <- apply(df, 1, function(r)
        paste0("| ", paste(vapply(seq_along(r), function(i) fmt(r[[i]]), ""),
                           collapse = " | "), " |"))
    paste(c(header, sep, rows), collapse = "\n")
}

## ---------------------------------------------------------------------------
## Datasets
## ---------------------------------------------------------------------------
results <- list()

data(ozone1, package = "earth")
results[["ozone1"]] <- run.dataset(
    "ozone1", O3 ~ ., ozone1, degree = 2, is.binary = FALSE, B = 30)

data(trees)
results[["trees"]] <- run.dataset(
    "trees", Volume ~ ., trees, degree = 1, is.binary = FALSE, B = 30)

results[["mtcars"]] <- run.dataset(
    "mtcars", mpg ~ ., mtcars, degree = 1, is.binary = FALSE, B = 30)

data(etitanic, package = "earth")
et <- etitanic
et$survived <- factor(ifelse(et$survived == 1, "yes", "no"))
# earth handles factor response via glm=binomial; keep numeric 0/1 for metrics
et2 <- etitanic
results[["etitanic"]] <- run.dataset(
    "etitanic", survived ~ ., et2, degree = 2, is.binary = TRUE, B = 30)

## ---------------------------------------------------------------------------
## Combined summary report
## ---------------------------------------------------------------------------
engine.used <- results[[1]]$engine
summary.lines <- c(
    "# Adaptive GCV Effect Cap: in-sample vs out-of-sample comparison",
    "",
    "This report compares ordinary earth (`adaptive.gcv = FALSE`, the default)",
    "against the experimental adaptive GCV effect cap (`adaptive.gcv = TRUE`) on",
    "several established datasets, measuring **out-of-sample** predictive power.",
    "",
    "## Methodology",
    "",
    sprintf("- **OOS bagging engine:** %s. bagEarth passes `...` through to `earth()`, so the identical bagging procedure is run with the feature OFF and ON.", engine.used),
    "- **Holdout:** a single 70/30 train/test split per dataset (`set.seed(2024)`); bagged earth is fit on train and scored on the held-out test rows.",
    "- **Cross-validation:** earth's own k-fold CV (`nfold=5, ncross=3`, `set.seed(2024)`), fully independent of caret, reports cross-validated RSq (and classification rate for the binary response).",
    "- **Metrics:** RMSE and R^2 for regression; RMSE, accuracy and Brier score for the binary `etitanic$survived` response.",
    "- **Interactions:** `ozone1` and `etitanic` are fit with `degree = 2`.",
    "- Per-dataset details (selected terms and coefficients for both settings) are in the companion files listed below.",
    "",
    "## Datasets",
    "",
    "| dataset | task | degree | rows | notes |",
    "| --- | --- | --- | --- | --- |",
    "| ozone1 | regression | 2 | 330 | canonical MARS / earth-vignette example |",
    "| trees | regression | 1 | 31 | base R |",
    "| mtcars | regression | 1 | 32 | base R |",
    "| etitanic | classification | 2 | 1046 | used in earth examples; binary `survived` |",
    "",
    "## Out-of-sample results (bagged earth, holdout test set)",
    "")

# regression OOS table
reg <- Filter(function(r) !r$is.binary, results)
summary.lines <- c(summary.lines,
    "### Regression (RMSE / R^2 on held-out test set)",
    "",
    "| dataset | setting | OOS RMSE | OOS R^2 |",
    "| --- | --- | --- | --- |")
for (r in reg) {
    b <- r$bag
    for (i in 1:2)
        summary.lines <- c(summary.lines, sprintf("| %s | %s | %.4g | %.4g |",
            r$name, b$setting[i], b$oos_rmse[i], b$oos_r2[i]))
}

# binary OOS table
bin <- Filter(function(r) r$is.binary, results)
if (length(bin)) {
    summary.lines <- c(summary.lines, "",
        "### Classification (etitanic$survived, held-out test set)",
        "",
        "| dataset | setting | OOS RMSE | OOS accuracy | OOS Brier |",
        "| --- | --- | --- | --- | --- |")
    for (r in bin) {
        b <- r$bag
        for (i in 1:2)
            summary.lines <- c(summary.lines, sprintf("| %s | %s | %.4g | %.4g | %.4g |",
                r$name, b$setting[i], b$oos_rmse[i], b$oos_acc[i], b$oos_brier[i]))
    }
}

# built-in CV table
summary.lines <- c(summary.lines, "",
    "## Cross-validation (earth built-in, independent of caret)",
    "",
    "| dataset | setting | in-sample RSq | CV RSq | CV class-rate |",
    "| --- | --- | --- | --- | --- |")
for (r in results) {
    for (s in c("off", "on")) {
        cv <- if (s == "off") r$cv.off else r$cv.on
        lbl <- if (s == "off") "ordinary (OFF)" else "adaptive (ON)"
        summary.lines <- c(summary.lines, sprintf("| %s | %s | %.4g | %.4g | %s |",
            r$name, lbl, cv$insample.rsq, cv$cv.rsq,
            if (is.na(cv$cv.class.rate)) "NA" else formatC(cv$cv.class.rate, digits = 4, format = "g")))
    }
}

# interpretation (computed, so it stays honest to the numbers)
interp <- c("", "## Interpretation", "")
for (r in reg) {
    d <- r$bag$oos_rmse[2] - r$bag$oos_rmse[1]  # adaptive - ordinary
    verdict <- if (d < -1e-6) "IMPROVES" else if (d > 1e-6) "HURTS" else "is NEUTRAL for"
    interp <- c(interp, sprintf(
        "- **%s** (regression, degree %d): adaptive.gcv %s out-of-sample RMSE (ordinary %.4g vs adaptive %.4g, delta %+.4g).",
        r$name, r$degree, verdict, r$bag$oos_rmse[1], r$bag$oos_rmse[2], d))
}
for (r in bin) {
    d <- r$bag$oos_brier[2] - r$bag$oos_brier[1]
    verdict <- if (d < -1e-6) "IMPROVES" else if (d > 1e-6) "HURTS" else "is NEUTRAL for"
    interp <- c(interp, sprintf(
        "- **%s** (classification, degree %d): adaptive.gcv %s out-of-sample Brier score (ordinary %.4g vs adaptive %.4g, delta %+.4g); OOS accuracy ordinary %.4g vs adaptive %.4g.",
        r$name, r$degree, verdict, r$bag$oos_brier[1], r$bag$oos_brier[2], d,
        r$bag$oos_acc[1], r$bag$oos_acc[2]))
}
interp <- c(interp, "",
    "The adaptive cap is a conservative, default-off forward-pass modification: it charges",
    "each admitted term only the GCV-justified share of the residual reduction, which limits",
    "how much any single term dominates the forward search. Whether that conservatism helps or",
    "hurts generalization is dataset dependent, as the table above shows; on these established",
    "examples the effect is generally small, and ordinary earth remains the default.",
    "",
    "## Companion per-dataset files", "")
for (r in results)
    interp <- c(interp, sprintf("- `doc/%s`", basename(r$outfile)))

summary.lines <- c(summary.lines, interp, "")

summary.file <- file.path(doc.dir, "adaptive_gcv_comparison.md")
writeLines(summary.lines, summary.file)
cat("wrote", summary.file, "\n")
cat("\nDONE. bagging engine used:", engine.used, "\n")
