# adaptive.gcv.comparison.R
#
# STAGE-1 empirical study of the adaptive GCV effect cap (adaptive.gcv=TRUE)
# versus ordinary earth (adaptive.gcv=FALSE, the default).
#
# The Stage-1 cap is GCV / PER-TERM-COMPLEXITY adaptive (FEAT-004): it is no
# longer a fixed fraction of total variance.  The budget for the incremental
# delta-RSS a newly admitted term may realise is
#
#   DeltaRssMax = BreakEven + slackFactor * (RssDelta - BreakEven)
#   slackFactor = (1 - clamp(Cost1))^gamma,   gamma = 1/effect.cap - 1
#   Cost1       = (nOldUsedTerms + deltaTerms + Penalty*(nKnotsOld + deltaKnots))/n
#
# with an EXPLICIT per-term knot charge: deltaKnots = 0 for a linear/linpreds
# term, deltaKnots = 1 for a hinge term.  effect.cap>=1 => slackFactor==1 =>
# stock earth.
#
# CORE STAGE-1 QUESTION this study answers empirically, with genuine
# out-of-sample (OOS) results: under the per-term-complexity-aware cap, do HINGE
# terms and LINEAR terms diverge as predicted?  I.e. does a DOMINANT predictor
# entered as a cheap LINEAR term (0 knots) receive a HIGHER justified effect
# budget / larger CapScale (is shrunk LESS) than the SAME signal expressed as a
# HINGE (1 knot)?  For each dataset we contrast the dominant predictor entered
# as a HINGE vs FORCED LINEAR (via the `linpreds` argument) and report the
# retained effect / CapScale (from trace>=6) and the OOS score of each.
#
# OUT OF SCOPE (this is Stage 2): generating BOTH a hinged and unhinged version
# of each candidate term inside the algorithm.  Stage 1 only MEASURES the
# divergence using the existing linpreds mechanism to force the linear form.
#
# OOS ENGINE: earth's BUILT-IN cross-validation (nfold=5, ncross=3, fixed seed
# 2024) is the PRIMARY and only-critical-path OOS engine (fast, independent of
# caret).  The caret::bagEarth block is OPTIONAL, guarded, and OFF the critical
# path: it runs only if options(adaptive.gcv.run.caret=TRUE) is set AND caret is
# installed.  Put a hard `timeout` on every run of this script; it is designed
# to finish well under ~15 minutes without caret.
#
# Datasets (shipped with earth or base R, established MARS / earth examples):
#   - ozone1    : canonical MARS / earth-vignette regression example, degree 2
#   - trees     : base R regression, degree 1
#   - mtcars    : base R regression, degree 1
#   - etitanic  : earth example, BINARY survived -> classification, degree 2
#
# Output: regenerates doc/adaptive_gcv_comparison.md and the per-dataset
# doc/adaptive_gcv_<name>.md files.  Reproducible via fixed seeds.

suppressWarnings(suppressMessages(library(earth)))
options(warn = 1)

SEED   <- 2024
NFOLD  <- 5
NCROSS <- 3
CAPS   <- c(0.5, 0.9)          # effect.cap values studied (both < 1)

run.caret <- isTRUE(getOption("adaptive.gcv.run.caret", FALSE)) &&
    requireNamespace("caret", quietly = TRUE) &&
    exists("bagEarth", where = asNamespace("caret"))

## ---------------------------------------------------------------------------
## Output location (robust whether run from repo root or inst/slowtests)
## ---------------------------------------------------------------------------
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
cat("optional caret bagEarth path enabled:", run.caret, "\n\n")

## ---------------------------------------------------------------------------
## Markdown helper
## ---------------------------------------------------------------------------
md.table <- function(df) {
    fmt <- function(x) {
        if (is.numeric(x)) ifelse(is.na(x), "NA", formatC(x, digits = 5, format = "g"))
        else as.character(x)
    }
    cols   <- names(df)
    header <- paste0("| ", paste(cols, collapse = " | "), " |")
    sep    <- paste0("| ", paste(rep("---", length(cols)), collapse = " | "), " |")
    rows <- apply(df, 1, function(r)
        paste0("| ", paste(vapply(seq_along(r), function(i) fmt(r[[i]]), ""),
                           collapse = " | "), " |"))
    paste(c(header, sep, rows), collapse = "\n")
}

## ---------------------------------------------------------------------------
## trace>=6 cap diagnostics parser (mirrors tests/test.adaptive.gcv.R).
## Returns a data.frame of the per-term "effectcap CAP" lines (only emitted for
## SATURATED terms) with the fields the study reports.
## ---------------------------------------------------------------------------
parse.field <- function(line, key) {
    m <- regmatches(line, regexpr(paste0(key, "\\s+[-+0-9.eE]+"), line))
    if (length(m) == 0) return(NA_real_)
    as.numeric(sub(paste0("^", key, "\\s+"), "", m))
}
# pmethod="none" so the returned $dirs is the FULL forward-pass term set, which
# lets us map each trace iTerm to its predictor(s) exactly (pruning does not
# affect the forward pass or the cap diagnostics).
cap.diag <- function(form, data, degree, effect.cap, linpreds = NULL,
                     is.binary = FALSE) {
    args <- list(form, data = data, degree = degree, adaptive.gcv = TRUE,
                 effect.cap = effect.cap, trace = 6, pmethod = "none")
    if (!is.null(linpreds)) args$linpreds <- linpreds
    if (is.binary)          args$glm      <- list(family = binomial)
    out <- capture.output(fit <- do.call(earth, args))
    cap.lines <- grep("effectcap CAP:", out, value = TRUE)
    cap.lines <- sub(".*effectcap CAP:", "effectcap CAP:", cap.lines)
    df <- if (length(cap.lines) == 0) data.frame() else data.frame(
        iTerm      = as.integer(sapply(cap.lines, parse.field, key = "iTerm")),
        dRSSols    = sapply(cap.lines, parse.field, key = "dRSS\\(ols\\)"),
        dRSSmax    = sapply(cap.lines, parse.field, key = "dRSSmax"),
        scale      = sapply(cap.lines, parse.field, key = "scale"),
        deltaKnots = as.integer(sapply(cap.lines, parse.field, key = "deltaKnots")),
        Cost1      = sapply(cap.lines, parse.field, key = "Cost1"),
        slackFactor= sapply(cap.lines, parse.field, key = "slackFactor"),
        row.names  = NULL)
    list(fit = fit, cap = df)
}

# Does the term-pair admitted at trace iTerm k use predictor `pred`?  The
# forward pass numbers terms with the intercept as term 0, so internal term k
# occupies rows k+1 (and k+2 for a hinge pair) of the full (pmethod="none")
# dirs matrix.
term.uses.pred <- function(fit, iTerm, pred) {
    dirs <- fit$dirs
    if (!(pred %in% colnames(dirs))) return(FALSE)
    rows <- intersect(c(iTerm + 1L, iTerm + 2L), seq_len(nrow(dirs)))
    any(dirs[rows, pred] != 0)
}

# earliest SATURATED cap row whose term uses the dominant predictor
dominant.cap.row <- function(cd, pred) {
    if (nrow(cd$cap) == 0) return(NULL)
    ord <- cd$cap[order(cd$cap$iTerm), , drop = FALSE]
    for (i in seq_len(nrow(ord)))
        if (term.uses.pred(cd$fit, ord$iTerm[i], pred))
            return(ord[i, ])
    NULL
}

## ---------------------------------------------------------------------------
## earth built-in CV: return in-sample and cross-validated metrics.
## ---------------------------------------------------------------------------
cv.fit <- function(form, data, degree, adaptive, effect.cap = 0.9,
                   linpreds = NULL, is.binary = FALSE) {
    set.seed(SEED)
    args <- list(form, data = data, degree = degree,
                 nfold = NFOLD, ncross = NCROSS,
                 adaptive.gcv = adaptive, effect.cap = effect.cap)
    if (!is.null(linpreds)) args$linpreds <- linpreds
    if (is.binary)          args$glm      <- list(family = binomial)
    m <- do.call(earth, args)
    cv.rsq <- if (!is.null(m$cv.rsq.tab))
        m$cv.rsq.tab[nrow(m$cv.rsq.tab), "mean"] else NA_real_
    cv.class <- if (!is.null(m$cv.class.rate.tab))
        m$cv.class.rate.tab[nrow(m$cv.class.rate.tab), "mean"] else NA_real_
    list(model = m, insample.rsq = m$rsq, cv.rsq = cv.rsq, cv.class = cv.class,
         nterms = length(m$selected.terms), gcv = m$gcv)
}

## ---------------------------------------------------------------------------
## Retained variance of the fitted contribution attributable to a predictor.
## ---------------------------------------------------------------------------
retained.effect <- function(fit, pred) {
    keep <- fit$selected.terms
    dirs <- fit$dirs[keep, , drop = FALSE]
    if (!(pred %in% colnames(dirs))) return(NA_real_)
    uses <- dirs[, pred] != 0
    if (!any(uses)) return(0)
    co      <- fit$coefficients[, 1]
    contrib <- fit$bx[, uses, drop = FALSE] %*% co[uses]
    var(as.numeric(contrib))
}

## ---------------------------------------------------------------------------
## Optional caret::bagEarth OOS check (OFF the critical path).
## ---------------------------------------------------------------------------
caret.oos <- function(form, data, degree, is.binary) {
    if (!run.caret) return(NULL)
    mf <- model.frame(form, data)
    y  <- model.response(mf)
    x  <- model.matrix(form, mf)
    x  <- x[, colnames(x) != "(Intercept)", drop = FALSE]
    if (is.binary) y <- as.integer(y) - min(as.integer(y))
    set.seed(SEED)
    n  <- nrow(x); tr <- sample.int(n, round(0.7 * n))
    fit.bag <- function(adaptive) {
        set.seed(SEED)
        caret::bagEarth(x = as.data.frame(x[tr, , drop = FALSE]), y = y[tr],
                        B = 20, degree = degree, adaptive.gcv = adaptive)
    }
    pr <- function(m) as.numeric(predict(m, as.data.frame(x[-tr, , drop = FALSE])))
    rmse <- function(a, p) sqrt(mean((a - p)^2))
    list(off = rmse(y[-tr], pr(fit.bag(FALSE))),
         on  = rmse(y[-tr], pr(fit.bag(TRUE))))
}

## ---------------------------------------------------------------------------
## Run one dataset end-to-end and write its markdown file.
## ---------------------------------------------------------------------------
run.dataset <- function(name, form, data, degree, dominant,
                        is.binary = FALSE) {
    cat("==== dataset:", name, "(degree =", degree,
        ", dominant =", dominant, ") ====\n")

    ## --- ordinary vs adaptive at each effect.cap, earth built-in CV ---
    off <- cv.fit(form, data, degree, adaptive = FALSE, is.binary = is.binary)
    on.caps <- lapply(CAPS, function(ec)
        cv.fit(form, data, degree, adaptive = TRUE, effect.cap = ec,
               is.binary = is.binary))
    names(on.caps) <- paste0("cap", CAPS)

    cv.tab <- data.frame(
        setting      = c("ordinary (OFF)",
                         sprintf("adaptive cap=%.2g", CAPS)),
        insample_rsq = c(off$insample.rsq,
                         sapply(on.caps, `[[`, "insample.rsq")),
        cv_rsq       = c(off$cv.rsq,   sapply(on.caps, `[[`, "cv.rsq")),
        cv_classrate = c(off$cv.class, sapply(on.caps, `[[`, "cv.class")),
        nterms       = c(off$nterms,   sapply(on.caps, `[[`, "nterms")),
        row.names    = NULL)

    ## --- HINGE vs FORCED-LINEAR contrast for the dominant predictor ---
    # single (non-CV) fits at effect.cap=0.5 to read the cap diagnostics, and
    # CV fits at effect.cap=0.5 for the OOS score of each representation.
    ec.contrast <- 0.5
    d.hinge <- cap.diag(form, data, degree, ec.contrast, linpreds = NULL,
                        is.binary = is.binary)
    d.lin   <- cap.diag(form, data, degree, ec.contrast, linpreds = dominant,
                        is.binary = is.binary)
    cv.hinge <- cv.fit(form, data, degree, adaptive = TRUE,
                       effect.cap = ec.contrast, linpreds = NULL,
                       is.binary = is.binary)
    cv.lin   <- cv.fit(form, data, degree, adaptive = TRUE,
                       effect.cap = ec.contrast, linpreds = dominant,
                       is.binary = is.binary)

    # the dominant predictor's earliest SATURATED cap row in each fit
    ch <- dominant.cap.row(d.hinge, dominant)
    cl <- dominant.cap.row(d.lin,   dominant)

    contrast.tab <- data.frame(
        representation = c("hinge", "forced linear"),
        deltaKnots     = c(if (!is.null(ch)) ch$deltaKnots else NA,
                           if (!is.null(cl)) cl$deltaKnots else NA),
        Cost1          = c(if (!is.null(ch)) ch$Cost1 else NA,
                           if (!is.null(cl)) cl$Cost1 else NA),
        slackFactor    = c(if (!is.null(ch)) ch$slackFactor else NA,
                           if (!is.null(cl)) cl$slackFactor else NA),
        dRSSmax_budget = c(if (!is.null(ch)) ch$dRSSmax else NA,
                           if (!is.null(cl)) cl$dRSSmax else NA),
        CapScale       = c(if (!is.null(ch)) ch$scale else NA,
                           if (!is.null(cl)) cl$scale else NA),
        retained_var   = c(retained.effect(cv.hinge$model, dominant),
                           retained.effect(cv.lin$model,   dominant)),
        cv_rsq         = c(cv.hinge$cv.rsq, cv.lin$cv.rsq),
        row.names      = NULL)

    # Verdict.  The Stage-1 claim is that the cheaper LINEAR form is shrunk LESS
    # than the HINGE form: it carries a LOWER per-term complexity cost (Cost1),
    # a LARGER slackFactor, and a LARGER CapScale (fraction of its own OLS effect
    # that is retained).  CapScale/slackFactor are the scale-relative shrinkage
    # signals and are the right comparison; the absolute budget dRSSmax mixes the
    # complexity charge with the raw OLS-effect ceiling, which differs between
    # the two differently-shaped fits, so it is reported but not asserted on.
    got.h <- !is.null(ch); got.l <- !is.null(cl)
    linear.favored <- got.h && got.l &&
        cl$deltaKnots < ch$deltaKnots &&
        cl$slackFactor > ch$slackFactor &&
        cl$scale > ch$scale
    verdict <- if (!got.h || !got.l)
        "the dominant predictor's term was NOT saturated in one representation (its cap did not bind at effect.cap=0.5); no divergence is expected there"
    else if (cl$deltaKnots == ch$deltaKnots)
        sprintf(paste0("both representations charged the same per-term knot cost (deltaKnots=%d); ",
                       "the forced-linear term did not reduce the knot charge here (the predictor's ",
                       "earliest saturated term was already linear), so no divergence is expected"),
                ch$deltaKnots)
    else if (linear.favored)
        "YES - the cheaper LINEAR form (0 knots) carries a lower Cost1, a larger slackFactor and a larger CapScale (shrunk less) than the HINGE form (1 knot), exactly as the per-term knot charge predicts"
    else
        "NO - despite the lower knot charge the linear form was not shrunk strictly less here"

    ## --- optional caret OOS (off critical path) ---
    caret.res <- caret.oos(form, data, degree, is.binary)

    ## --- write per-dataset markdown ---
    md <- c(
        sprintf("# Adaptive GCV effect cap (Stage 1): %s", name),
        "",
        sprintf("- Response formula: `%s`", deparse(form)),
        sprintf("- Rows: %d", nrow(data)),
        sprintf("- earth `degree` = %d", degree),
        sprintf("- Task type: %s",
                if (is.binary) "binary classification" else "regression"),
        sprintf("- Dominant predictor studied (hinge vs forced linear): `%s`", dominant),
        sprintf("- OOS engine: earth built-in cross-validation, nfold = %d, ncross = %d, seed %d",
                NFOLD, NCROSS, SEED),
        "",
        "## Stage-1 cap semantics",
        "",
        "The cap is GCV / per-term-complexity adaptive (not a fixed fraction of",
        "variance).  A term's realised delta-RSS is limited to",
        "`DeltaRssMax = BreakEven + slackFactor*(RssDelta - BreakEven)` where",
        "`slackFactor = (1 - clamp(Cost1))^(1/effect.cap - 1)` and `Cost1` uses an",
        "EXPLICIT per-term knot charge (0 knots for a linear/linpreds term, 1 knot",
        "for a hinge term).  `effect.cap >= 1` reproduces stock earth exactly.",
        "",
        "## Ordinary vs adaptive: in-sample and cross-validated fit",
        "",
        md.table(cv.tab),
        "",
        sprintf("## Hinge vs forced-linear contrast for `%s` (effect.cap = %.2g)",
                dominant, ec.contrast),
        "",
        "For the dominant predictor we compare its natural HINGE representation",
        "against the SAME predictor FORCED LINEAR via `linpreds`.  `deltaKnots`,",
        "`Cost1`, `slackFactor`, the effect budget `dRSSmax`, and the applied",
        "`CapScale` are read from the `trace >= 6` cap diagnostics for the",
        "predictor's earliest saturated term; `retained_var` is the variance of",
        "the fitted contribution attributable to the predictor; `cv_rsq` is the",
        "earth built-in CV RSq of the whole model under each representation.",
        "",
        md.table(contrast.tab),
        "",
        sprintf("**Divergence verdict: %s.**", verdict),
        "")
    if (!is.null(caret.res))
        md <- c(md,
            "## Optional caret::bagEarth holdout RMSE (off the critical path)",
            "",
            md.table(data.frame(setting = c("ordinary", "adaptive"),
                                oos_rmse = c(caret.res$off, caret.res$on))),
            "")
    outfile <- file.path(doc.dir, sprintf("adaptive_gcv_%s.md", name))
    writeLines(md, outfile)
    cat("wrote", outfile, "\n")
    cat("  ", verdict, "\n\n")

    list(name = name, degree = degree, is.binary = is.binary,
         dominant = dominant, cv.tab = cv.tab, contrast.tab = contrast.tab,
         linear.favored = linear.favored, verdict = verdict,
         off = off, on.caps = on.caps, outfile = outfile)
}

## ---------------------------------------------------------------------------
## Datasets
## ---------------------------------------------------------------------------
results <- list()

data(ozone1, package = "earth")
results[["ozone1"]] <- run.dataset(
    "ozone1", O3 ~ ., ozone1, degree = 2, dominant = "temp")

data(trees)
results[["trees"]] <- run.dataset(
    "trees", Volume ~ ., trees, degree = 1, dominant = "Girth")

results[["mtcars"]] <- run.dataset(
    "mtcars", mpg ~ ., mtcars, degree = 1, dominant = "disp")

data(etitanic, package = "earth")
# age is the dominant CONTINUOUS predictor (sex/pclass are factors, for which a
# hinge is meaningless); forcing a continuous predictor linear is the meaningful
# Stage-1 contrast.
results[["etitanic"]] <- run.dataset(
    "etitanic", survived ~ ., etitanic, degree = 2, dominant = "age",
    is.binary = TRUE)

## ---------------------------------------------------------------------------
## Combined summary report
## ---------------------------------------------------------------------------
lines <- c(
    "# Adaptive GCV Effect Cap (Stage 1): in-sample vs out-of-sample comparison",
    "",
    "This report compares ordinary earth (`adaptive.gcv = FALSE`, the default)",
    "against the experimental **Stage-1** adaptive GCV effect cap",
    "(`adaptive.gcv = TRUE`) on several established datasets, using genuine",
    "out-of-sample (OOS) scores from earth's built-in cross-validation.",
    "",
    "## What changed in Stage 1",
    "",
    "The effect cap is now **GCV / per-term-complexity adaptive**, not a fixed",
    "fraction of the total variance.  When a candidate term is admitted, the",
    "incremental delta-RSS it is allowed to realise is",
    "",
    "```",
    "DeltaRssMax = BreakEven + slackFactor * (RssDelta - BreakEven)",
    "slackFactor = (1 - clamp(Cost1))^gamma,     gamma = 1/effect.cap - 1",
    "Cost1       = (nOldUsedTerms + deltaTerms + Penalty*(nKnotsOld + deltaKnots)) / n",
    "```",
    "",
    "where `RssDelta` is the unconstrained OLS effect (ceiling), `BreakEven` is",
    "the GCV break-even reduction (floor), and the key Stage-1 change is the",
    "**explicit per-term knot charge**: `deltaKnots = 0` for a linear/`linpreds`",
    "term and `deltaKnots = 1` for a hinge term.  Because `Cost1` rises with",
    "`deltaKnots` and `slackFactor` decreases with `Cost1`, a HINGE term gets a",
    "SMALLER budget than a LINEAR term carrying the same OLS effect.",
    "`effect.cap >= 1` forces `slackFactor == 1` and reproduces stock earth",
    "byte-for-byte.",
    "",
    "## The Stage-1 question",
    "",
    "> Under the per-term-complexity-aware cap, do HINGE and LINEAR terms diverge",
    "> as predicted?  Does a dominant predictor entered as a cheap LINEAR term (0",
    "> knots) get a HIGHER justified effect budget / larger CapScale (shrunk",
    "> LESS) than the same signal expressed as a HINGE (1 knot)?",
    "",
    "For each dataset the dominant predictor is fit once as a HINGE (default) and",
    "once FORCED LINEAR via `linpreds`, and we report the retained effect /",
    "CapScale (from `trace >= 6`) and the OOS CV RSq of each representation.",
    "This uses the EXISTING `linpreds` mechanism only to MEASURE the divergence;",
    "generating both a hinged and unhinged version of every candidate inside the",
    "algorithm is Stage 2 and is deliberately out of scope here.",
    "",
    "## Methodology",
    "",
    sprintf("- **OOS engine (primary, only critical path):** earth built-in CV, `nfold = %d, ncross = %d`, `set.seed(%d)`.",
            NFOLD, NCROSS, SEED),
    sprintf("- **effect.cap values studied:** %s (both < 1; the hinge-vs-linear contrast uses effect.cap = 0.5).",
            paste(CAPS, collapse = ", ")),
    "- **Optional:** a `caret::bagEarth` 70/30 holdout is included only when",
    "  `options(adaptive.gcv.run.caret = TRUE)` is set and caret is installed;",
    "  it is OFF the critical path (caret bagEarth is slow).",
    "",
    "## Datasets",
    "",
    "| dataset | task | degree | rows | dominant predictor | notes |",
    "| --- | --- | --- | --- | --- | --- |",
    "| ozone1 | regression | 2 | 330 | temp | canonical MARS / earth-vignette example |",
    "| trees | regression | 1 | 31 | Girth | base R |",
    "| mtcars | regression | 1 | 32 | disp | base R |",
    "| etitanic | classification | 2 | 1046 | age | earth example, binary `survived` |",
    "",
    "## Cross-validated fit: ordinary vs adaptive",
    "",
    "| dataset | setting | in-sample RSq | CV RSq | CV class-rate | nterms |",
    "| --- | --- | --- | --- | --- | --- |")
for (r in results) {
    ct <- r$cv.tab
    for (i in seq_len(nrow(ct)))
        lines <- c(lines, sprintf("| %s | %s | %.4g | %.4g | %s | %d |",
            r$name, ct$setting[i], ct$insample_rsq[i], ct$cv_rsq[i],
            if (is.na(ct$cv_classrate[i])) "NA"
            else formatC(ct$cv_classrate[i], digits = 4, format = "g"),
            as.integer(ct$nterms[i])))
}

lines <- c(lines, "",
    "## Hinge vs forced-linear divergence (dominant predictor, effect.cap = 0.5)",
    "",
    "| dataset | predictor | representation | deltaKnots | slackFactor | dRSSmax budget | CapScale | retained var | CV RSq |",
    "| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
for (r in results) {
    cc <- r$contrast.tab
    reps <- c("hinge", "forced linear")
    for (i in 1:2)
        lines <- c(lines, sprintf("| %s | %s | %s | %s | %s | %s | %s | %s | %s |",
            r$name, r$dominant, reps[i],
            if (is.na(cc$deltaKnots[i])) "NA" else as.character(cc$deltaKnots[i]),
            formatC(cc$slackFactor[i], digits = 5, format = "g"),
            formatC(cc$dRSSmax_budget[i], digits = 5, format = "g"),
            formatC(cc$CapScale[i], digits = 5, format = "g"),
            formatC(cc$retained_var[i], digits = 5, format = "g"),
            formatC(cc$cv_rsq[i], digits = 5, format = "g")))
}

lines <- c(lines, "", "## Divergence verdict per dataset", "")
for (r in results)
    lines <- c(lines, sprintf("- **%s** (dominant `%s`): %s.",
                              r$name, r$dominant, r$verdict))

n.fav <- sum(vapply(results, function(r) isTRUE(r$linear.favored), logical(1)))
lines <- c(lines, "", "## Interpretation", "",
    sprintf(paste0("Across the %d datasets, the cheaper LINEAR representation of the ",
                   "dominant predictor received a strictly larger effect budget and ",
                   "larger CapScale (was shrunk less) than the HINGE representation in ",
                   "%d of them."),
            length(results), n.fav),
    "This is the divergence the explicit per-term knot charge (linear = 0 knots,",
    "hinge = 1 knot) is designed to produce: a hinge is charged for its extra knot",
    "through a higher `Cost1`, which lowers `slackFactor` and therefore the effect",
    "budget, so the same signal is regularised more heavily when expressed as a",
    "hinge than when forced linear.  Where the dominant predictor's term is not",
    "saturated in a given representation (the cap does not bind), no divergence is",
    "expected and the table reports that directly.",
    "",
    "Whether the extra regularisation helps or hurts OOS generalisation is",
    "dataset dependent (see the CV RSq columns); ordinary earth remains the",
    "default.  The `effect.cap` argument tunes the strength: `effect.cap >= 1`",
    "recovers stock earth, smaller values push saturated terms further toward",
    "their GCV break-even effect, with hinges pushed hardest.",
    "",
    "## Companion per-dataset files", "")
for (r in results)
    lines <- c(lines, sprintf("- `doc/%s`", basename(r$outfile)))
lines <- c(lines, "")

summary.file <- file.path(doc.dir, "adaptive_gcv_comparison.md")
writeLines(lines, summary.file)
cat("wrote", summary.file, "\n")
cat("\nDONE. linear-favored in", n.fav, "of", length(results), "datasets.\n")
