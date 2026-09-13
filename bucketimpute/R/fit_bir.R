#' Fit a Bucketed Imputation Regression (BIR) model
#'
#' \code{fit_bir} implements Bucketed Imputation Regression exactly as described
#' in the package specification. The response \code{y} is split into \code{n}
#' equal-frequency quantile buckets; within each bucket a linear, degree-2
#' \code{\link[earth]{earth}} (MARS) \emph{imputation} model is fit for every
#' predictor column (reconstructing that column from all of the other
#' predictors); per-bucket standardized absolute reconstruction errors are
#' turned into \emph{affinity} features; and finally a cross-validated lasso of
#' \code{y} on those affinity features is fit.
#'
#' The per-bucket, per-column imputation models are fit with
#' \code{earth(x_j_pred, y_j, degree = 2, linpreds = TRUE, thresh = 1e-6)}:
#' \code{linpreds = TRUE} makes every predictor enter \strong{linearly} (no
#' hinge functions), \code{degree = 2} allows pairwise interaction terms, and
#' \code{thresh = 1e-6} sets earth's forward-pass delta-GRSq stopping threshold
#' very small so the forward pass keeps adding terms until almost no
#' improvement remains. The result is a model of linear main effects plus
#' pairwise interactions with no hinges. (This replaces the earlier degree-1
#' hinge imputation models.) Any of \code{degree}, \code{linpreds}, or
#' \code{thresh} supplied through \code{...} overrides these fixed defaults.
#'
#' @section Algorithm:
#' \enumerate{
#'   \item \strong{Bucket assignment.} Compute \code{n + 1} quantile cut-points
#'     of \code{y} with \code{stats::quantile(y, probs = seq(0, 1,
#'     length.out = n + 1), type = 7)} and store them. Assign every training row
#'     to a bucket \code{k = 1..n} with \code{\link[base]{findInterval}}
#'     (rightmost-closed so the maximum \code{y} lands in bucket \code{n}).
#'     \code{n} must be \code{>= 2}. If the cut-points do not yield \code{n}
#'     \emph{distinct} non-empty buckets (for example when \code{y} has heavy
#'     ties or is near-binary), \code{fit_bir} stops with a clear error naming
#'     the distinct-buckets problem rather than silently merging buckets.
#'   \item \strong{Per-bucket earth imputation models.} For each bucket \code{k}
#'     and each column \code{j}, fit \code{earth(x_k[, -j], x_k[, j],
#'     degree = 2, linpreds = TRUE, thresh = 1e-6, ...)} (linear main effects
#'     plus pairwise interactions, no hinges) and record the residual scale
#'     \code{s_{k,j} = sqrt(mean(residuals^2))}. Each scale is floored at
#'     \code{scale_floor} to avoid divide-by-zero when a bucket's fit is
#'     essentially intercept-only. Column names and order are preserved so that
#'     model \code{(k, j)} always predicts column \code{j} from the other
#'     \code{p - 1} columns.
#'   \item \strong{Training affinities.} For every row \code{i} and bucket
#'     \code{k}, \code{unusualness_{i,k} = sum_j |x_{i,j} -
#'     predict(earth_{k,j}, x_i[, -j])| / s_{k,j}} and
#'     \code{affinity_{i,k} = 1 / (1 + unusualness_{i,k})}, giving an
#'     \code{N x n} affinity matrix.
#'   \item \strong{Final lasso (lars substitution).} The specification calls for
#'     \code{cv.glmnet(..., lambda.min)}. \strong{This implementation
#'     deliberately does not use glmnet.} Instead it fits
#'     \code{lars::lars(affinities, y, type = "lasso")} and selects the penalty
#'     with \code{lars::cv.lars(affinities, y, type = "lasso",
#'     mode = "fraction", plot.it = FALSE)}, choosing the fraction with the
#'     \emph{minimum} cross-validated MSE:
#'     \code{s_opt = cvobj$index[which.min(cvobj$cv)]}. This is the \code{lars}
#'     analogue of glmnet's \code{lambda.min}; net behaviour is a
#'     CV-selected lasso matching the spec's intent, only the fitting engine
#'     changes.
#'   \item \strong{Bucket variances.} For each bucket \code{k},
#'     \code{bucket_vars[k] = var(y[bucket == k])} (single-row buckets give 0).
#'   \item \strong{Optional per-bucket experts (soft mixture-of-experts).} When
#'     \code{experts} is enabled, an additional \emph{expert} regressor of
#'     \code{y} on all of \code{x} is fit within each bucket \code{k}:
#'     \code{earth(x_k, y_k, degree = 2)}. These experts are only used by
#'     \code{\link{predict.bir}} with \code{method = "moe_soft"}; they do not
#'     touch the default affinity-lasso path. See the \code{experts} parameter.
#' }
#'
#' @param x A numeric matrix or data.frame of predictors (all columns numeric,
#'   no missing values). Column names are preserved; if absent they are set to
#'   \code{V1..Vp}.
#' @param y A numeric response vector (continuous), one entry per row of
#'   \code{x}.
#' @param n Number of equal-frequency quantile buckets. Must be \code{>= 2};
#'   defaults to 10.
#' @param min_bucket_rows Minimum number of rows required in each bucket for the
#'   per-bucket, per-column imputation earth fits to be sensible. If \code{NULL}
#'   (the default) a sensible value of \code{max(p + 2, 10)} is used, where
#'   \code{p} is the number of predictor columns; each column model regresses on
#'   the other \code{p - 1} columns, so a bucket should hold at least a handful
#'   more rows than predictors. A bucket thinner than \code{min_bucket_rows}
#'   triggers a warning; a bucket with fewer than 2 rows is a hard error (an
#'   earth model cannot be fit).
#' @param scale_floor Small positive lower bound applied to every residual scale
#'   \code{s_{k,j}} to avoid divide-by-zero in the affinity computation. Default
#'   \code{1e-8}.
#' @param experts Optional switch enabling the soft mixture-of-experts mode
#'   (\dQuote{Option A}). Defaults to \code{NULL} (off), which preserves the
#'   current behaviour exactly. When truthy (\code{TRUE} or the string
#'   \code{"earth"}) \code{fit_bir} additionally fits, for each bucket \code{k},
#'   a degree-2 \code{\link[earth]{earth}} \emph{expert} of \code{y} on all of
#'   \code{x} using that bucket's rows: \code{earth(x_k, y_k, degree = 2)}. The
#'   fitted experts are stored in the \code{experts} field of the returned
#'   object and are consumed by \code{\link{predict.bir}} with
#'   \code{method = "moe_soft"}. If a bucket is too thin to fit a sensible
#'   degree-2 expert (fewer than \code{max(2 * p, min_bucket_rows)} rows) or the
#'   earth fit errors, \code{fit_bir} falls back gracefully to a constant expert
#'   equal to that bucket's \code{mean(y_k)}. The expert \code{degree} is fixed
#'   at 2 and is \emph{not} taken from \code{...} (which is reserved for the
#'   imputation fits); \code{...} is not forwarded to the expert fits
#'   to avoid a \code{degree} conflict. \strong{Known tension:} because buckets
#'   are equal-frequency slices \emph{of} \code{y}, within a bucket \code{y} has
#'   small variance, so an expert may mostly learn the bucket mean; this is
#'   expected.
#' @param ... Extra arguments passed through to \code{\link[earth]{earth}} for
#'   the per-bucket, per-column imputation fits (not the experts). The
#'   imputation fits use fixed defaults \code{degree = 2},
#'   \code{linpreds = TRUE}, and \code{thresh = 1e-6} (linear main effects plus
#'   pairwise interactions, no hinges); passing any of these through \code{...}
#'   overrides the corresponding default, and every other argument is forwarded
#'   to earth() unchanged.
#'
#' @return An S3 object of class \code{"bir"}: a list containing
#'   \describe{
#'     \item{\code{cuts}}{The \code{n + 1} quantile cut-points of \code{y}.}
#'     \item{\code{models}}{A length-\code{n} list; element \code{k} is a
#'       length-\code{p} list of the fitted earth models for bucket \code{k}.}
#'     \item{\code{scales}}{An \code{n x p} matrix of residual scales
#'       \code{s_{k,j}} (floored at \code{scale_floor}).}
#'     \item{\code{lasso}}{The fitted \code{lars} object (the lasso path).}
#'     \item{\code{lasso_s}}{The selected fraction (\code{s_opt}) at minimum CV
#'       MSE.}
#'     \item{\code{lasso_mode}}{The lars prediction mode, \code{"fraction"}.}
#'     \item{\code{bucket_vars}}{Length-\code{n} vector of within-bucket
#'       variances of \code{y}.}
#'     \item{\code{n}}{Number of buckets.}
#'     \item{\code{p}}{Number of predictor columns.}
#'     \item{\code{colnames}}{Predictor column names, in the order the models
#'       expect.}
#'     \item{\code{min_bucket_rows}, \code{scale_floor}}{Stored for
#'       transparency.}
#'     \item{\code{experts}}{\code{NULL} when \code{experts} was not requested
#'       (the default), preserving back-compatibility. When requested, a
#'       length-\code{n} list; element \code{k} is either a fitted degree-2
#'       \code{earth} expert of \code{y} on \code{x} for bucket \code{k}, or a
#'       constant-mean fallback of class \code{"bir_const_expert"} (a list with
#'       a single numeric \code{value = mean(y_k)}) when the bucket was too thin
#'       or the earth fit errored.}
#'     \item{\code{expert_type}}{Present only when experts were requested; a
#'       string recording the expert family and degree, currently
#'       \code{"earth_degree2"}.}
#'   }
#'
#' @note \strong{Reproducibility.} The lasso penalty is selected by
#'   \code{lars::cv.lars}, which draws its cross-validation folds from the
#'   global random number generator. \code{fit_bir} deliberately does not set an
#'   internal seed (so it will not silently override a caller's RNG state), so
#'   the selected \code{lasso_s} — and therefore the predictions — are only
#'   reproducible if the caller sets a seed before fitting. For repeatable
#'   results call \code{set.seed()} immediately before \code{fit_bir}; two fits
#'   run under the same seed give identical \code{lasso_s} and identical
#'   predictions.
#'
#' @seealso \code{\link{predict.bir}}
#'
#' @examples
#' set.seed(1)
#' N <- 120
#' x <- matrix(rnorm(N * 3), N, 3)
#' colnames(x) <- c("a", "b", "c")
#' y <- x[, 1] - 0.5 * x[, 2] + 0.2 * x[, 3] + rnorm(N, sd = 0.3)
#' fit <- fit_bir(x, y, n = 4)
#' print(fit)
#' head(predict(fit, x))
#'
#' @importFrom stats quantile var
#' @importFrom earth earth
#' @importFrom lars lars cv.lars
#' @importFrom utils modifyList
#' @export
fit_bir <- function(x, y, n = 10, min_bucket_rows = NULL,
                    scale_floor = 1e-8, experts = NULL, ...) {
  ## ---- input validation -------------------------------------------------
  if (!is.numeric(n) || length(n) != 1L || n < 2) {
    stop("`n` (number of buckets) must be a single number >= 2.")
  }
  n <- as.integer(n)

  if (is.data.frame(x)) {
    x <- as.matrix(x)
  }
  if (!is.matrix(x)) {
    x <- as.matrix(x)
  }
  if (!is.numeric(x)) {
    stop("`x` must be numeric (all predictor columns numeric, no factors).")
  }
  if (any(is.na(x))) {
    stop("`x` must not contain missing values.")
  }

  y <- as.numeric(y)
  if (any(is.na(y))) {
    stop("`y` must not contain missing values.")
  }
  if (length(y) != nrow(x)) {
    stop("length(y) must equal nrow(x).")
  }

  N <- nrow(x)
  p <- ncol(x)
  if (p < 2) {
    stop("`x` must have at least 2 predictor columns (each column is ",
         "reconstructed from the others).")
  }

  cn <- colnames(x)
  if (is.null(cn)) {
    cn <- paste0("V", seq_len(p))
    colnames(x) <- cn
  }

  if (is.null(min_bucket_rows)) {
    min_bucket_rows <- max(p + 2L, 10L)
  }
  if (!is.numeric(min_bucket_rows) || length(min_bucket_rows) != 1L ||
      min_bucket_rows < 2) {
    stop("`min_bucket_rows` must be a single number >= 2.")
  }

  if (!is.numeric(scale_floor) || length(scale_floor) != 1L ||
      scale_floor <= 0) {
    stop("`scale_floor` must be a single positive number.")
  }

  ## `experts`: NULL/FALSE => off (default, current behaviour); TRUE or
  ## "earth" => fit per-bucket degree-2 earth experts of y on x.
  fit_experts <- FALSE
  if (!is.null(experts)) {
    if (isTRUE(experts) ||
        (is.character(experts) && length(experts) == 1L &&
         identical(experts, "earth"))) {
      fit_experts <- TRUE
    } else if (isFALSE(experts)) {
      fit_experts <- FALSE
    } else {
      stop("`experts` must be NULL/FALSE (off), or TRUE / \"earth\" (on).")
    }
  }

  ## ---- 1. bucket assignment ---------------------------------------------
  cuts <- stats::quantile(y, probs = seq(0, 1, length.out = n + 1L),
                          type = 7)
  cuts <- as.numeric(cuts)

  ## rightmost-closed assignment into buckets 1..n
  bucket <- findInterval(y, cuts, rightmost.closed = TRUE, all.inside = TRUE)

  distinct_buckets <- sort(unique(bucket))
  if (length(distinct_buckets) != n ||
      !all(distinct_buckets == seq_len(n))) {
    stop("Equal-frequency quantile bucketing of `y` did not yield ", n,
         " distinct non-empty buckets (got ", length(distinct_buckets),
         "). This happens when `y` has heavy ties or is near-binary. ",
         "Reduce `n` or supply a response with more distinct values; ",
         "buckets are not silently merged.")
  }

  bucket_counts <- tabulate(bucket, nbins = n)
  if (any(bucket_counts < 2L)) {
    thin <- which(bucket_counts < 2L)
    stop("Bucket(s) ", paste(thin, collapse = ", "),
         " have fewer than 2 rows, which is too thin to fit an earth model. ",
         "Reduce `n`.")
  }
  if (any(bucket_counts < min_bucket_rows)) {
    thin <- which(bucket_counts < min_bucket_rows)
    warning("Bucket(s) ", paste(thin, collapse = ", "),
            " have fewer than min_bucket_rows (", min_bucket_rows,
            ") rows; per-bucket earth fits may be unreliable.")
  }

  ## ---- 2. per-bucket earth models ---------------------------------------
  ## Capture `...` once so it can be merged with the fixed imputation-model
  ## settings (degree = 2, linpreds = TRUE, thresh = 1e-6). Caller-supplied
  ## arguments in `...` win over these defaults; see .bir_impute_earth_args().
  imp_dots <- list(...)
  models <- vector("list", n)
  scales <- matrix(NA_real_, nrow = n, ncol = p,
                   dimnames = list(NULL, cn))

  for (k in seq_len(n)) {
    rows_k <- which(bucket == k)
    x_k <- x[rows_k, , drop = FALSE]
    models_k <- vector("list", p)
    for (j in seq_len(p)) {
      xj_pred <- x_k[, -j, drop = FALSE]
      yj <- x_k[, j]
      fit_kj <- do.call(
        earth::earth,
        .bir_impute_earth_args(x = xj_pred, y = yj, dots = imp_dots)
      )
      resid_kj <- as.numeric(yj) - as.numeric(predict(fit_kj, xj_pred))
      s_kj <- sqrt(mean(resid_kj^2))
      scales[k, j] <- max(s_kj, scale_floor)
      models_k[[j]] <- fit_kj
    }
    models[[k]] <- models_k
  }

  ## ---- 3. training affinities -------------------------------------------
  affinities <- .bir_affinities(x, models, scales, cn)

  ## ---- 4. final lasso via lars (NOT glmnet) -----------------------------
  lasso <- lars::lars(affinities, y, type = "lasso")
  cvobj <- lars::cv.lars(affinities, y, type = "lasso", mode = "fraction",
                         plot.it = FALSE)
  s_opt <- cvobj$index[which.min(cvobj$cv)]

  ## ---- 5. bucket variances ----------------------------------------------
  bucket_vars <- vapply(seq_len(n), function(k) {
    yk <- y[bucket == k]
    if (length(yk) < 2L) 0 else stats::var(yk)
  }, numeric(1))

  ## ---- 5b. optional per-bucket experts (soft mixture-of-experts) --------
  experts_list <- NULL
  expert_type <- NULL
  if (fit_experts) {
    ## a bucket needs enough rows for a degree-2 earth of y on p predictors;
    ## otherwise fall back to a constant bucket-mean expert.
    expert_min_rows <- max(2L * p, as.integer(min_bucket_rows))
    experts_list <- vector("list", n)
    for (k in seq_len(n)) {
      rows_k <- which(bucket == k)
      x_k <- x[rows_k, , drop = FALSE]
      y_k <- y[rows_k]
      const_expert <- structure(list(value = mean(y_k)),
                                 class = "bir_const_expert")
      if (length(rows_k) < expert_min_rows) {
        experts_list[[k]] <- const_expert
        next
      }
      experts_list[[k]] <- tryCatch(
        earth::earth(x = x_k, y = y_k, degree = 2),
        error = function(e) const_expert
      )
    }
    expert_type <- "earth_degree2"
  }

  ## ---- 6. return value --------------------------------------------------
  structure(
    list(
      cuts = cuts,
      models = models,
      scales = scales,
      lasso = lasso,
      lasso_s = s_opt,
      lasso_mode = "fraction",
      bucket_vars = bucket_vars,
      n = n,
      p = p,
      colnames = cn,
      min_bucket_rows = min_bucket_rows,
      scale_floor = scale_floor,
      experts = experts_list,
      expert_type = expert_type
    ),
    class = "bir"
  )
}

## Internal helper: assemble the argument list for a per-bucket, per-column
## imputation earth() fit. The imputation models are fixed to be linear
## (linpreds = TRUE, so predictors enter linearly with no hinge functions),
## degree = 2 (allowing pairwise interaction terms), and thresh = 1e-6 (a very
## small forward-pass delta-GRSq stopping threshold, so the forward pass keeps
## adding terms until almost no improvement remains). These are the intended
## imputation-model settings for this package. Any of `degree`, `linpreds`, or
## `thresh` supplied by the caller through `...` (`dots`) overrides the
## corresponding fixed default; all other `...` arguments are forwarded to
## earth() unchanged, preserving the passthrough.
.bir_impute_earth_args <- function(x, y, dots) {
  defaults <- list(degree = 2, linpreds = TRUE, thresh = 1e-6)
  ## caller-supplied dots win over the fixed imputation defaults
  merged <- utils::modifyList(defaults, dots)
  c(list(x = x, y = y), merged)
}

## Internal helper: compute the N x n affinity matrix for a matrix `x`
## whose columns are already ordered to match `cn`.
.bir_affinities <- function(x, models, scales, cn) {
  N <- nrow(x)
  n <- length(models)
  p <- ncol(x)
  aff <- matrix(NA_real_, nrow = N, ncol = n)
  for (k in seq_len(n)) {
    models_k <- models[[k]]
    unusual_k <- numeric(N)
    for (j in seq_len(p)) {
      xj_pred <- x[, -j, drop = FALSE]
      pred_j <- as.numeric(predict(models_k[[j]], xj_pred))
      unusual_k <- unusual_k + abs(x[, j] - pred_j) / scales[k, j]
    }
    aff[, k] <- 1 / (1 + unusual_k)
  }
  aff
}
