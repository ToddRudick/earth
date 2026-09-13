#' Predict from a Bucketed Imputation Regression (BIR) model
#'
#' Computes predictions from a fitted \code{\link{fit_bir}} object. For every
#' row of \code{newdata} the per-bucket \emph{unusualness} and \emph{affinity}
#' scores are recomputed exactly as in training (using the stored earth models
#' and residual scales), then the stored lasso path is evaluated at the
#' CV-selected penalty to produce a point prediction. A simple heteroscedastic
#' predictive scale is also available.
#'
#' @details
#' The point prediction is produced with the \code{lars} package (this package
#' deliberately does not use glmnet): \code{predict(object$lasso,
#' newx = affinities, s = object$lasso_s, mode = object$lasso_mode,
#' type = "fit")$fit}, where \code{object$lasso_s} is the fraction chosen at
#' minimum cross-validated MSE during fitting (the lars analogue of glmnet's
#' \code{lambda.min}).
#'
#' The predictive scale (\code{se}) is the square root of an
#' affinity-weighted average of the per-bucket variances of \code{y}: for each
#' row the affinities are normalized to sum to 1 to give weights \code{w_k},
#' and \code{var_hat = sum(w_k * object$bucket_vars)}, \code{se = sqrt(var_hat)}.
#'
#' @param object A fitted \code{"bir"} object from \code{\link{fit_bir}}.
#' @param newdata A numeric matrix or data.frame of predictors. It must contain
#'   all of the columns named in \code{object$colnames}; columns are reordered
#'   (and any extras dropped) to match the training order. Missing columns are
#'   an error.
#' @param type One of \code{"response"} (default; a numeric vector of point
#'   predictions), \code{"affinity"} (the \code{N x n} matrix of affinities), or
#'   \code{"all"} (a list with \code{fit}, \code{se}, and \code{affinities}).
#' @param method One of \code{"affinity_lasso"} (default) or \code{"moe_soft"}.
#'   \code{"affinity_lasso"} is the original behaviour: affinities are fed to
#'   the stored lasso path (\code{predict.lars}). \code{"moe_soft"} is the soft
#'   mixture-of-experts blend (\dQuote{Option A}): using the same affinities as
#'   a gate, each row's weights are \code{w_k = affinity_k / sum_j affinity_j}
#'   (reusing the all-zero-affinity guard, so a fully degenerate row becomes an
#'   unweighted average), and the point prediction is
#'   \code{yhat = sum_k w_k * expert_k(x)} where \code{expert_k} is the
#'   per-bucket degree-2 earth expert (or its constant-mean fallback) fit by
#'   \code{\link{fit_bir}} with \code{experts} enabled. Requesting
#'   \code{"moe_soft"} on a model fit without experts is an error.
#' @param ... Unused; present for S3 compatibility.
#'
#' @return Depends on \code{type}:
#'   \describe{
#'     \item{\code{"response"}}{A numeric vector of length \code{nrow(newdata)}.}
#'     \item{\code{"affinity"}}{A numeric \code{nrow(newdata) x object$n}
#'       matrix of affinities.}
#'     \item{\code{"all"}}{A list with \code{fit} (numeric vector), \code{se}
#'       (non-negative numeric vector, the predictive scale), and
#'       \code{affinities} (the affinity matrix).}
#'   }
#'   Under \code{method = "moe_soft"} the \code{"response"} value is the blended
#'   expert prediction \code{sum_k w_k * expert_k(x)}; \code{"affinity"} still
#'   returns the raw affinity matrix regardless of method; and \code{"all"}
#'   returns \code{fit} = the blended prediction alongside the same \code{se}
#'   and \code{affinities} as the default path.
#'
#'   \strong{Known tension.} Buckets are equal-frequency slices \emph{of}
#'   \code{y}, so within a bucket \code{y} has small variance and each expert
#'   may mostly learn its bucket mean; the \code{moe_soft} blend can therefore
#'   behave close to an affinity-weighted bucket-mean predictor. This is
#'   expected.
#'
#'   Note on the \code{se} fallback: the predictive scale weights the per-bucket
#'   variances by each row's affinities normalized to sum to 1. If a row has all
#'   affinities exactly zero (a fully degenerate row), the normalization would
#'   divide by zero; in that case the weight sum is forced to 1, so \code{se}
#'   for that row collapses to the square root of the plain (unweighted) average
#'   of \code{object$bucket_vars} rather than an affinity-weighted one.
#'
#' @seealso \code{\link{fit_bir}}
#'
#' @examples
#' set.seed(1)
#' N <- 120
#' x <- matrix(rnorm(N * 3), N, 3)
#' colnames(x) <- c("a", "b", "c")
#' y <- x[, 1] - 0.5 * x[, 2] + 0.2 * x[, 3] + rnorm(N, sd = 0.3)
#' fit <- fit_bir(x, y, n = 4)
#' pr <- predict(fit, x, type = "all")
#' str(pr)
#'
#' @importFrom stats predict
#' @export
predict.bir <- function(object, newdata,
                        type = c("response", "affinity", "all"),
                        method = c("affinity_lasso", "moe_soft"), ...) {
  type <- match.arg(type)
  method <- match.arg(method)

  if (method == "moe_soft" && is.null(object$experts)) {
    stop("method = \"moe_soft\" requires per-bucket experts, but this model ",
         "was fit without them. Refit with fit_bir(..., experts = TRUE) ",
         "(or experts = \"earth\") to enable the soft mixture-of-experts.")
  }

  if (is.data.frame(newdata)) {
    newdata <- as.matrix(newdata)
  }
  if (!is.matrix(newdata)) {
    newdata <- as.matrix(newdata)
  }
  if (!is.numeric(newdata)) {
    stop("`newdata` must be numeric.")
  }

  cn <- object$colnames
  nd_names <- colnames(newdata)
  if (is.null(nd_names)) {
    if (ncol(newdata) != object$p) {
      stop("`newdata` has no column names and ", ncol(newdata),
           " columns, but the model expects ", object$p, ".")
    }
    colnames(newdata) <- cn
  } else {
    missing_cols <- setdiff(cn, nd_names)
    if (length(missing_cols) > 0) {
      stop("`newdata` is missing required column(s): ",
           paste(missing_cols, collapse = ", "), ".")
    }
    newdata <- newdata[, cn, drop = FALSE]
  }

  if (any(is.na(newdata))) {
    stop("`newdata` must not contain missing values.")
  }

  ## recompute affinities exactly as in training
  affinities <- .bir_affinities(newdata, object$models, object$scales, cn)

  if (type == "affinity") {
    return(affinities)
  }

  if (method == "moe_soft") {
    ## soft mixture-of-experts: gate the per-bucket experts by normalized
    ## affinities, reusing the all-zero-affinity guard.
    fit <- .bir_moe_predict(object, newdata, affinities)
  } else {
    ## point prediction via lars (predict.lars); NOT glmnet
    fit <- as.numeric(
      predict(object$lasso, newx = affinities, s = object$lasso_s,
              mode = object$lasso_mode, type = "fit")$fit
    )
  }

  if (type == "response") {
    return(fit)
  }

  ## type == "all": also compute the heteroscedastic predictive scale
  row_sums <- rowSums(affinities)
  ## guard against a degenerate all-zero affinity row
  row_sums[row_sums == 0] <- 1
  weights <- affinities / row_sums
  var_hat <- as.numeric(weights %*% object$bucket_vars)
  se <- sqrt(var_hat)

  list(fit = fit, se = se, affinities = affinities)
}

## Internal helper: soft mixture-of-experts point prediction.
## `newdata` columns are already ordered to match object$colnames and
## `affinities` is the N x n affinity matrix from .bir_affinities().
.bir_moe_predict <- function(object, newdata, affinities) {
  N <- nrow(newdata)
  n <- object$n

  ## gate weights: normalize affinities per row. Reuse the all-zero-affinity
  ## guard so a fully degenerate row (all affinities exactly zero) becomes an
  ## unweighted average: replace such a row with uniform affinities before
  ## normalizing, so its weights are 1/n rather than all zero.
  zero_rows <- rowSums(affinities) == 0
  if (any(zero_rows)) {
    affinities[zero_rows, ] <- 1
  }
  row_sums <- rowSums(affinities)
  weights <- affinities / row_sums

  ## evaluate each per-bucket expert on newdata
  expert_pred <- matrix(NA_real_, nrow = N, ncol = n)
  for (k in seq_len(n)) {
    ek <- object$experts[[k]]
    if (inherits(ek, "bir_const_expert")) {
      expert_pred[, k] <- ek$value
    } else {
      expert_pred[, k] <- as.numeric(predict(ek, newdata))
    }
  }

  rowSums(weights * expert_pred)
}
