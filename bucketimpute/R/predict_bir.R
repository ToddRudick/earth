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
                        type = c("response", "affinity", "all"), ...) {
  type <- match.arg(type)

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

  ## point prediction via lars (predict.lars); NOT glmnet
  fit <- as.numeric(
    predict(object$lasso, newx = affinities, s = object$lasso_s,
            mode = object$lasso_mode, type = "fit")$fit
  )

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
