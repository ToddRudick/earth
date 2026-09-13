#' Print a Bucketed Imputation Regression (BIR) model
#'
#' @param x A \code{"bir"} object from \code{\link{fit_bir}}.
#' @param ... Unused; present for S3 compatibility.
#' @return \code{x}, invisibly.
#' @examples
#' set.seed(1)
#' N <- 100
#' xm <- matrix(rnorm(N * 3), N, 3)
#' colnames(xm) <- c("a", "b", "c")
#' y <- xm[, 1] - 0.5 * xm[, 2] + rnorm(N, sd = 0.3)
#' print(fit_bir(xm, y, n = 4))
#' @export
print.bir <- function(x, ...) {
  cat("Bucketed Imputation Regression (BIR) model\n")
  cat(sprintf("  buckets (n) : %d\n", x$n))
  cat(sprintf("  predictors (p): %d\n", x$p))
  cat(sprintf("  columns     : %s\n",
              paste(x$colnames, collapse = ", ")))
  cat(sprintf("  lasso engine: lars (fraction s = %.4g, mode = %s)\n",
              x$lasso_s, x$lasso_mode))
  cat(sprintf("  min_bucket_rows: %d   scale_floor: %g\n",
              as.integer(x$min_bucket_rows), x$scale_floor))
  if (is.null(x$experts)) {
    cat("  experts     : none (affinity_lasso only)\n")
  } else {
    cat(sprintf("  experts     : %s (%d buckets; method = \"moe_soft\")\n",
                if (is.null(x$expert_type)) "yes" else x$expert_type,
                length(x$experts)))
  }
  invisible(x)
}

#' Summarize a Bucketed Imputation Regression (BIR) model
#'
#' @param object A \code{"bir"} object from \code{\link{fit_bir}}.
#' @param ... Unused; present for S3 compatibility.
#' @return \code{object}, invisibly.
#' @examples
#' set.seed(1)
#' N <- 100
#' xm <- matrix(rnorm(N * 3), N, 3)
#' colnames(xm) <- c("a", "b", "c")
#' y <- xm[, 1] - 0.5 * xm[, 2] + rnorm(N, sd = 0.3)
#' summary(fit_bir(xm, y, n = 4))
#' @export
summary.bir <- function(object, ...) {
  print(object)
  sc <- object$scales
  cat(sprintf("  residual scales s_{k,j}: range [%.4g, %.4g]\n",
              min(sc), max(sc)))
  cat(sprintf("  bucket variances of y  : range [%.4g, %.4g]\n",
              min(object$bucket_vars), max(object$bucket_vars)))
  invisible(object)
}
