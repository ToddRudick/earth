#' bucketimpute: Bucketed Imputation Regression (BIR)
#'
#' Bucketed Imputation Regression (BIR) is a regression method with a secondary
#' uncertainty estimate. It works in five stages:
#' \enumerate{
#'   \item Divide the continuous response \code{y} into \code{n} equal-frequency
#'     quantile buckets.
#'   \item Inside each bucket fit a degree-1 \code{\link[earth]{earth}} (MARS)
#'     model for every predictor, reconstructing each column from all of the
#'     others.
#'   \item For any observation compute an \emph{unusualness} score per bucket
#'     (the sum over columns of standardized absolute reconstruction error).
#'   \item Convert those scores into \emph{affinity} features
#'     \code{1 / (1 + unusualness)}.
#'   \item Fit a cross-validated lasso of \code{y} on the affinity features.
#' }
#' At prediction time BIR returns a point prediction and a simple
#' heteroscedastic scale (an affinity-weighted average of the per-bucket
#' variances of \code{y}).
#'
#' @section Deliberate deviation from the specification (lars, not glmnet):
#' The original specification fits the final lasso with
#' \code{cv.glmnet(..., alpha = 1)} and predicts at \code{lambda.min}. This
#' package \strong{does not use glmnet}. Instead it uses the \code{lars}
#' package: it fits \code{lars::lars(affinities, y, type = "lasso")} and selects
#' the penalty with \code{lars::cv.lars(affinities, y, type = "lasso",
#' mode = "fraction", plot.it = FALSE)}, choosing the fraction with the
#' \emph{minimum} cross-validated MSE
#' (\code{s_opt = cvobj$index[which.min(cvobj$cv)]}). This is the lars analogue
#' of glmnet's \code{lambda.min}: the net behaviour is a CV-selected lasso that
#' matches the specification's intent; only the underlying fitting engine
#' differs. Point predictions use \code{predict.lars} with the stored fraction
#' and \code{mode = "fraction"}.
#'
#' @section Edge-case handling:
#' \itemize{
#'   \item \code{n} must be \code{>= 2}.
#'   \item If equal-frequency quantile bucketing of \code{y} does not yield
#'     \code{n} distinct non-empty buckets (heavy ties or near-binary \code{y}),
#'     \code{\link{fit_bir}} stops with a clear error; buckets are never
#'     silently merged.
#'   \item \code{min_bucket_rows} (default \code{max(p + 2, 10)}) sets the
#'     minimum bucket size for reliable per-bucket earth fits; thinner buckets
#'     warn, and buckets with fewer than 2 rows are an error.
#'   \item Residual scales \code{s_{k,j}} are floored at \code{scale_floor}
#'     (default \code{1e-8}) to avoid divide-by-zero.
#' }
#'
#' @seealso \code{\link{fit_bir}}, \code{\link{predict.bir}}
#'
#' @name bucketimpute-package
#' @aliases bucketimpute
#' @keywords internal
"_PACKAGE"
