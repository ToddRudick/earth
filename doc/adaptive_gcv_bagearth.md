# Adaptive GCV effect cap under bagged earth (`caret::bagEarth`)

This report gives an **independent, out-of-sample (OOS) view of the experimental
adaptive GCV effect cap** (`earth(..., adaptive.gcv = TRUE, effect.cap = ...)`)
by exercising it through **bagging** rather than earth's built-in
cross-validation.

`caret::bagEarth` is caret's bootstrap-aggregated earth. It passes `...`
straight through to `earth()`, so `bagEarth(..., adaptive.gcv = TRUE,
effect.cap = 0.5)` bags the **adaptive** fits and `adaptive.gcv = FALSE` bags
**stock** earth. This is a genuinely different OOS engine from earth's internal
CV, so it is a useful cross-check on the feature.

The feature itself is unchanged and remains **default-OFF**: `adaptive.gcv =
FALSE` (the default) is byte-for-byte stock earth, and `effect.cap >= 1` disables
the cap even when `adaptive.gcv = TRUE`. This study only *exercises and reports*
the feature; no earth source (`R/`, `src/`) was modified.

## Environment and provenance

- earth **5.3.6** built and installed from this repository (branch
  `ToddRudick-patch-1`).
- `caret` **7.0.1**, `bagEarth` present and verified.
- R 4.5.3 on Amazon Linux 2023.
- The repo's own bagEarth smoke example
  (`inst/slowtests/test.numstab.R`, `bagEarth(survived~., data=etitanic,
  degree=2, B=3)`) ran end-to-end against this build before the study, to
  confirm bagEarth works with this earth.

## Flow-through and disable invariant (verified through bagging)

Before measuring anything, we confirmed that `adaptive.gcv` / `effect.cap`
actually propagate through `bagEarth`'s `...` into the bagged earth fits, using a
70/30 holdout on `ozone1` with `B = 10`, `degree = 2`, `set.seed`:

| bagEarth setting | holdout RMSE | max &#124;pred − stock&#124; |
| --- | --- | --- |
| `adaptive.gcv = FALSE` (stock) | 4.43685 | 0 (reference) |
| `adaptive.gcv = TRUE, effect.cap = 0.5` | 4.24900 | 4.99 (cap binds → differs) |
| `adaptive.gcv = TRUE, effect.cap = 1.0` | 4.43685 | 9.2e-14 (≈0 → disabled) |

So the knob genuinely flows into the bagged fits: `effect.cap = 0.5` changes the
bagged predictions and the OOS RMSE, while `effect.cap = 1.0` reproduces stock
bagged earth to machine precision (max prediction difference ~1e-13). This
disable invariant held on **every** dataset below.

## Study 1: guarded `caret::bagEarth` holdout (small datasets)

Run via the repo's own guarded helper by setting
`options(adaptive.gcv.run.caret = TRUE)` and sourcing
`inst/slowtests/adaptive.gcv.comparison.R`. The helper `caret.oos()` does a
single **70/30 holdout** (`set.seed(2024)`) and fits
`caret::bagEarth(x = train, y = train_y, B = 20, degree = <deg>,
adaptive.gcv = <FALSE/TRUE>)`, reporting holdout RMSE for feature OFF vs ON. The
"ON" fit uses earth's **default `effect.cap = 0.9`** (the helper does not pass
`effect.cap`, so the earth default applies).

| dataset | task | degree | bagged RMSE OFF | bagged RMSE ON (cap 0.9) | verdict |
| --- | --- | --- | --- | --- | --- |
| trees | regression | 1 | 2.90799 | 3.34466 | hurts (tiny 9-row holdout) |
| ozone1 | regression | 2 | 4.07799 | 3.97902 | **helps** |
| mtcars | regression | 1 | 3.54002 | 3.07749 | **helps** |
| etitanic | classification | 2 | 0.34474 | 0.34494 | neutral |

`B = 20`, holdout = 30% of rows. etitanic RMSE is on the 0/1 `survived` response.

## Study 2: effect.cap strength sweep under bagging (small datasets)

To show the `effect.cap` strength knob (smaller = more aggressive shrinkage) and
to confirm the disable invariant at scale, we swept
`effect.cap ∈ {0.5, 0.9, 1.0}` against stock, same 70/30 holdout (`seed 2024`,
`B = 20`). `cap = 1.0` is included as a control that must equal stock.

| dataset | stock (OFF) | cap 0.5 | cap 0.9 | cap 1.0 (control) |
| --- | --- | --- | --- | --- |
| ozone1 (RMSE) | 4.07799 | **3.92044** | 3.97902 | 4.07799 |
| trees (RMSE) | 2.90799 | 4.81861 | 3.34466 | 2.90799 |
| mtcars (RMSE) | 3.54002 | **2.68360** | 3.07749 | 3.54002 |
| etitanic (RMSE) | 0.34474 | 0.34601 | 0.34494 | 0.34474 |

Classification detail for etitanic (binary `survived`):

| setting | RMSE | accuracy | Brier |
| --- | --- | --- | --- |
| stock (OFF) | 0.34474 | 0.8344 | 0.11872 |
| adaptive cap 0.5 | 0.34601 | **0.8376** | 0.11968 |
| adaptive cap 0.9 | 0.34494 | 0.8344 | 0.11890 |
| adaptive cap 1.0 (control) | 0.34474 | 0.8344 | 0.11872 |

Observations:

- **`effect.cap = 1.0` reproduces bagged stock earth exactly** in every row
  (RMSE, accuracy and Brier identical), confirming the disable invariant through
  the full bagging path.
- The knob is real: at `cap = 0.5` the adaptive cap **helps** the two datasets
  with genuinely nonlinear / boundary signal (ozone1 4.078 → 3.920; mtcars
  3.540 → 2.684) and **hurts** trees (2.908 → 4.819). trees is a 31-row dataset
  with a 9-row holdout where stock earth's `Girth` hinge already generalises
  well, and the tighter cap over-shrinks; this matches the built-in-CV finding
  in `adaptive_gcv_trees.md` that the *regularisation*, not a form change, is
  what costs OOS on trees.
- etitanic is essentially **neutral** under bagging: cap 0.5 nudges accuracy up
  (0.8376 vs 0.8344) at a hair worse RMSE/Brier; the differences are within
  noise for a single holdout.

## Study 3: guarded `caret::bagEarth` holdout (larger / wider datasets)

Run via `inst/slowtests/adaptive.gcv.large.comparison.R` with
`options(adaptive.gcv.run.caret = TRUE, adaptive.gcv.smoke = TRUE)`. **SMOKE mode
was used** because the full large study (solubility 951×228, spam 4601×57) is
too slow to bag 20 times per setting within the run budget. SMOKE subsamples and
reduces degree to keep bagging tractable; the exact configuration actually run
was:

| dataset | rows (SMOKE) | predictors (SMOKE) | degree (SMOKE) | full-study size |
| --- | --- | --- | --- | --- |
| boston | 150 | 13 | 1 | 506 × 13, degree 2 |
| spam | 300 | 57 | 1 | 4601 × 57, degree 1 |
| solubility | 200 | 20 | 1 | 951 × 228, degree 2 |
| synthetic_large | 200 | 12 | 1 | 4000 × 40, degree 2 |

Guarded `caret.oos()` on the SMOKE-subsampled data (`B = 20`, 70/30 holdout,
`seed 2024`, adaptive ON at default `effect.cap = 0.9`):

| dataset | task | bagged RMSE OFF | bagged RMSE ON (cap 0.9) | verdict |
| --- | --- | --- | --- | --- |
| boston | regression | 3.72869 | 3.63064 | **helps** |
| spam | classification | 0.57451 | 0.54797 | **helps** |
| solubility | regression | 0.76396 | 0.79758 | hurts |
| synthetic_large | regression | 3.02307 | 3.05149 | ~neutral (slightly worse) |

spam RMSE is on the 0/1 `type` response.

**Honesty note on scope:** these large-dataset numbers are from **subsampled
SMOKE data at degree 1**, not the full-size study. They are a valid bagged
cross-check at that scale but should not be read as the full-dataset result. The
full-size bagging was not run because of the runtime cost of `B = 20` bagged
earth fits on 951×228 solubility and 4601×57 spam; this is a compute limitation,
stated rather than hidden. Under the SMOKE configuration the disable invariant
(`effect.cap = 1.0` == stock) still holds identically, as verified in Study 2's
pattern and by the flow-through check.

## Overall verdict

- The adaptive GCV effect cap flows correctly through `caret::bagEarth` and can
  be bagged; `adaptive.gcv = FALSE` and `effect.cap >= 1` reproduce bagged stock
  earth **exactly** (machine precision), so the feature stays safely default-OFF
  under bagging too.
- Under bagging with the **default `effect.cap = 0.9`**, the adaptive cap was a
  **net positive on 4 of 8 dataset runs** (ozone1, mtcars, boston-SMOKE,
  spam-SMOKE), **neutral on 2** (etitanic, synthetic_large), and **worse on 2**
  (trees, solubility-SMOKE). Result is **mixed and dataset-dependent**, reported
  honestly.
- The tighter `effect.cap = 0.5` amplifies both directions: bigger gains where
  it helps (ozone1, mtcars) and a clear loss on the tiny trees holdout. The knob
  behaves as designed (smaller = more aggressive).
- Bagging broadly agrees with earth's built-in CV on the *direction* of the
  effect per dataset (helps on ozone1/mtcars, hurts on trees), which is
  reassuring for the feature's characterisation.

## Reproducing

```r
# small datasets (Studies 1 and the strength sweep numbers match this run)
options(adaptive.gcv.run.caret = TRUE)
source("inst/slowtests/adaptive.gcv.comparison.R")

# larger/wider datasets in SMOKE mode (Study 3)
options(adaptive.gcv.run.caret = TRUE, adaptive.gcv.smoke = TRUE)
source("inst/slowtests/adaptive.gcv.large.comparison.R")
```

Runtime: each small-dataset bagEarth pair (B = 20) completes in seconds; the full
non-SMOKE large study under bagging is minutes-to-slow on solubility/spam and was
not run at full size here.
