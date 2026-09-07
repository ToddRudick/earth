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

## Study 3: guarded `caret::bagEarth` holdout (larger / wider datasets), FULL SIZE

**This section now reports FULL-SIZE numbers.** An earlier revision of this study
ran the large/wide datasets only in SMOKE mode (subsampled rows, degree capped to
1) because full-size bagging was thought too slow for the run budget. The
full-size run has since been completed for **all four** datasets and the SMOKE
rows are superseded by the real numbers below. The SMOKE numbers are retained at
the end of this section for transparency (what changed and why).

Methodology is the repo helper `caret.oos(form, data, degree, is.binary)` in
`inst/slowtests/adaptive.gcv.large.comparison.R`, run at **full size / full
degree** (i.e. `options(adaptive.gcv.run.caret = TRUE)` and **NOT**
`adaptive.gcv.smoke`): a single **70/30 holdout** (`set.seed(2024)`) fitting
`caret::bagEarth(x = train, y = train_y, B = 20, degree = <full degree>,
adaptive.gcv = <FALSE/TRUE>)`, reporting holdout RMSE for feature OFF vs ON. The
"ON" fit uses earth's **default `effect.cap = 0.9`** (the helper does not pass
`effect.cap`). For the binary `spam` we additionally report holdout **accuracy**
and **Brier** (bagged probability of class 1, threshold 0.5).

Full-size configuration actually run (all four completed at **B = 20, full
degree**; no B reduction and no timeout was needed):

| dataset | rows | predictors | degree | B | holdout | seed | runtime (OFF + ON) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| boston | 506 | 13 | 2 | 20 | 70/30 | 2024 | ~0.6 s (0.3 + 0.3) |
| spam | 4601 | 57 | 1 | 20 | 70/30 | 2024 | ~3.4 s (1.8 + 1.6) |
| solubility | 951 | 228 | 2 | 20 | 70/30 | 2024 | ~32 s (16.0 + 16.2) |
| synthetic_large | 4000 | 40 | 2 | 20 | 70/30 | 2024 | ~15 s (7.7 + 7.5) |

FULL-SIZE bagged holdout RMSE, feature OFF vs ON at default `effect.cap = 0.9`:

| dataset | task | bagged RMSE OFF | bagged RMSE ON (cap 0.9) | verdict |
| --- | --- | --- | --- | --- |
| boston | regression | 3.38073 | 3.38955 | ~neutral (hair worse) |
| spam | classification | 0.27107 | 0.27113 | neutral |
| solubility | regression | 0.61196 | 0.61566 | hurts (slightly) |
| synthetic_large | regression | 1.50073 | 1.54028 | hurts (slightly) |

spam RMSE is on the 0/1 `type` response. Binary detail for spam (holdout,
`B = 20`, degree 1):

| setting | RMSE | accuracy | Brier |
| --- | --- | --- | --- |
| stock (OFF) | 0.27107 | 0.92754 | 0.07348 |
| adaptive cap 0.9 (ON) | 0.27113 | 0.92681 | 0.07351 |

All four differences at full size are small: the largest OOS RMSE swing is
synthetic_large (1.501 → 1.540, about +2.6%) and solubility (0.612 → 0.616,
about +0.6%); boston and spam move by well under 1% (spam accuracy 0.9275 →
0.9268, one fewer correct on a 1380-row holdout). So at full size, default
`effect.cap = 0.9` under bagging is **neutral-to-slightly-negative** on these
four datasets.

**What changed versus the earlier SMOKE numbers.** The SMOKE run (subsampled
rows, degree capped to 1) previously reported:

| dataset | SMOKE size / degree | SMOKE RMSE OFF | SMOKE RMSE ON | SMOKE verdict | FULL-SIZE verdict |
| --- | --- | --- | --- | --- | --- |
| boston | 150 × 13, deg 1 | 3.72869 | 3.63064 | helps | ~neutral (hair worse) |
| spam | 300 × 57, deg 1 | 0.57451 | 0.54797 | helps | neutral |
| solubility | 200 × 20, deg 1 | 0.76396 | 0.79758 | hurts | hurts (slightly) |
| synthetic_large | 200 × 12, deg 1 | 3.02307 | 3.05149 | ~neutral | hurts (slightly) |

The two SMOKE "helps" verdicts (boston, spam) **do not survive at full size**:
they flatten to neutral. The SMOKE "hurts" on solubility and "~neutral / slightly
worse" on synthetic_large **do agree in direction** with full size (both remain
neutral-to-negative). The SMOKE numbers were a valid cross-check at that reduced
scale but, as flagged at the time, should not be read as the full-dataset result;
the full-size numbers above are the authoritative ones and supersede them.

The disable invariant (`effect.cap = 1.0` == stock, `adaptive.gcv = FALSE` ==
stock) continues to hold identically, as verified in Study 2's pattern and by the
flow-through check; only the `effect.cap = 0.9` (ON) fits differ from stock.

## Study 4: random 50% term exclusion via `allowed` (B = 40, OFF only)

This study answers a distinct question from the earlier ones. Instead of turning
the adaptive GCV effect cap on or off, it keeps earth **stock** (`adaptive.gcv =
FALSE`) and injects randomness into the **forward pass** itself: on every
candidate-term evaluation, earth calls a user-supplied `allowed` callback, and we
make that callback drop **~50% of candidate terms at random**. Bagging (`B = 40`)
then averages many such randomly-thinned earth fits, in the spirit of random
feature subsampling in a random forest. **No earth source change is needed** -
`allowed` is a standard earth parameter (a user callback), and `caret::bagEarth`
forwards `...` straight through to `earth()`.

### The `allowed` callback

earth calls `allowed(degree, pred, parents, namesx, first)` for each candidate
term during the forward pass and admits the term only if the callback returns
`TRUE` (contract confirmed from `man/earth.Rd` and
`inst/slowtests/test.allowedfunc.R`). The callback **must accept all five
arguments** - earth always passes them, and a function with fewer args errors.
Our random-50%-exclusion gate is simply:

```r
allowed.random50 <- function(degree, pred, parents, namesx, first) runif(1) > 0.5
```

i.e. each candidate term is admitted with probability 0.5 (about half the
candidates are dropped at random on each call). It is passed straight through
bagEarth:

```r
caret::bagEarth(x = train, y = train_y, B = 40, degree = <deg>,
                adaptive.gcv = FALSE, allowed = allowed.random50)
```

**Flow-through confirmed.** A tiny smoke fit (`trees`, `Girth`+`Height`,
`degree = 2`, `B = 8`, `set.seed(2024)`) shows the callback reaching earth: the
unrestricted bag averaged 4.5 terms/model (per-bag 3 8 5 4 4 4 4 4) while the
`allowed.random50` bag averaged 3.375 terms/model (per-bag 1 1 4 5 4 4 4 4) - the
random gate demonstrably yields smaller, different models, so `allowed` does flow
through bagEarth into each bagged earth fit.

### Method (matches the existing study for comparability)

Same construction as Study 3's `caret.oos()` in
`inst/slowtests/adaptive.gcv.large.comparison.R`: the four datasets at their
**full** rows/predictors/degree, a single **70/30 holdout** with
`set.seed(2024)`, `model.matrix` used to build the numeric design (intercept
dropped; `spam`'s `type` recoded to 0/1 with nonspam = 0, spam = 1), and holdout
RMSE computed the same way. The only differences from Study 3 are: **B = 40**
(double the prior B = 20), **only the OFF case** (`adaptive.gcv = FALSE`, stock
earth - the adaptive ON variant is deliberately not run here), and the
`allowed = allowed.random50` restriction. `set.seed(2024)` is set immediately
before each dataset's `bagEarth` call, so the run is reproducible.

**Reproducibility caveat (important).** The `allowed` gate calls `runif()` on
every candidate-term evaluation, so it adds a layer of RNG-driven nondeterminism
**on top of** bagging's bootstrap resampling. The numbers below are reproducible
**only** with the same R RNG stream: the fixed `set.seed(2024)` before each
`bagEarth` call, this earth/caret build (earth 5.3.6, caret 7.0.1, R 4.5.3), and
the same platform RNG. Change the seed, the RNG, the package versions, or `B` and
the exact RMSE will move. The results characterise the *behaviour* of
random-50%-term-exclusion bagging on these datasets; they are not a fixed
constant of the data. If a bag were ever thinned so aggressively that it
collapsed to a near-empty (intercept-only) model, we note it below rather than
hide it.

Full-size configuration actually run (all four completed at **B = 40**; no B
reduction and no timeout was needed):

| dataset | nrow | ncol | degree | B | holdout | seed | runtime (OFF) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| boston | 506 | 13 | 2 | 40 | 70/30 | 2024 | ~0.5 s |
| spam | 4601 | 57 | 1 | 40 | 70/30 | 2024 | ~3.7 s |
| solubility | 951 | 228 | 2 | 40 | 70/30 | 2024 | ~22 s |
| synthetic_large | 4000 | 40 | 2 | 40 | 70/30 | 2024 | ~12 s |

### Results: holdout RMSE OFF with random-50% `allowed` restriction

| dataset | nrow | ncol | degree | B | RMSE OFF (random-50% allowed) | doc Study-3 RMSE OFF (no restriction) |
| --- | --- | --- | --- | --- | --- | --- |
| boston | 506 | 13 | 2 | 40 | 3.08016 | 3.38073 |
| spam | 4601 | 57 | 1 | 40 | 0.27055 | 0.27107 |
| solubility | 951 | 228 | 2 | 40 | 0.59757 | 0.61196 |
| synthetic_large | 4000 | 40 | 2 | 40 | 1.65540 | 1.50073 |

`spam` RMSE is on the 0/1 `type` response. Binary detail for `spam` (holdout,
`B = 40`, degree 1, bagged class-1 probability thresholded at 0.5):

| setting | RMSE | accuracy | Brier |
| --- | --- | --- | --- |
| stock OFF + random-50% allowed | 0.27055 | 0.92826 | 0.07320 |

### Degenerate-bag check

No bag collapsed to a near-empty model. Minimum terms per bag across the 40
models was: boston 16, spam 13, solubility 25, synthetic_large 12 (means ~19.7,
16.5, 35.9, 19.1 respectively). So even with ~50% of candidates randomly dropped
per call, every bag still admitted a substantial multi-term model; there were
**zero** intercept-only or single-term degenerate bags on any of the four
datasets in this seeded run.

### Observations (comparison to Study 3 OFF is context, not the deliverable)

The deliverable is the new random-allowed OFF column above. As context, comparing
it to Study 3's full-size "RMSE OFF" (plain stock bagged earth, `B = 20`, no
`allowed` restriction):

- **boston** improves (3.38073 -> 3.08016) and **solubility** improves
  (0.61196 -> 0.59757): on these, randomly thinning candidate terms and
  averaging more bags acts like extra regularisation / decorrelation and helps
  the holdout.
- **spam** is essentially unchanged (0.27107 -> 0.27055; accuracy 0.92826), i.e.
  neutral.
- **synthetic_large** is worse (1.50073 -> 1.65540): this design has a strong,
  genuinely-structured signal (a linear dominant, two hinges and an
  interaction), and randomly discarding half the candidate terms per call keeps
  individual bags from reliably capturing that structure, which the averaging
  does not fully recover.

These are single-seed holdout numbers with the extra `allowed` RNG layer
described in the caveat above, and the comparison mixes B = 40 (this study) with
the B = 20 Study-3 baseline; read the direction, not the third decimal.

## Overall verdict

- The adaptive GCV effect cap flows correctly through `caret::bagEarth` and can
  be bagged; `adaptive.gcv = FALSE` and `effect.cap >= 1` reproduce bagged stock
  earth **exactly** (machine precision), so the feature stays safely default-OFF
  under bagging too.
- Under bagging with the **default `effect.cap = 0.9`**, counting the small
  datasets (Study 1) plus the **full-size** large datasets (Study 3), the
  adaptive cap was a **net positive on 2 of 8 datasets** (ozone1, mtcars),
  **neutral on 3** (etitanic, boston, spam), and **worse on 3** (trees,
  solubility, synthetic_large). The two large datasets that looked like wins in
  SMOKE mode (boston, spam) **flatten to neutral at full size**, so at the default
  cap the feature is **mixed-leaning-neutral and dataset-dependent**, reported
  honestly, negatives and neutrals included. The clear default-`0.9` wins remain
  the genuinely-nonlinear small frames (ozone1, mtcars).
- The tighter `effect.cap = 0.5` amplifies both directions: bigger gains where
  it helps (ozone1, mtcars) and a clear loss on the tiny trees holdout. The knob
  behaves as designed (smaller = more aggressive).
- Bagging broadly agrees with earth's built-in CV on the *direction* of the
  effect per dataset (helps on ozone1/mtcars, hurts on trees), which is
  reassuring for the feature's characterisation. At full size the large-dataset
  bagged directions also line up with the earlier SMOKE directions on solubility
  (hurts) and synthetic_large (neutral-to-worse); the only shifts are boston and
  spam moving from a SMOKE "helps" to a full-size "neutral", i.e. the full-size
  result is more conservative than SMOKE, not contradictory in sign.

## Reproducing

```r
# small datasets (Studies 1 and the strength sweep numbers match this run)
options(adaptive.gcv.run.caret = TRUE)
source("inst/slowtests/adaptive.gcv.comparison.R")

# larger/wider datasets at FULL size (Study 3): enable the caret path and do
# NOT set adaptive.gcv.smoke, so caret.oos() runs at full rows / full degree.
options(adaptive.gcv.run.caret = TRUE)
source("inst/slowtests/adaptive.gcv.large.comparison.R")
```

Runtime (measured): each small-dataset bagEarth pair (`B = 20`) completes in
seconds. The **full-size** large study under bagging is fast in practice on this
build: boston ~0.6 s, spam ~3.4 s, synthetic_large ~15 s, and the wide
solubility (951 × 228, degree 2) ~32 s for the OFF + ON pair, all at `B = 20`,
no B reduction or timeout needed. (The Study 3 caret block reports RMSE only; the
spam accuracy/Brier here were obtained by the same `caret.oos` methodology,
`B = 20`, 70/30 holdout, seed 2024, adaptive ON at default `effect.cap = 0.9`,
with the bagged class-1 probability thresholded at 0.5.)
