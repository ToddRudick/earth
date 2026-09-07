# Fast R + caret setup for the adaptive-GCV experiments (Amazon Linux 2023)

The adaptive-GCV experiment runs (see the other `doc/adaptive_gcv_*.md`
reports) each start in a **fresh sandbox** and have to install R plus the
`caret` dependency tree before any modelling can happen. Historically that
install compiled `caret` and its dependencies **from CRAN source**, which
dominated wall-clock time (minutes of `gcc`/`g++`).

This note records a measured, reproducible **fast** setup: install R from
the system package manager, then install every CRAN package we need as a
**precompiled Linux binary** from the Posit Public Package Manager (PPM).
All numbers below were measured on this sandbox (Amazon Linux 2023,
`x86_64`, R 4.5.3, `Ncpus=4`).

> Scope: this is an environment/tooling note only. It does not touch the
> `earth` sources (`R/`, `src/`) and does not change any test.

---

## 1. Can the system package installer provide these packages? No.

The direct question first: **`dnf`/`yum` on Amazon Linux 2023 cannot install
`caret`, `cpp11`, `kernlab`, `AppliedPredictiveModeling`, `Formula`,
`plotmo`, `plotrix`, `TeachingDemos`, `Rcpp`, `ggplot2`, `foreach`, ...**

The only `amazonlinux` base repo is enabled, and it ships **R itself and
nothing else from CRAN**. The complete set of R RPMs available is:

```
R  R-core  R-core-devel  R-devel  R-java  R-java-devel  R-rpm-macros
```

There is **no `R-CRAN-*` package namespace at all**:

```console
$ dnf list available 'R-CRAN*'
Error: No matching Packages to list
$ dnf search caret
No matches found.
```

So `dnf` gives us the interpreter and the toolchain to build packages, but
the CRAN packages themselves must come from a CRAN-style repo. That is what
the rest of this note optimises.

---

## 2. Install R itself with dnf (fast, ~40s)

```console
$ dnf install -y R          # gives R 4.5.3 on AL2023
```

Measured: **`real 0m41.6s`**. This is fine as-is; leave it.

---

## 3. Install the CRAN packages as PPM binaries (the big win)

`install.packages()` from a plain CRAN mirror compiles from source. PPM
serves **prebuilt Linux binaries**, chosen by the client's HTTP
`User-Agent`. Amazon Linux 2023 is RHEL9-family, and the PPM **RHEL9**
binaries are **ABI-compatible** here (verified below).

### Reachability caveat (important)

In this sandbox the canonical PPM host `packages.posit.co` **does not
resolve**:

```console
$ curl -sI https://packages.posit.co/...   ->  Could not resolve host
```

but the newer Posit domain **`p3m.dev` does resolve and works**. Use
`https://p3m.dev/cran/...`. (`cloud.r-project.org` / `cran.rstudio.com`
also resolve, for the source fallback.)

### The exact options that yield binaries

```r
options(repos = c(CRAN = "https://p3m.dev/cran/__linux__/rhel9/latest"))
options(HTTPUserAgent = sprintf(
  "R/%s R (%s)",
  getRversion(),
  paste(getRversion(), R.version["platform"], R.version["arch"], R.version["os"])
))
```

PPM keys off that `User-Agent` to decide binary-vs-source. With it set,
downloads arrive as `Content type 'binary/octet-stream'` and install with
`* installing *binary* package ...` — **no compiler is invoked**.

### Verified: RHEL9 binaries work on AL2023 (yes, with evidence)

`Rcpp` (a package that normally compiles) installed from PPM as a binary
and loaded cleanly — no ABI/linker errors:

```
trying URL '.../rhel9/latest/src/contrib/Rcpp_1.1.2.tar.gz'
Content type 'binary/octet-stream' length 2205161 bytes (2.1 MB)
* installing *binary* package 'Rcpp' ...
* DONE (Rcpp)
```

Functional check after installing the full set: `library(caret)`,
`library(kernlab)`, `library(AppliedPredictiveModeling)` all load, the
`spam` (4601x58) and `solubility` datasets load, and a real
`caret::train(method="lm", trControl=trainControl("cv"))` runs to
completion. That exercises the compiled `Rcpp`/`foreach` paths, so the
binaries are genuinely functional, not just loadable.

### Measured timings: source vs PPM binary

| What | Route | Time | Notes |
|------|-------|------|-------|
| `Rcpp` alone | CRAN source | **26.5s** | 1 package compiled |
| `Rcpp` alone | PPM binary  | **1.4s**  | binary, loads OK |
| `cpp11` + `caret` (+63 compiled deps) | CRAN source, `Ncpus=4` | **4m30s** | 63 source compiles |
| `Formula`,`plotmo`,`plotrix`,`TeachingDemos`,`cpp11`,`caret`,`kernlab`,`AppliedPredictiveModeling` (+all deps: 70 pkgs) | PPM binary, `Ncpus=4` | **8s** | 70 binaries, 0 compiles |

So the PPM binary route replaces a **~4.5 minute** caret build (and even
more once `kernlab` + `AppliedPredictiveModeling` are added) with an
**8-second** download-and-unpack of the entire dependency set: roughly a
**30x+ speedup**, and it grows with the number of packages.

---

## 4. Library location and reuse

Each experiment subagent starts in a fresh sandbox. Empirically, within a
sandbox **`/usr` persists** (R stays installed once `dnf`-installed) but
**`/tmp` does not persist** across separate command invocations, and there
is **no verified cross-session persistent volume** for an R library.

Per the task owner, the setup does **not** depend on a cached library
surviving across sessions — the PPM binary downloads are fast enough (8s
for the whole tree) that re-installing each session is cheap. What matters
is remembering **which repo + options to use**, which this doc captures.

If you still want to avoid re-downloading within a session, install into a
workspace-relative, **git-ignored** library and point `.libPaths()` at it:

```r
lib <- "/projects/sandbox/.Rcache"          # git-ignored, see .gitignore
dir.create(lib, showWarnings = FALSE, recursive = TRUE)
.libPaths(lib)
```

A `.gitignore` entry (`/.Rcache/` at the workspace root) keeps this library
out of git so the repo is never bloated by a committed library tree.

**Persistence status: NOT verified as cross-session.** `/projects/sandbox`
is the git working tree and is the most likely-persistent area, but I could
not spawn a new sandbox to prove a later run sees the cache. Treat the cache
purely as an in-session convenience; correctness of the workflow relies only
on the PPM repo being reachable, not on any cache surviving.

---

## 5. Quick start (copy-paste)

```bash
# 1) R from the system package manager (~40s)
dnf install -y R

# 2) All CRAN deps as PPM RHEL9 binaries (~8s), into a git-ignored library
Rscript -e '
  lib <- "/projects/sandbox/.Rcache"
  dir.create(lib, showWarnings = FALSE, recursive = TRUE)
  .libPaths(lib)
  options(repos = c(CRAN = "https://p3m.dev/cran/__linux__/rhel9/latest"))
  options(HTTPUserAgent = sprintf("R/%s R (%s)", getRversion(),
    paste(getRversion(), R.version["platform"], R.version["arch"], R.version["os"])))
  pkgs <- c("Formula","plotmo","plotrix","TeachingDemos",
            "cpp11","caret","kernlab","AppliedPredictiveModeling")
  install.packages(pkgs, lib = lib, Ncpus = 4)
'

# 3) Build/install the earth package under test from its source tree as usual,
#    e.g.  R CMD INSTALL --library=/projects/sandbox/.Rcache /projects/sandbox/earth
#    then run the slowtests / doc scripts with:
#      Rscript -e '.libPaths("/projects/sandbox/.Rcache"); source("inst/slowtests/....R")'
```

If `p3m.dev` is ever unreachable, fall back to a plain CRAN mirror with
`type="source"` and `Ncpus=4` (the slow ~4.5-minute route measured above);
`https://cloud.r-project.org` and `https://cran.rstudio.com` both resolve
here.
