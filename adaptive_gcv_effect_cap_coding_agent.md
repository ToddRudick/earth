# Adaptive GCV Effect Cap for MARS Forward Selection

## Objective

Explore and implement an **Adaptive GCV Effect Cap** in the R package `earth` (MARS), specifically in the **forward pass**, not as a backward-pass coefficient shrinkage or pruning method.

The desired behavior is:

> A candidate MARS basis function should be allowed to enter the model and have its coefficient increase only up to the amount of predictive effect justified by the model's current GCV/complexity tradeoff. If its unconstrained least-squares coefficient would produce more effect than is justified, add the term at the maximum allowed effect and then allow the forward search to find other terms.

The purpose is to prevent one term from absorbing a disproportionate amount of the remaining signal and thereby give other potentially useful terms an opportunity to enter the model.

This should be treated as an experimental statistical feature. Preserve the existing `earth` behavior by default.

---

## What has been established so far

### 1. This is a forward-pass idea

Do **not** implement this primarily as:

- backward pruning,
- post-hoc coefficient shrinkage,
- ridge,
- lasso,
- or simply a coefficient bound.

The desired sequence is:

```text
generate candidate basis function
        ↓
evaluate its unconstrained least-squares coefficient
        ↓
determine its maximum justified predictive effect
        ↓
if OLS effect is below cap:
       add normally
else:
       add at cap
        ↓
term is saturated if it hit the cap
        ↓
continue forward search against remaining residual
```

A candidate that exceeds its cap should **not automatically be rejected**. It should be admitted at the boundary.

---

## 2. Do not constrain the raw coefficient

A raw MARS coefficient is not invariant to scaling.

For example:

\[
\beta \max(x-5,0)
\]

will have a different numerical coefficient if `x` is expressed in dollars rather than thousands of dollars, even though the fitted function can be identical.

Therefore the meaningful quantity to constrain is the **predictive effect of the term**, not the raw numerical value of `beta`.

Potential effect measures include:

\[
\operatorname{Var}(\beta B_j)
\]

or, more directly, the candidate's incremental reduction in RSS/R².

The preferred direction is to define the cap in terms of **incremental predictive improvement**.

---

## 3. Define the candidate's effect through incremental fit

Let the current model have

\[
RSS_0.
\]

For candidate basis function \(B_j\), with coefficient \(b\), define

\[
RSS_j(b)
\]

as the RSS after adding \(bB_j\) to the current model.

Then define the candidate's incremental improvement:

\[
\Delta RSS_j(b)
=
RSS_0-RSS_j(b).
\]

Equivalently:

\[
\Delta R^2_j(b)
=
\frac{RSS_0-RSS_j(b)}{TSS}.
\]

The cap should preferably be expressed as:

\[
\boxed{
\Delta R^2_j(b)\leq\Delta R^2_{\max,j}
}
\]

rather than:

\[
|\beta_j|\leq K.
\]

This keeps the concept tied to predictive contribution and avoids arbitrary units.

---

# Core proposed idea: Adaptive GCV Effect Cap

The central hypothesis is:

> The amount of predictive improvement a newly admitted term is allowed to contribute should be determined adaptively from the same GCV/complexity principle that MARS already uses to decide when model growth is no longer worthwhile.

The existing MARS GCV is approximately:

\[
GCV =
\frac{RSS/n}{(1-C/n)^2}
\]

where \(C\) is the effective model complexity.

The MARS effective complexity is approximately of the form

\[
C(M)=M+p\frac{M-1}{2},
\]

where:

- \(M\) is the number of basis functions,
- \(p\) is the MARS `penalty`.

The exact details of the current `earth` implementation should be inspected rather than assumed. Reuse the package's existing GCV/penalty machinery wherever possible.

---

# Desired interpretation

For a candidate term, consider gradually increasing its coefficient from zero toward the unconstrained OLS coefficient:

\[
b(t)=t\hat b,\qquad 0\le t\le1.
\]

At \(t=0\), the term contributes nothing.

At \(t=1\), the term has its unconstrained least-squares coefficient.

The proposed algorithm should identify the point at which the candidate's additional predictive benefit is no longer justified by the relevant GCV/complexity cost.

Conceptually:

\[
\boxed{
\text{allow coefficient growth while marginal predictive benefit}
>
\text{marginal complexity cost}
}
\]

When the equality/boundary is reached, the candidate is saturated.

---

# Important distinction from ordinary GCV term selection

Ordinary MARS forward selection effectively asks:

> Which candidate basis function gives the greatest reduction in error?

The proposed algorithm asks a more nuanced question:

> Which candidate gives the greatest justified improvement, and how much of that improvement should it be allowed to realize?

This means a strong candidate may enter the model but be deliberately prevented from explaining all the signal that its unconstrained coefficient would explain.

This is intended to encourage the forward search to discover multiple explanatory terms rather than allowing one term to dominate merely because it has the largest immediate RSS reduction.

---

# Desired forward algorithm

Implement and test an experimental version along these lines:

### Step 1: Existing candidate generation

Leave the existing MARS candidate generation mechanism unchanged initially.

Generate the same candidate hinge functions and interactions that ordinary `earth` would consider.

### Step 2: Existing candidate evaluation

For each candidate, obtain the unconstrained least-squares coefficient and its corresponding RSS improvement.

Do not unnecessarily rewrite the existing linear algebra.

### Step 3: Determine the adaptive effect cap

For each candidate, determine the maximum incremental effect justified by the current GCV/complexity state.

The cap should depend on the current state of the model, not be a fixed percentage of R² unless used as a diagnostic baseline.

Potential inputs to the cap include:

- `n`,
- current RSS,
- current effective complexity,
- current GCV,
- MARS `penalty`,
- the incremental complexity associated with admitting the candidate,
- candidate basis-function scale,
- and potentially correlations with the existing basis matrix.

### Step 4: Compare OLS effect with cap

If the unconstrained candidate effect satisfies:

\[
\Delta R^2_{\text{OLS}}
\le
\Delta R^2_{\max},
\]

admit the candidate normally.

If:

\[
\Delta R^2_{\text{OLS}}
>
\Delta R^2_{\max},
\]

solve for the coefficient \(b_{\max}\) satisfying approximately:

\[
\Delta R^2_j(b_{\max})
=
\Delta R^2_{\max}.
\]

Then add the candidate using:

\[
b=b_{\max}.
\]

### Step 5: Mark saturated terms

A candidate that hits the cap should be treated as **saturated** for purposes of the forward search.

It should not be allowed to continue increasing its coefficient simply because later residual changes make that attractive.

However, carefully investigate how this interacts with later basis additions and interactions. The first implementation may need a simpler rule.

### Step 6: Continue forward selection

After adding a capped term, recompute the residual and continue the ordinary forward search.

The important intended behavior is:

```text
strong term
    ↓
reaches effect cap
    ↓
residual still contains signal
    ↓
other candidate terms compete for entry
```

---

# A useful mathematical route

The simplest candidate-specific derivation may come from writing the RSS as a quadratic function of the candidate coefficient.

Given current residual vector \(r\) and candidate basis vector \(B\):

\[
RSS(b)=
(r-bB)^T(r-bB).
\]

Therefore:

\[
RSS(b)
=
r^Tr
-2bB^Tr
+b^2B^TB.
\]

The unconstrained OLS coefficient is:

\[
\hat b
=
\frac{B^Tr}{B^TB}.
\]

The RSS improvement at coefficient \(b\) is:

\[
\Delta RSS(b)
=
2bB^Tr-b^2B^TB.
\]

Equivalently, using \(\hat b\):

\[
\Delta RSS(b)
=
(B^TB)\left(2b\hat b-b^2\right).
\]

The maximum occurs at \(b=\hat b\).

If the desired cap is stated as a maximum allowable \(\Delta RSS_{\max}\), then solve:

\[
(B^TB)\left(2b\hat b-b^2\right)
=
\Delta RSS_{\max}.
\]

This has a closed-form solution on the path from zero toward the OLS coefficient.

For a positive OLS coefficient:

\[
b_{\max}
=
\hat b-
\sqrt{
\hat b^2-\frac{\Delta RSS_{\max}}{B^TB}
}.
\]

For a negative coefficient, use the corresponding signed solution.

This is potentially much simpler and more numerically stable than repeatedly refitting the model for different coefficient values.

**However:** this derivation assumes the candidate is represented as a residualized direction relative to the current model, or that the current residual/candidate evaluation is consistent with the way the existing `earth` forward pass calculates candidate improvements. Inspect the actual implementation before using this formula directly.

---

# Critical issue: correlated candidate basis functions

This is one of the most important implementation questions.

MARS candidate basis functions are generally not orthogonal to existing terms.

Therefore, a raw coefficient is not a clean measure of independent effect.

The candidate's incremental contribution should be evaluated **conditional on the existing model**.

A useful way to formulate this is to residualize the candidate basis function against the current model matrix:

\[
B_j^\perp
=
(I-P_X)B_j,
\]

where \(X\) is the existing basis matrix.

Then the candidate's incremental RSS reduction is determined by the component of the candidate that is genuinely new relative to the existing model.

This may already be implicit in the linear algebra used by `earth`; inspect the package before introducing a second residualization implementation.

The effect cap should be based on the same conditional candidate improvement that the existing forward pass uses.

---

# Do not assume the existing GCV penalty maps directly to a coefficient cap

The existing MARS `penalty` parameter represents a complexity cost associated with the MARS model/basis search. It is not inherently a coefficient-magnitude penalty.

Therefore:

- do not simply multiply the coefficient by `penalty`;
- do not reinterpret the existing `penalty` argument without understanding the current implementation;
- do not silently change ordinary `earth` behavior.

Instead, derive how the existing GCV changes when a candidate term is admitted and determine how that can define a candidate-specific maximum useful effect.

A separate experimental parameter may eventually be appropriate, but ideally the first prototype should minimize arbitrary new tuning parameters.

---

# Strong hypothesis to test

The most interesting version is an **adaptive** cap rather than a fixed cap.

The hypothesis is that as the model becomes more complex, the amount of additional effect that a new term can justify should change automatically.

Conceptually:

\[
\boxed{
\text{maximum effect}
=
f(
n,
RSS,
C,
\text{MARS penalty},
\text{candidate complexity},
\text{current model}
)
}
\]

rather than:

\[
\Delta R^2_{\max}=constant.
\]

This should produce progressively stronger regularization as the model consumes complexity.

---

# Possible implementation strategies

Investigate these in order.

## Strategy 1: Exact GCV boundary

For each candidate, construct the GCV as a function of its coefficient:

\[
GCV_j(b).
\]

Determine the coefficient boundary where the marginal benefit of increasing \(b\) is no longer justified.

This is the preferred conceptual implementation if it can be integrated cleanly with the existing forward pass.

## Strategy 2: Incremental GCV budget

Calculate the incremental complexity cost of adding the candidate.

Convert that into an allowable RSS/R² improvement and solve analytically for the maximum coefficient.

This may be easier to integrate into existing `earth` code.

## Strategy 3: Numerical one-dimensional optimization

If the exact analytical relationship is awkward because of interactions with the existing model, use a one-dimensional search over:

\[
0\le t\le1
\]

where

\[
b=t\hat b.
\]

This is slower but useful as a reference implementation for validating an analytical implementation.

---

# Important behavior to verify

The implementation should answer these questions explicitly:

1. What happens when the candidate's OLS effect is below the cap?
   - Expected: no change from ordinary MARS.

2. What happens when the candidate's OLS effect exceeds the cap?
   - Expected: candidate enters at the boundary, rather than being rejected.

3. After a candidate is capped, can another candidate enter and explain residual signal?
   - Expected: yes.

4. Does the capped candidate remain fixed?
   - This needs an explicit design decision. Initial prototype: treat it as saturated during the relevant forward-search stage.

5. What happens when later terms are correlated with the capped term?
   - Must be tested carefully.

6. What happens with interaction terms?
   - Do not assume the same cap interpretation automatically applies to interactions.

7. What happens when the candidate's maximum possible effect is already below the cap?
   - It should behave like ordinary forward selection.

8. Does the method preserve standard `earth` results when the feature is disabled?
   - Absolutely required.

---

# Suggested experimental API

Do not commit to an API until the implementation is understood.

Possible experimental naming:

```r
earth(..., adaptive.gcv = FALSE)
```

or perhaps:

```r
earth(..., effect.cap = ...)
```

The default should preserve current behavior.

If a new tuning parameter is required, prefer an explicit name that describes the concept rather than overloading `penalty`.

For example:

```r
adaptive.gcv = TRUE
```

could enable the method, while the ordinary `penalty` continues to mean what it currently means.

---

# Validation plan

Build tests before attempting to optimize the implementation.

## Synthetic test 1: Two independent predictors

Generate data such as:

\[
y=f_1(x_1)+f_2(x_2)+\epsilon
\]

where both predictors have comparable explanatory power.

Compare:

- ordinary `earth`,
- adaptive-cap `earth`.

Check whether the adaptive version gives the second predictor more opportunity to enter.

## Synthetic test 2: Dominant predictor

Generate:

\[
y=10f_1(x_1)+f_2(x_2)+\epsilon.
\]

Check whether the cap prevents \(x_1\) from consuming essentially all available explanatory power.

## Synthetic test 3: Pure noise

Generate:

\[
y=\epsilon.
\]

Verify that the adaptive procedure does not create spurious large effects.

## Synthetic test 4: Correlated predictors

Generate strongly correlated \(x_1,x_2\).

Verify that the effect measure is conditional on the existing model and does not behave pathologically.

## Synthetic test 5: Scale invariance

Scale a predictor:

```text
x
10*x
1000*x
```

The fitted function and adaptive behavior should be essentially invariant.

This is a critical test.

## Synthetic test 6: Interactions

Test:

\[
y=f(x_1,x_2)+g(x_3).
\]

Verify that the cap does not create unexpected behavior for interaction basis functions.

---

# Diagnostics to expose during development

For each forward step, it would be useful to be able to inspect:

```text
candidate
unconstrained coefficient
unconstrained ΔRSS
unconstrained ΔR²
adaptive cap
capped coefficient
actual ΔRSS
actual ΔR²
current model complexity
current GCV
```

A debug mode or internal diagnostic structure would make it much easier to determine whether the algorithm is behaving as intended.

---

# Important conceptual distinction

The goal is **not**:

> Make every term contribute roughly equally.

The goal is:

> Prevent a term from continuing to grow when the additional predictive improvement is no longer justified by the complexity budget, thereby allowing other terms to compete.

A dominant predictor should still be allowed to dominate if the data genuinely justify it.

The adaptive cap should merely prevent the forward search from giving that predictor unlimited explanatory priority solely because it happened to be the strongest candidate at an early step.

---

# Expected relationship to boosting

There is a useful analogy to boosting:

\[
\hat y_{k+1}
=
\hat y_k+\alpha_k h_k(x).
\]

The proposed method similarly builds the model incrementally, but uses MARS basis functions and an adaptive GCV-derived limit on how much each newly admitted component may contribute.

The conceptual difference is:

```text
ordinary MARS:
find strongest basis → allow full least-squares effect

adaptive MARS:
find strongest basis → allow justified effect → saturate → find another basis
```

This analogy may be useful when reasoning about the behavior, but do not implement this as boosting.

---

# Recommended development approach

1. Inspect the current `earth` forward-pass implementation and identify exactly where candidate basis functions are scored and admitted.
2. Identify the existing code that calculates RSS/GCV/effective complexity.
3. Add instrumentation without changing behavior.
4. Implement a simple candidate-specific effect cap using incremental ΔRSS as a prototype.
5. Validate against hand-calculated synthetic examples.
6. Compare the analytical coefficient-bound calculation against numerical one-dimensional searches.
7. Add correlated-basis tests.
8. Only after the mechanics are correct, derive the most principled adaptive GCV formula.
9. Keep the feature disabled by default until its statistical behavior is understood.

---

# Open mathematical question

The key unresolved question is:

> **Exactly how should the existing MARS GCV complexity cost be converted into a maximum allowable incremental effect for an individual candidate during the forward pass?**

Do not assume an answer.

Derive it from the actual `earth` implementation and the original MARS GCV formulation.

The desired final rule should ideally have this property:

\[
\boxed{
\text{candidate coefficient grows until marginal predictive benefit}
=
\text{marginal GCV complexity cost}
}
\]

and then:

\[
\boxed{
\text{candidate is saturated and the forward search continues.}
}
\]

If a clean derivation is impossible, implement the simplest defensible approximation and document exactly what approximation was made.

---

# Bottom line for the coding agent

Implement and investigate **Adaptive GCV Effect Capping in the MARS forward pass**.

The essential behavior is:

> **Let a candidate term enter normally, but constrain its predictive contribution to the amount justified by the current GCV/complexity tradeoff. If the unconstrained OLS coefficient would exceed that effect, add the term at the cap rather than rejecting it. Then continue the forward search so other terms can explain remaining signal.**

Do not turn this into a backward-pass pruning method.

Do not simply bound raw coefficients.

Do not assume `earth`'s existing `penalty` parameter is itself a coefficient penalty.

First understand and reuse the existing `earth` forward-pass candidate scoring, RSS, GCV, and complexity calculations.

The ultimate test of the idea is whether it produces a principled and stable way to distribute explanatory opportunity among candidate terms while preserving ordinary MARS behavior when the new feature is disabled.
