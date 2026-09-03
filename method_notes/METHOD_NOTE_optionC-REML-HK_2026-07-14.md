# SteerQuant — Method note: Option C run-level aggregation = REML + modified Knapp–Hartung

*Dated 2026-07-14 (Ben + agent). Status: instantiation of Saurav's Option C ruling
(email 2026-07-12, S2B-07-12-26 11th response) — the two-level bootstrap for combining
the 5 physical resample runs per (model, scheme) into one confirmatory estimate + CI.
Written BEFORE the confirmatory H1/H2 numbers are produced from the 45-file sentiment
set. Mention this note in the next email to Saurav.*

## What Saurav ruled, and what it left open

Option C (his ruling): within-run variance from the analysis-internal bootstrap;
between-run variance from the spread of the per-run estimates; point estimate and CI
from the combination. The options doc pinned this down in one sentence
("SE² = mean(within-run bootstrap var) + between-run var; point = mean of runs"),
which leaves open (a) how the within-run component scales when 5 runs are averaged,
(b) how between-run variance is estimated from k=5 points without absorbing the
within-run noise twice, and (c) the reference distribution for the CI at 4 degrees
of freedom. These are resolved here, pre-numbers, as follows.

## The instantiation (Ben's specification, 2026-07-14)

Treat the 5 runs per (model, scheme) cell as a small **random-effects meta-analysis**:

- Unit: run r = 1..5. For the H1 quantity, y_r = the run's point contrast
  IECC_scheme − IECC_fp16, computed within the run triad (the triad shares
  `meta.resample` for run r, so the contrast is run-paired), with the arm sign and E*
  level fixed from that run's fp16 point estimate (S11) and the E* fraction from the
  prereg §7 acceptance ladder evaluated on that run's fp16 curve.
- v_r: the within-run variance of y_r from the S12 paired bootstrap
  (`bootstrap_contrast`, variance of the draws).
- Model: y_r ~ N(μ, v_r + τ²). **τ² estimated by REML** (iterative fixed-point;
  truncated at 0), NOT DerSimonian–Laird.
- CI: **Knapp–Hartung** — SE²(μ̂) = q̂ / Σw*, where w*_r = 1/(v_r + τ̂²) and
  q̂ = Σ w*_r (y_r − μ̂)² / (k−1); reference distribution t with k−1 = 4 df.
  **Modified** KH: SE²(μ̂) = max(q̂, 1) / Σw* — never anti-conservative when τ̂² ≈ 0.
- H1: three-label rule on the KH 90% CI (t₄,0.95 = 2.1318). H2 Rule A: KH one-sided
  95% lower bound (same t₄,0.95). Cross-model pooling above this layer is unchanged
  (prereg §2A DL pool + heterogeneity gate); the run-level μ̂ and SE(μ̂) are its inputs.

## Why this resolves the open points

1. **No double-counting under literal-§4 files.** The runs resampled pairs AND
   prompts AND items (Saurav: files stand as collected). Each run's v_r already
   carries the prompt/item noise, and REML estimates τ² as the between-run spread
   IN EXCESS of the v_r — so τ² captures the contrast-pair/vector variability and
   nothing twice. The additive one-sentence formula would have counted item noise in
   both components; the meta-analytic decomposition attributes each source once.
2. **Within-run scaling falls out of the weights.** Var(μ̂) ≈ 1/Σw* behaves as
   (v̄ + τ²)/k in the balanced case — the 1/5 shrinkage the literal sentence was
   silent on, derived rather than chosen.
3. **Small-k calibration.** REML at k=5 is the standard small-sample choice (unbiased
   in the balanced case; DL is not); KH with t₄ is the accepted fix for normal-theory
   CIs being anti-conservative at small k. The modified-KH max() guards the one known
   KH failure mode (CI narrower than the conventional one when q̂ < 1).

## Disclosure

- Post-data timing: run-level point estimates and per-run bootstrap CIs for the 15
  Qwen cells were seen on 2026-07-12 (interim look, disclosed to Saurav in the
  results packet sent that day). The Llama/Mistral per-run numbers have been glanced
  at only as pass/fail run summaries (0 failed). The REML/KH choice is made before
  any combined (run-level-pooled) number has been computed for ANY cell.
- The estimator choice (REML + modified KH, t₄) is frozen by this note. If it
  produces degenerate output (e.g., τ̂² non-convergence), the fallback is
  Paule–Mandel τ² + the same KH CI, and the switch gets its own dated note BEFORE
  the numbers are read.

## Implementation record

`steerquant_analysis.py` (2026-07-14): `reml_tau2()`, `hk_ci()` (modified KH),
`combine_runs()` run-level combiner; `bootstrap_contrast`/`bootstrap_iecc` now also
return the draw variance; `report(..., combine_runs=True)` / CLI `--combine-runs`
groups files by (model, target) → run triads → per-run contrasts → REML+KH per
scheme → H1 three-label / H2 Rule A on the KH CI. Selftest [18] covers: balanced-case
REML identity (τ̂² = S² − v̄ exactly), KH vs modified-KH behavior at τ̂² = 0, t₄
quantiles, and an end-to-end 2-scheme × 5-run synthetic combine. Runs under
`run_offline_checks.py` (analysis selftest; check count unchanged at 17).
