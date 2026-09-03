# SteerQuant — Pre-Registration of Hypotheses & Analysis Plan (Re-Frozen)
*Registered: 2026-06-24. Re-frozen: 2026-06-29 with the power analysis resolved and the reasoning-length target folded in. Authors: Ben Wade; Saurav Bhandari.*
*Supersedes `SteerQuant_prereg_2026-06-24.md` and `SteerQuant_prereg_addendum_reasoning-length_2026-06-29.md` (both retained for provenance). Status: FROZEN before the confirmatory matrix. All prior [DECIDE] blanks are now resolved (rules locked); the only pilot-dependent steps are `max_new_tokens` fine-tuning and an H3 power sanity check, neither of which involves researcher discretion.*

*Update 2026-07-02 (pre-data; no confirmatory runs have executed): Saurav's final H1
power-analysis decisions received and folded in. N=200, δ=3%, and K=16 + heterogeneity gate
are **confirmed as already written** (§2A/§6/§7). Two decision-rule refinements applied:
§H1 rewritten to Saurav's **three-label outcome** (Equivalent / Meaningfully Worse /
Inconclusive), and §H2 rewritten to his **Rule A** (95% CI lower bound must exceed Δ_min).
These refine decision rules only — hypotheses, endpoints, and margins are unchanged. Still
outstanding from Saurav's side: sign-off on the refusal-drop → K-pool change (§2A;
currently his K=16 assumes refusal is retained).*

*Update 2026-07-11 (pre-data; no confirmatory runs have executed): Saurav's pool decision
(email 2026-07-03) is now APPLIED. **Refusal is dropped from the v1 pool.** The primary H1
pool changes from K=16 to **K=12 = 4 models × 3 judged targets {sentiment, sycophancy,
truthfulness}** (§2A/§4/§6/§9). Reasoning-length (§8) remains secondary-confirmatory and sits
OUTSIDE the K=12 judged pool; the §2A heterogeneity gate is unchanged. Saurav confirmed K=12
is "still sufficient" (the √(16/12)≈1.15× half-width inflation to ~0.021–0.025 stays under
δ=0.03 at low τ). This RESOLVES the refusal-drop → K-pool item left outstanding in the
2026-07-02 update; no open K-pool question remains.*

*Update 2026-07-10 (pre-data; no confirmatory runs have executed): the **primary capability
probe** is changed from single-forward-pass MMLU (log-prob MC) to a **generation-consistent
GSM8K** probe, forced by a validity failure found in the fp16 pilot (the single-pass MC probe is
dose-blind under the prereg-locked `site=last`; see `SteerQuant_capability_probe_decision_2026-07-07.md`).
The model generates a chain-of-thought and a final answer under `site=last`; capability =
exact-match accuracy on the parsed final number. N=200/condition, δ=3%, Δ_min=10%, and the E\*
rule are all UNCHANGED — only the measurement instrument changes. Single-pass MMLU is retained
as a SECONDARY probe; the generative-vs-single-pass contrast is reported as a methods result.*

> This document is **confirmatory**. Anything not specified here, or added after data
> are seen, is **exploratory** and labeled as such in the paper. Companion docs:
> `SteerQuant_protocol.md` (design + §4A statistics), `SteerQuant_background.md` (citations).
> Phase 0 de-risk (2026-06-24, Qwen2.5-7B FP16) is complete and is NOT part of the
> confirmatory set.

---

## 0. One-line claim under test

Quantization degrades activation steering by an amount that depends on the *compression
type* (weight-only vs weight+activation) and *algorithm*, and the degradation tracks
damage to outlier / massive-activation channels.

**Locked framing sentence (Saurav-approved 2026-06-26; for abstract/intro):** "Because
modern quantization methods already attempt to preserve salient weights, activation
outliers, or high-impact channels, it is not obvious whether activation steering should fail
under practical quantization. SteerQuant measures that directly." (The protection is tuned
for accuracy/perplexity, not steering; steering rides the same channels, so whether
accuracy-driven protection also preserves steering is the empirical crux — this answers the
"isn't weight-only preservation obvious?" objection.)

---

## 1. Primary endpoint (defined once, used everywhere)

**Iso-effect capability cost (IECC).** For a given (model × scheme × method × target):

1. Define a fixed **reference behavioral effect** `E*` — a target steering efficacy inside
   the reliable band.
2. On each scheme, find the coefficient `α*` whose steering efficacy = `E*` (interpolate on
   the dose–response curve; §4 item 5).
3. **IECC = capability(baseline α=0) − capability(α*)** on the held-out benchmark.

IECC is read **at matched behavioral effect, never at matched coefficient.**

**Primary capability probe (generation-consistent; changed 2026-07-10, pre-data — supersedes the
earlier "MMLU primary" designation).** GSM8K, generative. Under greedy decoding and `site=last`,
the model produces a chain-of-thought followed by a final answer; capability = **exact-match
accuracy** on the parsed final number. Answer extraction: the last `#### <number>` if present, else
the last `\boxed{...}`. Numbers are normalized (strip `$`, thousands separators, and surrounding
whitespace) before comparison to the GSM8K gold answer. This makes the capability dose
**accumulate across generated tokens**, matching the behavioral efficacy measure (the reason the
single-pass MC probe failed). **Secondary capability probe:** single-pass MMLU (log-prob MC),
retained unchanged as a robustness/contrast reference (see the generative-vs-single-pass
dissociation, § methods). ARC remains a secondary reference.

Secondary endpoints (pre-registered): vector geometric fidelity (cosine FP16-vs-quant
vector), coefficient inflation (α* ratio vs FP16), dose–response curve parameters (Xu et al.
2026 log-odds form).

---

## 2. Hypotheses, operationalization, and decision rules

Each test uses the §4A machinery: per-model estimate with a within-model 95% CI from
data-level resampling, then **random-effects aggregation** reporting the pooled effect, its
95% CI, and heterogeneity τ². Pooling structure and its decision rules are fixed in §2A
(this is the resolution of the 2026-06-24 power caveat).

### H1 — Weight-only quantization PRESERVES steering (equivalence claim)
- **Operationalization:** IECC(W8A16), IECC(W4A16) ≈ IECC(FP16).
- **EQUIVALENCE test (TOST)** against margin **δ = 3% MMLU accuracy**, on the contrast
  (IECC_weightonly − IECC_FP16).
- **Inference is on the pooled K=12 estimate** per §2A (single-cell equivalence is
  unachievable at realistic N — see §6).
- **Three-way outcome (Saurav, 2026-07-02; replaces the earlier binary support/disconfirm
  rule — "more honest than binary"):**
  - **Equivalent** — the pooled 90% CI on the contrast lies **entirely within ±δ** (both
    one-sided tests of the TOST reject). This is the H1-supporting outcome.
  - **Meaningfully Worse** — the pooled 90% CI lies **entirely outside the ±δ region on the
    degradation side** (the contrast is bounded away from equivalence, weight-only costing
    more than FP16 by more than δ). This disconfirms H1.
  - **Inconclusive** — the CI **straddles a ±δ boundary** (neither equivalence nor a bounded
    meaningful difference is established). Reported as inconclusive at the achieved N; **not**
    counted as support for H1.

### H2 — Activation quantization DEGRADES steering (difference claim)
- **Operationalization:** IECC(W8A8), especially IECC(W4A4) > IECC(FP16).
- **Test (Rule A; Saurav, 2026-07-02 — supersedes the earlier "pooled 95% CI excludes 0,
  with Δ_min = 10%" wording, which was looser and ambiguous):** one-sided difference on the
  contrast (IECC_actquant − IECC_FP16); support = the **lower bound of the pooled 95% CI
  exceeds Δ_min = 10% MMLU accuracy** — i.e. the *entire* 95% CI sits above the 10%
  meaningful-degradation line, in the degradation direction. (Deliberate 3–10% gray zone is
  neither "preserved" nor "meaningfully degraded.")
- **Disconfirmation:** the 95% CI lower bound does **not** clear 10% (the CI dips into the
  gray zone or includes 0), or W4A4 ≈ FP16.

### H3 — Degradation TRACKS outlier-channel damage (mechanism)
- **Operationalization:** outlier-distortion metric `D_outlier` = **MSE over the top-1%
  activation channels (by magnitude), pre- vs post-quantization**, correlates positively
  with IECC degradation across (scheme × model) cells.
- **Test:** mixed-effects correlation / slope; support = slope 95% CI excludes 0, positive,
  AND **R² ≥ 0.25** (|r| ≥ 0.5, Cohen "large"; fixed now by convention, not pilot-dependent).
  Bit-width pre-registered as a control covariate. **Pilot use:** a power sanity check only
  (with ~16–24 cells, confirm r=0.5 is detectable); the threshold itself is locked.
- **Disconfirmation:** no correlation, or degradation explained better by bit-width alone.

### H4 — ALGORITHM matters at equal bit-width (outlier-aware > naive)
- **Operationalization:** at equal bits, IECC(AWQ/SmoothQuant/QuaRot) < IECC(naive RTN).
- **Test:** paired (within model, equal bit-width) difference; support = pooled 95% CI of
  (IECC_RTN − IECC_outlieraware) excludes 0, positive.
- **Disconfirmation:** no algorithm effect once bit-width is fixed.

### H5 — REGIME matters (where the vector is built)
- **Operationalization:** IECC/efficacy differ between (a) FP16-extracted → quantized-applied
  (deployment-realistic) and (b) quantized-extracted → quantized-applied (on-device).
- **Test:** two-sided paired difference. **Direction not pre-committed.** Support = pooled
  95% CI excludes 0.
- **Disconfirmation:** regimes statistically indistinguishable.

---

## 2A. Pooling structure + heterogeneity decision rules (resolves the power caveat)

The H1 power analysis (Saurav, 2026-06-29) shows single-cell equivalence power is ~0 at
realistic N (0.00 at N=200/400/800; 0.23 at N=1600). **Pooling is required, not optional.**
The K=16 configuration achieved a pooled CI half-width (~0.018–0.022) under δ=0.03; dropping
refusal to **K=12** inflates this by √(16/12)≈1.15× to ~0.021–0.025, still under δ=0.03 at low
τ (Saurav, 2026-07-03, "still sufficient"). Target-specific pooling (K=4/K=3) roughly doubles
the half-width and exceeds the margin.

**Primary analysis = pooled across all 12 cells (4 models × 3 judged targets {sentiment,
sycophancy, truthfulness}), K=12, conditional on the heterogeneity gate below.** Refusal is
not part of the v1 pool (dropped 2026-07-03, Saurav sign-off); reasoning-length is analyzed
as secondary-confirmatory (§8) and is not part of the K=12 judged pool. We report, regardless
of outcome:
`τ_model` (between-model SD), `τ_target` (between-target SD), and `I²`. The pooled estimate
is reported **if and only if** the gate passes:

- `I² < 50%` AND `τ_model ≤ 0.058` AND `τ_target ≤ 0.058` → **report pooled K=12.**
- `τ_model > 0.058`, `τ_target ≤ 0.058` → **per-model pooled estimates** (no cross-model pool).
- `τ_target > 0.058`, `τ_model ≤ 0.058` → **per-target pooled estimates** (no cross-target pool).
- both exceeded → **report all 12 cells individually, no pooled inference.**

Fallback branches are underpowered for equivalence and are reported as **descriptive
(point estimate + CI), not confirmatory.** The `τ` threshold (0.058) is the H1 breakeven
value with a qualitative literature sanity check only; it is **not** described as
literature-derived anywhere in the paper.

---

## 3. Confirmatory vs exploratory

- **Primary confirmatory (safety-net paper):** H1, H2 on weight-only + W8A8. Stand alone;
  do not depend on W4A4 kernels.
- **Secondary confirmatory:** H3, H4, H5; W4A4 schemes; H1/H2 on the reasoning-length
  target (§8).
- **Exploratory (labeled):** dose–response *shape* changes; per-target idiosyncrasies;
  SAE-feature steering as a 3rd method; the length/behavioral **dissociation** comparison;
  native-reasoning-model (delimited thinking block) length steering; any post-hoc subgroup.

---

## 4. Design locked for the confirmatory set

- **Models (≥4 families):** Llama-3-8B, Mistral-7B, Qwen2.5-7B, + one size jump
  (Llama-3-70B-4bit, cloud); add a 4th 7–8B family if the size jump slips.
- **Schemes:** FP16 baseline; W8A16, W4A16 (GPTQ, AWQ); W8A8 (SmoothQuant);
  W4A4 (QuaRot/Atom/SpinQuant). Per protocol §3.
- **Methods:** CAA (primary), mean-difference/ActAdd. (SAE = exploratory.)
- **Judged targets:** sentiment/persona, sycophancy, truthfulness (TruthfulQA). *(Refusal
  removed from the v1 pool 2026-07-03, Saurav sign-off; retained as a possible v2/appendix target.)*
- **Judge-free target:** reasoning-trace length (§8).
- **Steering layer:** floor(0.5 × N_layers), pilot-verified → Llama-3-8B 16, Mistral-7B 16,
  Qwen2.5-7B 14, Llama-3-70B 40.
- **Steering mechanism (uniform across ALL targets -- locked 2026-06-30, pre-data).** Every target, judged and judge-free alike, uses the identical intervention mechanism: (i) **CAA / mean-difference vector** built from contrast-pair residuals extracted in the model's **assistant role** (the exemplars are activations of the model *generating* the contrasted text, not reading it as a user turn); (ii) added to the residual stream at **site = last** (ActAdd-style: applied only at the final position of each forward pass, steering the generation frontier and leaving the prompt encoding intact during prefill); (iii) at the per-model layer **floor(0.5 x N_layers)** fixed above. The stimulus content differs per target by necessity (sentiment pairs vs. verbose/terse reasoning pairs), but the extraction, injection site, and layer rule do **not** vary by target. Fixed a priori for the whole confirmatory matrix.
- **Replication:** **5 data-level resampled runs per cell** (resample contrast pairs +
  bootstrap eval/benchmark items). No RNG-seed replication (greedy decoding).

**Why uniform mechanism matters (positioning; added 2026-06-30).** The reasoning-length target uses a different *stimulus* and a judge-free *metric* from the four judged targets, which invites the objection that it is "a different method." We neutralize this in three layers: (1) the judge-free token metric is a **strength**, not a liability -- it is the objective anchor against which the judged targets' judge-validity is calibrated; (2) a differing stimulus is trivially necessary -- you cannot elicit length with sentiment pairs; (3) the steering **mechanism** -- the one thing that could confound a cross-target comparison -- is held identical across all targets by the uniform-mechanism lock. Because IECC, H1, and H2 are evaluated **within each target** (the method is fixed across quantization schemes inside a target), any residual cross-target method difference cannot contaminate the primary claims; it bears only on the secondary cross-target generality story. The site=last + assistant-role choices were fixed as documented **pre-data** decisions (validation run 2026-06-30) and are prereg'd here before the confirmatory matrix.

## 5. Inference rules (fixed)

- Headline numbers with **95% CIs**; equivalence claims (H1) with **90% CIs via TOST**.
- **Holm–Bonferroni** across the confirmatory hypothesis family.
- Aggregation = **random-effects meta-analysis** per §2A; always report **τ² / I²** and a
  per-scheme forest plot. A claim "generalizes" only if no model flips sign.
- **Fixed extraction/eval splits across all schemes; no per-scheme tuning.**
- **Real behavioral judge** replaces the Phase-0 lexical scorer before any confirmatory
  efficacy number (judged targets only; the length target needs no judge).

## 6. Sample size + power (resolved)

- **Capability benchmark N = 200 items per condition** (generation-consistent GSM8K primary per §1, changed 2026-07-10; single-pass MMLU + ARC secondary).
- **H2 contrast (10% gap):** adequately powered at N=200.
- **H1 equivalence:** single-cell is unachievable at realistic N; the binding analysis is
  the **pooled K=12** estimate (§2A), where the pooled CI half-width stays under δ=0.03 (refusal
  dropped 2026-07-03; the √(16/12)≈1.15× inflation to ~0.021–0.025 stays under δ at low τ —
  reasoning-length sits outside the K=12 pool, so its tighter SEs strengthen the §8 secondary
  line and §9 anchors, not the pooled half-width).
- **δ=3% justification:** above MMLU test-retest noise (~1–2%), under the ~5% quant
  "lossless" convention; observed paired contrast SEs ~0.015–0.021 support it.
- **Generality** is pinned by the **number of model families (≥4)** and observed **τ²**, not
  per-cell run count.

## 7. Confirmed blank values (re-frozen)

- `E*` (reference behavioral effect): **50% of the per-model maximum behavioral shift**
  (pilot steering curve). **Acceptance rule (locked, auto-evaluated on pilot):** E* is in the
  reliable band iff the **FP16** capability cost at E* (clean-model IECC) is **< Δ_min =
  10%**; if not, step the fraction down 0.5 → 0.4 → 0.3 until satisfied. (Rationale: the
  reference effect must not, even on the undamaged model, already cost a "meaningfully
  degrading" amount of capability — this keeps E* inside the smooth regime, off the collapse
  cliff, with no researcher discretion. "Healthy" ≠ zero loss, since IECC is nonzero by
  design; it means below the meaningful-degradation line on FP16.)
- `δ` (H1 equivalence margin): **3% MMLU accuracy.** Confirmed.
- `Δ_min` (H2 minimum meaningful degradation): **10% MMLU accuracy.** Confirmed.
- `D_outlier` (H3): **MSE over the top-1% activation channels, pre- vs post-quant.** Confirmed.
  Correlation floor: **R² ≥ 0.25 (|r| ≥ 0.5, Cohen "large"). Locked.**
- Per-model steering layer: fixed (§4). Confirmed.
- Capability benchmark N: **200/condition.** Confirmed. **(2026-07-10, pre-data: the primary
  capability instrument changed from single-pass MMLU to generation-consistent GSM8K — §1; N,
  δ=3%, Δ_min=10%, the E\* ladder, K pool, and the heterogeneity gate are all UNCHANGED. The
  measured baseline SE at N=200 is already tight, so no increase is needed.)**

- **Capability scoring & failure rule (GSM8K primary; added 2026-07-10, pre-data).** A GSM8K trace
  is scored **correct** iff it terminates (emits EOS before the token cap) AND its parsed final
  number equals the gold answer. A trace that is **non-terminating** (no EOS by the cap) or
  **degenerate** (no parseable `#### <number>` / `\boxed{}` answer) is scored **incorrect**. The
  **failure rate** (fraction of non-terminating + degenerate traces) is reported **separately** as a
  co-primary diagnostic, exactly as the reasoning-length target reports its termination-failure rate
  (§8). Token cap for the capability generation: **512 new tokens (confirmed).** In the fp16 N=200
  adjudicator (2026-07-10) the failure rate is ~0.01 across the entire usable band and saturates to
  ~1.0 only at the extreme alphas where deep steering genuinely breaks generation, so 512 does not
  truncate legitimate CoT.

- **Compute economy (added 2026-07-10, pre-data; run-time α-grid choice, not a threshold change).**
  The capability probe is evaluated at **baseline (α=0) plus the E\*-relevant arm** rather than the
  full symmetric α grid. The fp16 cell establishes the gradient and α*; the reduced α set for the
  remaining scheme cells is scoped from it. Efficacy continues to sweep the full grid so α* stays
  well-resolved.

## 8. Reasoning-length target (judge-free; secondary confirmatory)

- **Construct / vector:** propensity to produce a longer vs shorter reasoning trace; CAA
  vector from prompt-matched verbose-vs-terse contrast pairs; same layer/method as other
  CAA targets. Fixed CoT-eliciting template applied identically across schemes and α values.
- **Efficacy metric (judge-free):** median **generated-token count** (model tokenizer),
  **computed only over non-failure traces** (termination-failure rule below). `Δ_len(α) = tokens(α) − tokens(α=0)`. Tokens, not
  lines (lines = readability sidecar only). Deterministic under greedy decoding; same
  5-resample error bars as every cell.
- **IECC integration (no new statistics):** `E*_len = 50% of per-model max length shift`;
  find α* hitting E*_len per scheme; IECC = capability(α=0) − capability(α*). H1/H2 apply at
  the same δ=3% / Δ_min=10%.
- **Termination-failure rule (deterministic, content-blind — not a judge):** a generation
  is a **failure** if ANY of: (a) it does not emit EOS before `max_new_tokens`
  (non-terminating / ran to the cap); (b) **structural loop** — an n-gram
  repeats *consecutively* at or above a length-graduated threshold (n≤2 words: ≥5 repeats;
  n=3–5: ≥3; n≥6: ≥2; plus a sub-word char-level check), the literature-grounded
  deterministic signature of self-reinforcing degeneration (Xu et al. 2022). **Implemented
  by the existing, tested `Tools/degenerate_repeat_detector.py` (detection only — we use its
  loop flag; we do NOT use its truncate-and-recover output, per the no-recovery rule).**
  (c) **rep-4** (fraction of tokens in
  any repeated 4-gram, the standard seq-rep-n metric of Welleck et al. 2019 / Holtzman et
  al. 2019) exceeds the operational cutoff **0.5**. A non-terminating run has no
  well-defined length; it is **excluded from the length measurement and counted in the
  failure rate.** **No loop collapse, no length recovery** — a failure is a failure even if
  it contains some genuine reasoning. The detector is a fixed function (same input → same
  output, no semantics inspected), so it does not reintroduce a judge. Cost is negligible
  (CPU token post-processing on already-generated traces; no GPU/model call).
  - **n=4 is the conventional seq-rep-n order** (Welleck et al. 2019); the structural
    criterion (b) follows loop-detection work (Xu et al. 2022).
  - **The 0.5 cutoff is a pre-registered OPERATIONAL value, NOT literature-derived** — the
    field reports rep-n continuously and has no canonical discard threshold. Accordingly,
    report the length analysis at rep-4 ∈ {0.4, 0.5, 0.6} so nothing hinges on the line
    (robustness, exploratory). The structural criterion (b) is the primary, threshold-free
    failure signal; rep-4 (c) catches diffuse non-consecutive repetition it would miss.
- **Failure-rate decision rule (handles non-random censoring):** failures are not random —
  steering toward verbosity will tend to raise the failure rate and the failures are
  disproportionately the longest traces, so dropping them silently would bias surviving
  length downward. Therefore: a length comparison between two cells is **interpreted only
  when their failure rates are comparable**; a material failure-rate difference is reported
  as a **degradation finding in its own right**, not absorbed into the length estimate.
- **Confound controls (mandatory):**
  2. **Iso-effect only:** raw length never compared across schemes; α* solved per-scheme to
     hit the same E*_len, so baseline-verbosity shifts cancel by construction (length-domain
     difference-in-differences).
  3. **`max_new_tokens` (sets the failure boundary):** set generously so *legitimate* long
     traces get to terminate — anything still going at the cap is a failure (rule above), not
     a counted short measurement, so the cap defines the non-termination line rather than
     silently clipping the distribution. **Pre-pilot default = 2,048.** Tuning rule: set to
     ~1.5× the 99th-percentile length of *terminating* traces at maximum α from the pilot;
     **raise to 4,096 if that 99th pct exceeds ~1,300.** A cap so low that legitimate traces
     are forced into failure inflates the failure rate artificially, so the failure rate is
     monitored against the cap during the pilot.
- **Status:** H1/H2 on this target = secondary confirmatory. Dissociation comparison and
  native-reasoning-model version = exploratory (§3).

## 9. Single-cell deep anchor (validates the pool, judge-free)

Because the length target needs no judge, pre-register **one** cell run to high N to obtain
an *individually-powered* equivalence result that defends the K=12 pool against the
"pooling hides a failing cell" attack:

**Two anchors** are pre-registered (judge-free, ~5,000 evals total, no judge cost):

- **Anchor 1 — hardest cell (a priori stress test):** `(model = Mistral-7B, scheme = W4A16,
  method = CAA, target = reasoning-length)`. W4A16 is the most aggressive *weight-only*
  scheme, so it is the weight-only cell most likely to break equivalence; if it passes, that
  is the strongest single-cell defense. **"Hardest model" criterion (fixed a priori):** the
  smallest model in the 7–8B band by parameter count (Mistral-7B, ~7.2B) — least parameter
  redundancy under weight-only compression. This is a transparent a priori proxy, not a
  proven hardness ordering (massive-activation severity could instead make a larger model
  hardest); the random Anchor 2 covers us if the proxy is wrong. Criterion fixed BEFORE any
  confirmatory data.
- **Anchor 2 — random cell:** drawn uniformly from the *remaining* (model × scheme × target)
  cells, **numpy default_rng(seed=20260629)**, to defuse the cherry-picking objection.
- Each run to **N ≈ 2,500 eval items**, single-cell TOST at δ=3% (~80% power per the H1
  analysis).
- These complement, do not replace, the §2A pooled primary.

---

### Open [DECIDE] items before the runs they govern
1. *(resolved — rule locked)* `E*` acceptance is auto-evaluated on the pilot (FP16 IECC at
   E* < Δ_min=10%, else step fraction down); no discretion remains.
2. *(resolved — locked)* H3 floor = R² ≥ 0.25 (Cohen large). Pilot supplies only a power
   sanity check (is r=0.5 detectable with our cell count), not the threshold.
3. `max_new_tokens` pilot-tuning only — termination-failure rule (rep-4 > 0.5, no EOS) and
   failure-rate co-endpoint now fixed.
4. *(resolved)* Deep-anchor hardest model = Mistral-7B (smallest in 7–8B band); random draw
   = default_rng(20260629). Two-anchor rule and `max_new_tokens` default fixed.

*All other values are immutable. Any later change is a documented, dated deviation reported
in the paper.*
