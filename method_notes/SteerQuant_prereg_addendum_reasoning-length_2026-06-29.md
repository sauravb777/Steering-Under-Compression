# SteerQuant — Pre-Registration Addendum: Reasoning-Length as a Steering Target
*Registered: 2026-06-29. Authors: Ben Wade; Saurav Bhandari. Status: pre-data (no confirmatory runs executed). Amends `SteerQuant_prereg_2026-06-24.md`; does not alter any existing target, hypothesis, or locked design element.*

> This is a **dated, documented deviation** from the 2026-06-24 prereg, made BEFORE the
> confirmatory matrix runs. It ADDS one steering target and one judge-free endpoint. It
> changes nothing about refusal / sycophancy / truthfulness / sentiment, the IECC primary
> endpoint, H1–H5, or §4–§5 inference rules. Companion: `SteerQuant_prereg_2026-06-24.md`,
> `SteerQuant_protocol.md`.

---

## A. What is being added and why

A fifth steering target — **reasoning-trace length (verbosity)** — whose efficacy is read
**purely quantitatively, with no behavioral judge**: the number of tokens the model emits.
Motivation: (i) it is a fully objective, deterministic, continuous endpoint, so it removes
the LLM-judge from one whole target — no judge compute, no judge noise; (ii) a continuous
low-variance metric gives much tighter CIs than a judge-scored binary, which makes a
single-cell, individually-powered equivalence test affordable (see §F); (iii) it adds a
second, orthogonal axis on which to ask the paper's core question — *does quantization
preserve steering at iso-effect?* — measured on something a judge cannot color.

Length-steering preservation and behavioral-steering preservation can dissociate. That
dissociation is informative but means length is a **complement, not a substitute** for the
judged targets. Status is set accordingly in §G.

---

## B. The target and its steering vector

- **Construct:** propensity to produce a longer vs shorter reasoning trace before answering.
- **Vector:** CAA (primary method, consistent with the locked design), built from contrast
  pairs that hold the prompt fixed and vary only reasoning length — a verbose/extended
  step-by-step completion vs a terse/minimal completion reaching the same answer. Built and
  applied exactly like the other CAA vectors. **Direction lock (added 2026-06-30, pre-confirmatory):** the signed length axis is defined as **terse - verbose**, i.e. **+alpha lengthens** the trace and -alpha shortens it. The raw authored contrast is verbose - terse, but the 2026-06-30 validation run (Qwen2.5-7B, fp16, layer 14, site=last, paired estimator) showed +alpha along that raw contrast SHORTENS (monotone, clean, both-arms-collapse absent). We fix the sign a priori by reversing the pair order in `SteerQuant_length_stimulus_2026-06-30.py`; this is a sign convention only -- axis, stimulus, layer, and site unchanged. Fixed before the confirmatory matrix. (Same layer per model, §4 of base prereg).
- **Uniform steering mechanism (added 2026-06-30):** this target uses the SAME mechanism as the judged targets -- assistant-role contrast extraction, injection at **site=last**, and the floor(0.5 x N_layers) layer -- per the uniform-mechanism lock in base prereg §4. Only the stimulus content and the judge-free metric differ; the mechanism does not.
- **Elicitation harness (fixed before runs):** a single CoT-eliciting prompt template
  ("reason step by step, then answer") applied identically across all schemes. The same
  template is used at α=0 and at every α, so the steering effect is measured against a fixed
  baseline elicitation, not against the model's default verbosity.

---

## C. Efficacy metric (judge-free)

- **Primary length metric:** median **generated-token count** of the reasoning trace
  (model's own tokenizer), computed over the eval set, **after a repetition filter** (§E).
- **Length effect:** `Δ_len(α) = tokens(α) − tokens(α=0)`, per (model × scheme × method).
- **Tokens, not lines.** Line counts depend on newline/formatting behavior, which
  quantization perturbs independently; that is added confound. Line count is reported only
  as a human-readability sidecar, never as the analyzed endpoint.
- Deterministic under greedy decoding; consistent with the base prereg's rejection of
  RNG-seed replication. Within-cell error bars come from the locked procedure — **5
  data-level resampled runs per cell** (resample contrast pairs + bootstrap eval items),
  identical to every other cell.

---

## D. Integration with the IECC primary endpoint (no new machinery)

This target plugs into the existing IECC frame unchanged:

1. Reference effect **`E*_len` = 50% of the per-model maximum length shift** (from the pilot
   steering curve) — parallel to the base prereg's `E* = 50% of max behavioral shift`,
   scale-invariant, inside the reliable band.
2. Find `α*` whose `Δ_len = E*_len` on each scheme (interpolate on the dose–response curve).
3. **IECC = capability(α=0) − capability(α*)** on MMLU, read at matched length-effect.

H1 (weight-only preserves) and H2 (activation quant degrades) therefore apply to the
reasoning-length target with **no change** to their TOST / difference machinery, margins
(`δ = 3%`), or `Δ_min` (`10%`).

---

## E. Confound controls (mandatory — this is where length steering can go wrong)

1. **Quantization-induced repetition.** Low-bit quantization is known to induce degenerate
   repetition loops, which would inflate token count and masquerade as length steering.
   **Mitigation:** before counting, apply a fixed n-gram repetition filter (collapse/truncate
   runs of repeated n-grams; parameters fixed here before runs). The repetition-filtered
   count is the analyzed metric. **Report the per-scheme repetition rate as a diagnostic;** a
   scheme whose "length" effect is driven by repetition is flagged, not scored as steering.
2. **Iso-effect cancels baseline length shift.** Quantization changes baseline verbosity on
   its own. We therefore **never compare raw length across schemes.** `α*` is solved
   per-scheme to hit the same `E*_len`, and only the iso-effect *capability cost* (IECC) and
   the per-scheme `α*` / efficacy are compared. This is the length-domain analog of the
   base prereg's iso-effect logic and removes the baseline-verbosity confound by
   construction. (Equivalently: the preservation question is a difference-in-differences —
   steered-minus-unsteered length delta within FP16 vs within each quantized scheme.)
3. **Truncation.** `max_new_tokens` is fixed generously so the cap does not censor the
   length distribution. **Report truncation rate per scheme;** differential truncation is a
   flagged confound, not silent.

---

## F. Role in the power / pooling plan

Because this target is judge-free, it is the natural **single-cell deep anchor** that
validates the pooled equivalence analysis without paying for judges:

- Pre-register **one** cell — `(model = ____, scheme = W4A16, method = CAA, target =
  reasoning-length)` — run to **N ≈ 2,500 eval items**, to obtain an *individually-powered*
  H1 equivalence result (single-cell TOST at `δ = 3%`, ~80% power per the H1 power analysis).
- This complements, and does not replace, the pooled K=16 primary analysis: it gives a
  reviewer a concrete, judge-free cell where single-cell equivalence holds on its own,
  directly answering "is the pooled result masking a failing cell?" for at least one cell.
- Cell selection (record before runs): **____** (randomized for bias, or the *a priori*
  hardest cell as a stress test — decide and record here).

---

## G. Confirmatory vs exploratory status

- **Secondary confirmatory:** H1 and H2 on the reasoning-length target (preservation /
  degradation of length steering at iso-effect), via the §D IECC machinery and §E controls.
- **Exploratory (labeled as such in the paper):**
  - the *dissociation* comparison — whether quantization preserves length-steering
    differently from behavioral steering;
  - any extension to native reasoning models with a delimited thinking block (the current
    matrix is non-reasoning models under CoT elicitation; a delimited-CoT variant, e.g. a
    QwQ/Qwen3-class model, is a future probe, not part of this confirmatory set).

---

## H. Blanks to fill before re-freeze

- `E*_len`: 50% of per-model max length shift — confirm the pilot curve places it inside the
  reliable band (MMLU still healthy at the corresponding `α`).
- n-gram repetition-filter parameters: **____** (fix before runs).
- `max_new_tokens`: **____** (set generously; record truncation-rate threshold for flagging).
- Deep-anchor cell + N: model **____**, N **____ (~2,500)**, selection rule **____**.

*Once these are filled and the pilot is run, fold this addendum's resolved values into the
re-frozen prereg. Any later change is itself a documented, dated deviation.*
