# METHOD NOTE — Censored E* rule for floor-bounded (length) targets
*2026-08-03. STATUS: ratified by Ben this date; Saurav countersign PENDING (his
2026-08 email agreed in substance; this note fixes the canonical wording his
Methods section should cite). This is a dated deviation to the application of
prereg §7 for the reasoning-length target only. The ladder fractions
(0.5→0.4→0.3), DELTA_MIN=0.10, the S11 arm/level discipline, and the sentiment
pathway are all unchanged.*

## 1. Problem

The §7 E* rule anchors its reference levels on the curve "extreme," defined as
the efficacy value with maximum |e − baseline| on the selected arm. The length
target's efficacy (survivor-median generated tokens) has a hard floor: at
alphas where all generations terminate degenerately, the harness records
efficacy = 0. On grids that reach termination collapse — all four models —
the floor competes with, and usually beats, the true behavioral peak:

- Qwen, Llama, Gemma: extreme = the collapse floor (0), placing all three
  ladder levels BELOW baseline on the lengthening arm; the "crossing" becomes
  the interpolated point on the collapse cliff between the last surviving and
  first all-fail alpha.
- Mistral: extreme = the cap-runaway peak (survivor medians inflated by 30–40%
  failure), a different anchor artifact in the same rule.

Consequences, measured over all 60 cells (`estar_crossings.py`, 2026-08-02):
fp16 ladder acceptance 0/5 (Qwen), 0/5 (Llama), 1/5 (Mistral), 2/5 (Gemma);
every f=0.5 crossing bracketed by at least one alpha at 100% length failure
(58/58 crossing-bearing resamples). Contamination is not benign: across cells,
the worst bracketing-alpha failure rate correlates with measured IECC
(Spearman ρ = 0.395, Pearson r = 0.664, n = 50), with mean IECC +0.080 at
clean brackets (<5% failure) vs +0.529 at contaminated brackets (≥30%).

## 2. Rule (canonical wording)

For targets whose efficacy metric has a degenerate floor (v1: reasoning
length only):

1. **Censoring.** The E* efficacy curve uses the survivor-median efficacy at
   each alpha, EXCLUDING alphas whose per-alpha length-failure rate exceeds
   **0.10** (failure flags from `termination_failure_detector`). Excluded
   alphas take no part in level derivation or crossing search.
2. **Levels.** E* levels = baseline + frac × (extreme − baseline),
   frac ∈ {0.5, 0.4, 0.3}, where extreme is the maximum-|shift| efficacy on
   the RESTRICTED fp16 curve (the maximum surviving effect). Arm and levels
   are fixed from the same-resample sibling fp16 cell (S11); each scheme's
   crossing is found on its own restricted curve.
3. **Acceptance.** The §7 ladder and DELTA_MIN=0.10 apply unchanged, computed
   on the restricted curves. Non-accepting resamples are excluded from the
   run-level combine with the exclusion reported (existing Option C practice).
4. **Capability reads.** Capability values (GSM8K accuracy over all N=200
   items) are NOT censored — accuracy at a high-failure alpha is a valid
   accuracy. Capability at α* is linear interpolation over the probed alphas.
   **Each cell must report Δprobe, the distance from α* to the nearest probed
   alpha, in grid steps.** Cells with Δprobe > 2 grid steps are
   resolution-limited and their IECC is reported as such. (This bites: probe
   windows were placed online under the pre-censoring rule, so Qwen and Gemma
   crossings sit ~2–4 steps from the nearest probe; Llama and Mistral sit
   ≤1.5 steps.)
5. **Contaminated zone.** Alphas excluded under (1) are reported separately —
   per-alpha failure rate alongside survivor-median length — as the
   "contaminated zone," with the two failure modes named (negative arm:
   immediate-EOS empties; positive arm: cap-runaway then collapse).
6. **Sensitivity.** Thresholds 0.05 and 0.25 are reported alongside 0.10.
   Note: 0.25 and 0.10 coincide for Qwen/Llama/Mistral because their grids
   contain no alphas with failure rates inside (0.10, 0.25] on the relevant
   segments — a grid-discreteness fact, stated in Methods. At the ~10-prompt
   efficacy resolution, per-alpha failure rates step in units of 0.1; the
   0.10 threshold therefore means "at most one failed prompt."

## 3. Effect of the rule (fp16, accepted-resample means, thr = 0.10)

| model | accept | α* (sd) | IECC (sd) | Δprobe |
|---|---|---|---|---|
| Qwen2.5-7B | 5/5 | 16.0 (8.0) | +0.006 (.010) | 2.3 steps |
| Llama-3.1-8B | 4/5 | 2.8 (2.5) | +0.021 (.037) | 1.4 |
| Mistral-7B-v0.3 | 3/5 | 1.5 (0.9) | +0.023 (.032) | 1.2 |
| Gemma-2-9B | 5/5 | 100.5 (19.1) | +0.021 (.038) | 3.2 |

Full per-scheme table: `final_table_thr10.json` / `final_table_thr10.py`.
Even under this rule, v1 reports length IECC descriptively (not as the
preregistered confirmatory endpoint): acceptance is not 5/5 everywhere, and
Δprobe exceeds 2 steps for two models. Confirmatory length IECC would require
a re-probed capability pass with windows placed under this rule — noted as
future work, not planned for v1.

## 4. What this rule does NOT change

Sentiment (and any judge-scored target): no failure floor, no censoring —
verified by the ladder accepting frac=0.5 in 14/15 sentiment cells with fp16
IECC +0.02…+0.08. The prereg's H1/H2 machinery, K-pool definitions, and the
sentiment Option C results are untouched.

## 5. Provenance

Cliff/anchoring diagnosis + censoring proposal: Cowork session 2026-08-02
(`estar_crossings.py`, both-convention analysis). Guard-rule structure
(threshold restriction, suspect-inflection flag, separate contaminated-zone
reporting): Saurav email 2026-08, harmonized here — his draft wording defined
the crossing on the capability metric; the canonical rule above keeps E* on
the EFFICACY curve (capability is read AT the crossing), preserving
comparability with sentiment and the prereg. Threshold sweep + contamination
correlation: `censor_sweep.py`, `ANALYSIS_NOTE_saurav-requests_2026-08-03.md`.
