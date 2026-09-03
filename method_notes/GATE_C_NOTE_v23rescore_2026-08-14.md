# GATE C — v2.3.1 offline rescore of the sentiment matrix (2026-08-14)

**Method.** All 45 sentiment COMPLETE files re-scored offline from their stored
audit trail (capability_texts + capability_eos_flags), per the harness rule:
correct = EOS AND v2.3.1_parse(text) == gold. Gold reconstruction per
rescore_v23_fliplist.py's convention (mmlu_item_indices mod the 200-item test
prefix). Parser copied verbatim from steerquant_phase0_harness.py (v2.3.1).
Siblings written as *_COMPLETE_v23rescore_*.json (bundle committed alongside).

**Validation (both fail-closed checks pass).**
1. Fliplist reproduction: this rescore's flip set matches the 2026-07-15
   reference fliplist EXACTLY — 1,644 flips, 0 missing, 0 extra.
2. Gold guard: every item the original run scored correct re-verifies under
   v2.3.1 (no downgrades; v2.3 strictly rescues, never revokes).
3. Pipeline check: Option C on the ORIGINAL files reproduces the 2026-07-14
   first-look numbers to the fourth decimal.

**Option C, original vs rescored (90% CI):**

| cell | original | rescored (v2.3.1) |
|---|---|---|
| Qwen fp16 IECC | +0.030 [-0.006,+0.066] | unchanged |
| Qwen int8 / nf4 contrast | -0.011 / +0.014 | unchanged |
| Mistral fp16 IECC | +0.050 [-0.005,+0.104] | unchanged |
| Mistral int8 contrast | -0.024 [-0.070,+0.022] | unchanged |
| Mistral nf4 contrast | -0.068 [-0.123,-0.014] | -0.080 [-0.135,-0.024] |
| Llama fp16 IECC | +0.077 [+0.009,+0.145], runs 2-5 | **+0.019 [-0.034,+0.072], runs 1-5** |
| Llama int8 contrast | +0.016 [-0.084,+0.117] | +0.012 [-0.039,+0.063] |
| Llama nf4 contrast | -0.040 [-0.123,+0.043] | +0.012 [-0.055,+0.079] |
| **Pooled int8** | -0.013 [-0.030,+0.005] Inconclusive | **-0.010 [-0.026,+0.007] Equivalent** |
| **Pooled nf4** | -0.028 [-0.076,+0.020] Inconclusive | -0.017 [-0.067,+0.033] Inconclusive |

**Reading.** The parser correction moves numbers in exactly the direction its
method note predicted: Qwen (format-obedient) untouched; Llama (the ~50%
format-disobedient model) substantially repaired — its fp16 IECC falls from
+0.077 to +0.019 because the old parser was scoring correct-but-marker-less
answers as failures, and its r1 exclusion REVERSES (the ladder rejection was a
parser artifact, not an instrument failure; all 5 runs now pool). The
descriptive three-label for the pooled int8 contrast improves from
Inconclusive (missed by 0.0002) to **Equivalent** (still k=3, still
descriptive, still not the K=12 confirmatory claim). No conclusion weakens;
two strengthen.

**Decision needed (Ben + Saurav) before the Results tables lock:** adopt the
rescored numbers as primary (with the 1,644-flip audit + this note as App. C,
and Saurav's flip-review sign-off on record), or keep originals primary with
the rescore as a robustness appendix. Recommendation: rescored-primary — the
parser fix is a documented, preregistration-consistent correction, and the
audit trail (texts + EOS flags) exists precisely so this rescore replaces GPU
re-runs. If adopted: Table 2 / Figure F3 / §4.1 prose need the number swap
(one session's work, scripts ready).
