# SteerQuant — Analysis note: Saurav's three requested analyses
*2026-08-03, Cowork session. Data: the 60-cell length matrix (`length_matrix_long.csv`)
and all 105 per-cell steering vectors (`results/*_vector_*.npy`). Scripts:
`censor_sweep.py` (this session, committed) extending `estar_crossings.py`;
cosine computation run against the raw .npy files. All numbers below are
descriptive; nothing here is a confirmatory claim.*

## 1. Steering-vector cosine similarity across schemes

Vectors are extracted per scheme per resample and stored unit-normalized (norms
carry no information; coefficient inflation lives in alpha, not vector scale).
Cosines computed within model × target × resample, mean (sd) over the 5 resamples:

| model | target | fp16·int8 | fp16·nf4 | int8·nf4 |
|---|---|---|---|---|
| Qwen2.5-7B | length | 0.997 (.000) | 0.983 (.001) | 0.980 (.002) |
| Qwen2.5-7B | sentiment | 0.997 (.001) | 0.987 (.001) | 0.984 (.002) |
| Llama-3.1-8B | length | 0.992 (.001) | 0.962 (.003) | 0.956 (.003) |
| Llama-3.1-8B | sentiment | 0.989 (.002) | 0.956 (.004) | 0.945 (.005) |
| Mistral-7B-v0.3 | length | 0.995 (.001) | 0.975 (.002) | 0.970 (.002) |
| Mistral-7B-v0.3 | sentiment | 0.996 (.001) | 0.981 (.001) | 0.978 (.001) |
| Gemma-2-9B | length | 0.998 (.000) | 0.990 (.000) | 0.988 (.001) |

Readings: (a) the steering **direction survives weight-only quantization** —
int8 vectors are near-identical to fp16 (0.989–0.998) in every cell; (b) nf4
diverges more, consistently (0.945–0.990), with the ordering
fp16·int8 > fp16·nf4 > int8·nf4 in all 21 rows; (c) model-dependence is real:
Llama's directions are the most quantization-sensitive, Gemma's the least —
matching Llama's status as the costliest model in the E* analyses. This is the
protocol's "vector geometric fidelity" measure (H4 lineage), now computed.
No Gemma sentiment (not collected).

## 2. E* sensitivity to the failure-censoring threshold

Sweep: recompute the E* machinery dropping alphas whose length-failure rate
exceeds thr. fp16 ladder acceptance (of 5 resamples) and fp16 f=0.5 crossing
mean (sd) / IECC:

| thr | Qwen | Llama | Mistral | Gemma |
|---|---|---|---|---|
| as-coded (none) | 0/5 · α*52.8 · IECC .67 | 0/5 · 11.9 · .84 | 1/5 · 4.2 · .36 | 2/5 · 262 · .79 |
| all-fail only | 4/5 · 30.5 (sd 18) · .15 | 2/5 · 7.4 (3.4) · .51 | 3/5 · 2.5 (1.2) · .16 | 4/5 · 137 (48) · .25 |
| 25% | 5/5 · 16.0 (8.0) · .006 | 4/5 · 3.8 (3.1) · .15 | 3/5 · 2.2 (0.9) · .12 | 4/5 · 111 (35) · .13 |
| 10% | 5/5 · 16.0 (8.0) · .006 | 4/5 · 3.8 (3.1) · .15 | 3/5 · 2.2 (0.9) · .12 | 5/5 · 101 (19) · .02 |
| 5% | 5/5 · 12.7 (2.1) · .006 | 3/5 · 3.9 (3.1) · .29 | 3/5 · 2.2 (0.9) · .12 | 5/5 · 99.5 (20) · .02 |

Readings: (a) acceptance recovers **monotonically** as censoring tightens —
the cliff was the problem; (b) crossings stabilize (Qwen sd 18 → 2.1); (c) with
clean censoring the fp16 story becomes: steering length is **nearly free at E***
for Qwen (+0.006) and Gemma (+0.02), moderate for Mistral (+0.12), and
intrinsically expensive for Llama (+0.15–0.29, acceptance never 5/5) — real
model heterogeneity, not artifact. Caveats: 10% vs 25% identical for three
models because the grids have no alphas with failure in (10%, 25%] on the
relevant segments (grid discreteness, worth one sentence in methods); the
~10-prompt efficacy resolution makes per-alpha failure rates coarse (steps of
0.1). **Recommended primary rule: thr = 10%, with 5% and 25% reported as
sensitivity.**

## 3. Does contamination predict measured cost?

Across the 50 cells with an f=0.5 crossing (censor-all-fail convention),
correlation between the worst bracketing-alpha failure rate and measured IECC:
**Spearman ρ = 0.395, Pearson r = 0.664.** Split: mean IECC **+0.080** where
bracket failure < 5% (n=30) vs **+0.529** where ≥ 30% (n=17). Contamination is
not noise around the true cost — it inflates it, which is the direct empirical
justification for the guard rules and for reporting the contaminated zone
separately.

## Corrections of record (vs Saurav's 2026-08 email)

1. "Clean crossings α=6 (Llama) / α=2 (Mistral)" are not outputs of our
   analysis. Under the recommended 10% rule: **Llama fp16 α* = 3.8 (sd 3.1),
   Mistral fp16 α* = 2.1 (sd 0.9)**; per-scheme values differ and all are
   convention-dependent. (The −6/−2.5 figures circulating earlier were
   negative-arm length-failure survival edges, a different quantity.)
2. Qwen and Gemma **are** affected under the as-coded rule (crossings on the
   collapse cliff at ~52 and ~280; acceptance 0/5 and 2/5). They recover under
   censoring; "unaffected" is not the right description. Clean windows of
   record: Qwen ±40 (failures from |α|≥60), Gemma ±200 (failures from |α|≥250).
3. Guard-rule wording: E* remains an **efficacy** level. Proposed canonical
   wording: "E* levels are anchored on the maximum surviving effect of the
   fp16 efficacy curve restricted to alphas with length-failure ≤ 10%;
   crossings are computed on each scheme's own restricted curve; alphas
   exceeding the threshold constitute the contaminated zone and are reported
   separately with per-alpha failure rates." This is a dated prereg-§7
   deviation requiring joint sign-off.
