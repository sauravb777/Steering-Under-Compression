# SteerQuant — Method note: GSM8K parser v2 + per-model alpha grids
*Dated 2026-07-13. PRE-DATA for Llama-3.1-8B-Instruct and Mistral-7B-Instruct-v0.3
(zero confirmatory cells collected for either). Saurav APPROVED (relayed by Ben,
2026-07-13). Evidence: the 07-13 Vast band smokes
(`SteerQuant_bandsmoke_llama31_COMPLETE_20260713.json`,
`SteerQuant_bandsmoke_mistral_COMPLETE_20260713.json`,
`SteerQuant_diag_llama_cap1024_COMPLETE_20260713.json`, and five raw baseline
traces from `diag_llama_texts.py`). The Qwen 15-file sentiment matrix is
untouched by both changes.*

## 1. Findings that forced this note (both caught by required pre-launch smokes)

**(a) GSM8K format compliance is model-dependent; the strict parser scored
compliant-but-prose answers as failures.** At alpha=0 (NO steering), Llama-3.1
showed gsm8k=0.400/fail=0.60 and Mistral 0.600/fail=0.40 (Qwen baseline:
0.900/fail~0.01). Raw traces confirmed: `eos=True` on every item (termination
is fine; the 512 cap is fine — fail unchanged at cap 1024). The flags came from
`gsm8k_parse_answer` returning None on traces that stated a correct final
answer in prose ("The final answer is 540.") instead of the instructed
'#### <number>' form. Llama/Mistral obey the format instruction only ~50–60%
at baseline; Qwen obeys ~99%.

**(b) The alpha grid was Qwen-calibrated and does not transfer.** Residual
norms at the steering layer: Qwen L14 ~409; Llama-3.1 L16 ~13.0; Mistral L16
~20.6. Observed E* crossings: Qwen ~ −25; Llama ~ −7; **Mistral ~ −2.5 — below
the default grid's first step**, a hard FAIL of the band rule
(RUN_PLAN_vast-3model-expansion §3a).

## 2. Change 1 — GSM8K parser v2 (`steerquant_phase0_harness.py`)

Precedence: LAST `#### <number>` → LAST `\boxed{<number>}` → LAST prose
`answer is <number>` (case-insensitive; `$` tolerated before the number in all
three). An EXPLICIT answer statement is still required — no bare last-number
heuristic, so a rambling trace with no stated answer remains degenerate and
failure-flagged (the Saurav 2026-07-10 scoring rule is otherwise unchanged:
non-terminating or unparseable = incorrect + failure, reported separately).

Verified: 14/14 parser cases (8 original + 6 new, two lifted verbatim from the
07-13 Llama traces, including one that must STAY degenerate) — sandbox PASS;
the new cases are added to `--gsm8k-selftest`, which `run_offline_checks.py`
runs.

**v2.1 amendment (same day, 2026-07-13).** The Mistral v2check (5 items,
alpha=0) still showed fail=0.40; the STORED capability_texts (follow-up 1,
built hours earlier) diagnosed it without any GPU run: one failure was a THIRD
answer form — "Final answer: $18.00." (colon, no 'is'; the answer itself was
CORRECT) — and the other a genuine no-answer ramble (stays flagged). The prose
pattern is amended to accept 'answer is' OR 'answer:'; explicit answer
statement still required. 16/16 parser cases sandbox PASS (adds the verbatim
colon trace + an 'Answer: 42' case); meta stamp bumped to `gsm8k_parser:
v2.1`. Files carrying `v2` stamps are the 07-13 v2check DIAGNOSTICS only —
never pooled; zero confirmatory cells exist under v2. Covered by the same
Saurav approval (parser-broadening fix, relayed by Ben 2026-07-13); flag the
v2.1 detail in the next Saurav email for completeness.

**v2.2 amendment (same day, 2026-07-13).** The offline re-score of the stored
v2check texts caught a v2.1 REGRESSION before launch: Llama's item-0 trace
ends "The final answer is:\n18" (answer on the NEXT line) — parsed by v2's
`[:\s]*`, missed by v2.1's rewrite; Llama's re-score went 0.8/0.2 → 0.6/0.4,
which a strictly-broader parser cannot legitimately do. v2.2: (a) the prose
pattern allows colons/whitespace incl. newlines between statement and number;
(b) `gsm8k_parse_answer` skips matches whose capture does not normalize to a
number — last PARSEABLE match wins within each pattern's precedence tier — so
a junk capture can no longer mask an earlier valid answer. 18/18 parser cases
sandbox PASS (adds the verbatim item-0 trace + a junk-capture fallback case);
stamp `gsm8k_parser: v2.2`. Re-score acceptance for launch: BOTH v2checks at
0.8/0.2 under v2.2. Lesson recorded: every parser amendment must re-score ALL
stored texts before deployment — that check is what caught this.

**Qwen comparability:** the collected Qwen matrix was scored with parser v1.
Its in-band capability failure rate was ~1%, which bounds the maximum shift v2
could produce on Qwen at ~1 item in 100 — an order of magnitude below the
delta=3% margin. Recorded here rather than re-collected; capability texts are
not persisted, so re-scoring without re-running is impossible (see §4).

## 3. Change 2 — per-model alpha grids (`run_matrix.py MODEL_ALPHAS`)

`run_matrix.py` now carries a per-model confirmatory grid, used automatically
when `--alphas` is not passed (an explicit `--alphas` overrides for every cell,
diagnostics unchanged):

- Qwen2.5-7B: the existing DEFAULT_ALPHAS (unchanged — its matrix is collected).
- Llama-3.1-8B (crossing ~ −7): −40, −30, −20, −15, −12, −10, −8, −6, −4, −2,
  0, +mirror (21 points; step 2 through the crossing band).
- Mistral-7B-v0.3 (crossing ~ −2.5): −30, −20, −15, −10, −7, −5, −4, −3, −2,
  −1, 0, +mirror (21 points; step 1 through the crossing band).

Grids remain symmetric (prereg symmetric sweep), dense around each model's own
crossing, with coarse far points for the capability-collapse shoulder. The
adaptive capability-alpha selection and the analysis are grid-agnostic (they
operate on whatever alphas a file contains), so no other code changes.

Design caveat, accepted: the crossings come from subset-5 band smokes (noisy).
The fp16 spine runs the FULL grid at N=200, and E*/alpha* are recomputed
post-hoc from that, so the grid only needs to BRACKET the crossing with
density — it does. If a model's N=200 fp16 curve puts alpha* outside its dense
band, STOP and re-note before its quantized cells run.

## 4. Follow-ups this note creates (not blockers for the sentiment expansion)

1. **Persist capability generation TEXTS — DONE same day (2026-07-13, Ben's
   call, before any expansion cell).** `gsm8k_probe` now returns and the
   harness persists `capability_texts` (ALL items, raw generations) +
   `capability_eos_flags` per alpha, and meta stamps `gsm8k_parser: v2`.
   Any future parser/format issue is an offline re-score against stored text
   (correct = eos AND parse(text) == gold; failed = (not eos) OR unparseable)
   — never a GPU re-run. Cost: ~2-3MB per adaptive cell file.
2. Failure rate is now partly a FORMAT-COMPLIANCE measure at baseline for
   non-Qwen models even under v2 (genuinely answer-free rambles like the 07-13
   Llama item-2 trace still flag). Report per-model baseline failure rates in
   the paper; do not interpret cross-model failure-rate differences as pure
   capability differences.
3. The paper's cross-model dose comparison must use the S13 normalized
   alpha*/residual-norm ratio (already implemented) — raw alpha* is
   incomparable across these residual scales (13 vs 409).

## 5. Re-verification sequence (box)

1. Local: `python run_offline_checks.py` → 17/17 (gsm8k selftest now 13 cases).
2. Re-zip (`python make_upload_zip.py`) + re-upload to the box.
3. Per model, cheap baseline re-check (~2 min each, model cached):
   `python steerquant_phase0_harness.py --model <m> --layer 16 --subset 5 --alphas 0 --run-tag v2check_<m>`
   PASS: baseline fail ≤ ~0.2 and gsm8k plausible (Llama ~0.8; Mistral ~0.5).
4. Then the matrix per RUN_PLAN_vast-3model-expansion §3b (grids now automatic).
