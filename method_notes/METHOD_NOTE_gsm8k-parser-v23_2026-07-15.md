# METHOD NOTE — GSM8K parser v2.3 (dated deviation, 2026-07-15)

**Status:** ratified. Countersigned by Saurav Bhandari (email, 2026-07-15, 4:08 PM,
subject "My reddit acc."). Exact-rule boundary ratified by Ben, 2026-07-15.
This note must exist BEFORE any analysis is run on the re-scored (`*_v23rescore_*`)
files. It does not modify the frozen pre-registration; it records a measurement
correction, per Saurav's framing.

## 1. What changed

`gsm8k_parse_answer` (in `steerquant_phase0_harness.py`) gains a fourth,
**last-resort** extraction step that fires ONLY after the three existing steps —
`####`, `\boxed{}`, and the prose `answer is|:` fallback (v2 / v2.1 / v2.2) — all
fail to yield a parseable number. Precedence is unchanged:

    #### > \boxed{} > prose "answer is|:" > v2.3 terminal-line

The v2.3 step accepts a bare number **standing alone on the final non-empty line**
of the trace. Tolerated on that line: leading/trailing markdown emphasis
(`*` `_` `` ` ``), one bracket/angle wrapper (e.g. `<14>`), an optional leading
`$`, sign, internal commas/decimals, and trailing punctuation. A number **embedded
in a prose sentence does NOT fire**.

Implementation: `_GSM8K_TERMINAL_LINE_RE`, matched against the last non-empty line
only (the scan stops at the first non-empty line from the end).

## 2. Why

The 2026-07-13 band-smoke work established that Llama-3.1 (and, less often,
Mistral) frequently terminate a correct GSM8K trace with the final answer as a
bare number on its own line — e.g. `…over 30 days.\n\n$75.00` or
`…was Raymond's son born.\n\n14` — with no `####`, `\boxed{}`, or explicit
"the answer is" statement. Under v2.2 these parsed to `None` and were scored
degenerate/incorrect, even though the generation terminated (EOS) with the
correct value. This is an **instrument artifact** (a parser miss), not a genuine
capability failure. v2.3 reads those terminal answers so the score reflects the
model's actual output.

## 3. Boundary decision (the part that needed a human call)

A naive "grab the last number that ends the trace" rule collides with an existing,
deliberately-degenerate selftest case:

    "Actual profit = -$96,666.67\nBut, Josh made a profit of $0."  ->  None

That trace ends in `$0.`; a loose rule would resurrect it to `0`, contradicting a
case the pre-registration intentionally keeps degenerate (a rambling trace that
never states a formal answer). The **standalone-final-line** boundary resolves the
conflict: `$75.00` alone on the last line fires; `$0` embedded mid-sentence does
not. The consequence is that a trace like `…count of 8 years.` also stays `None`,
but that changes no score — it is a wrong answer either way and never produces a
flip. Every real correct-answer flip is caught; the deliberately-degenerate case
is preserved.

## 4. Scoring is unchanged

    correct = EOS_emitted AND (parse(text) == gold)

The EOS flag (`capability_eos_flags`, the immutable generation fact) still gates
correctness. A NON-TERMINATING trace stays failed regardless of what v2.3 parses.
v2.3 never resurrects a degenerate/non-terminating generation; it only reads a
well-formed terminal answer line from a trace that already terminated. Gold answers
are parsed by the same function (unchanged: GSM8K gold ends `#### N`, caught by the
`####` step before v2.3 is reached).

## 5. Verification

- `python run_offline_checks.py` → 17/17 ALL GREEN (2026-07-15), including the
  gsm8k selftest with the new v2.3 terminal-line cases and the preserved
  `$0.`-degenerate guard.
- Selftest cases added: `…$75.00` → 75, `…\n14` → 14, `<14>` → 14,
  `**1,024**` → 1024, `…8 years.` → None, degenerate loop → None.

## 6. Re-score provenance

- New generation runs stamp `meta.gsm8k_parser = "v2.3"`.
- Original result files in `results\` are **untouched**. The offline re-score
  (`rescore_v23_fliplist.py`, 2026-07-15) reads stored `capability_texts` /
  `capability_eos_flags` and re-parses with v2.3; it writes NO result files and
  recomputes no pooled quantity. Its gold-reconstruction guard re-verifies every
  originally-correct item and fails closed on any mismatch.
- Q3 (applying the re-score to sibling `*_COMPLETE_v23rescore_<date>.json` files)
  is GATED on Saurav's approval of the flip list. It has NOT run.

## 7. Flip list sent for human review

- `rescore_v23_fliplist.py` over the 45 confirmatory files produced
  **1,644 flips** (all direction old→new; the guard confirmed no old-correct item
  regressed). Delivered to Saurav 2026-07-15 as
  `SteerQuant_v23_fliplist_2026-07-15.json` (+ `.csv`), reviewed in
  `SteerQuant_flip_reviewer.html`. Awaiting his per-flip / per-file rulings before
  Gate C (Q3) opens.

## 8. Related open item (Saurav, 2026-07-15 email)

Saurav also requested a **sensitivity analysis** comparing E*/arm results with and
without degenerate alphas as a robustness check, to be reported as a limitation if
they diverge. Logged for the queue; not part of this parser deviation.
