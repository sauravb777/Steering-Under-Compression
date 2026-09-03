# METHOD NOTE — GSM8K parser v2.3.1: non-finite number guard (crash fix)

**Date:** 2026-07-17 (during the Vast length-matrix collection)
**Type:** Implementation bug fix. NOT a scoring-rule change. No prereg quantity touched.
**Status:** Deployed to the local master and both Vast boxes mid-collection; disclose to Saurav with the length data.

## What happened

During the length matrix on Box B, cell `Mistral-7B-Instruct-v0-3_fp16_length_L16_r3`
crashed 2h48m in with:

```
File "steerquant_phase0_harness.py", line 639, in _gsm8k_normalize_number
OverflowError: cannot convert float infinity to integer
```

Cause: a degenerate steered GSM8K trace ended with a very long digit run
(hundreds of digits). The v2.3 terminal-line rule (correctly) captured it as a
candidate answer; `float()` of a several-hundred-digit literal returns
`inf` (no exception), and the normalizer's `int(f)` then raised
`OverflowError`, killing the entire cell. The same crash class is the likely
cause of Box A's earlier silent `Qwen ... fp16_length_L14_r1` failure
(verify via `grep OverflowError run_matrix_log_*` on Box A).

## The fix (v2.3.1)

`_gsm8k_normalize_number` gains a non-finite guard between `float()` and the
int-collapse:

```python
if f != f or f in (float("inf"), float("-inf")):
    return None
```

A non-finite parse is treated as **unparseable → None → trace scored
incorrect and flagged as a failure** — exactly the parser's documented
contract ("Returns None if not numeric") and the standing failure semantics
for degenerate traces. Parser precedence and every v2/v2.1/v2.2/v2.3 rule
are unchanged.

`meta.gsm8k_parser` stamp bumped `"v2.3"` → `"v2.3.1"`. Two selftest cases
added (digit-blob terminal line; huge capture inside `####`). Verified: all
prior selftest cases still pass; a valid `####` answer still takes precedence
over a trailing digit blob.

## Why this changes no collected score

The old code **crashed** on any trace in this class — it never emitted a
wrong score. Therefore no COMPLETE file anywhere can contain a trace this fix
re-scores: files collected under v2.3 and v2.3.1 are score-identical on their
shared domain. The stamp difference across cells of this matrix
(v2.3 on cells completed before 2026-07-17, v2.3.1 after) documents the code
version only; it implies no measurement heterogeneity.

## Deployment

- Local master `steerquant_phase0_harness.py`: patched 2026-07-17.
- Box A (45130430) and Box B (45128865): patched in place mid-collection
  (each cell launches a fresh interpreter, so the next cell picks it up; the
  in-flight cell is unaffected and, if it crashes on this class, is swept up
  by the standard resumable re-run).
- Crashed cells (`Mistral fp16 length r3`, `Qwen fp16 length r1` + its two
  quantized dependents) recover via the normal re-run of the same launch
  command; without this fix their crashes were deterministic (greedy
  decoding), so re-running the old code could never complete them.
