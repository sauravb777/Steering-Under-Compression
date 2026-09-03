#!/usr/bin/env python3
"""
Per-prompt PAIRED re-scorer for the reasoning-length target.
===========================================================
Why: the harness stores length efficacy as the MEDIAN of the pooled per-prompt
token counts. With ~10 prompts whose baseline lengths span ~90-410 tokens, that
median is dominated by *which prompts are in the pool*, not by the steering
effect -- it can hide a real dose-response (see 2026-06-30 length diagnostics).

This script recomputes a sensitive estimator with NO GPU run: for each prompt it
takes length(alpha) - length(alpha=0), so each prompt is its own control and the
huge between-prompt variance cancels. It averages those deltas across the prompts
that TERMINATED at BOTH that alpha and at alpha=0 (a clean paired set), and also
reports the failure rate separately. A real "+alpha => longer" direction shows a
monotone positive mean-delta on the positive arm; "-alpha => shorter" shows
negative deltas on the negative arm.

Usage:
    python rescore_length_paired.py results\\SteerQuant_phase0_fp16_2_COMPLETE_20260630.json
    python rescore_length_paired.py results\\*.json      # several files
"""
import sys
import glob
import json
import statistics as st
from pathlib import Path


def load_alpha_table(path):
    """Return (meta, list of dicts: alpha, lengths[], flags[])."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = []
    for rec in data.get("by_alpha", []):
        lengths = rec.get("length_tokens_per_prompt")
        flags = rec.get("failure_flags")
        if lengths is None:
            # judged target (no length data) -- skip
            continue
        if flags is None:
            flags = [0] * len(lengths)
        rows.append({"alpha": rec["alpha"],
                     "lengths": [int(x) for x in lengths],
                     "flags": [int(f) for f in flags]})
    return data.get("meta", {}), rows


def baseline_row(rows):
    for r in rows:
        if abs(r["alpha"]) < 1e-9:
            return r
    return None


def rescore(path):
    meta, rows = load_alpha_table(path)
    if not rows:
        print(f"  [skip] {Path(path).name}: no per-prompt length data "
              f"(target={meta.get('target')!r}).")
        return
    base = baseline_row(rows)
    if base is None:
        print(f"  [skip] {Path(path).name}: no alpha=0 baseline to pair against.")
        return

    print("=" * 72)
    print(f"  {Path(path).name}")
    print(f"  model={meta.get('model')}  scheme={meta.get('scheme')}  "
          f"layer={meta.get('layer')}  target={meta.get('target')}")
    n_prompts = len(base["lengths"])
    print(f"  {n_prompts} prompts; paired vs alpha=0 (only prompts terminating "
          f"at BOTH alphas counted)")
    print("-" * 72)
    print(f"  {'alpha':>7} | {'mean dL':>8} {'median dL':>9} | {'n_pair':>6} | "
          f"{'fail_rate':>9} | {'surv.med.len':>12}")
    print("-" * 72)
    for r in rows:
        deltas = []
        for i in range(n_prompts):
            # paired: both must have terminated (flag 0) to compare lengths
            if base["flags"][i] == 0 and r["flags"][i] == 0:
                deltas.append(r["lengths"][i] - base["lengths"][i])
        mean_d = st.mean(deltas) if deltas else float("nan")
        med_d = st.median(deltas) if deltas else float("nan")
        survivors = [L for L, f in zip(r["lengths"], r["flags"]) if f == 0]
        surv_med = st.median(survivors) if survivors else 0
        fail_rate = sum(r["flags"]) / len(r["flags"]) if r["flags"] else 0.0
        print(f"  {r['alpha']:>7.1f} | {mean_d:>8.1f} {med_d:>9.1f} | "
              f"{len(deltas):>6} | {fail_rate:>9.2f} | {surv_med:>12.1f}")
    print("-" * 72)
    print("  Read: a working '+alpha => longer' direction gives monotone POSITIVE")
    print("  mean dL on the positive arm and NEGATIVE on the negative arm. Flat /")
    print("  sign-inverted / noisy dL => the contrast vector is not a length axis.")
    print()


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        sys.exit(1)
    paths = []
    for a in args:
        paths.extend(glob.glob(a))
    if not paths:
        print(f"  no files matched: {args}")
        sys.exit(1)
    for p in sorted(paths):
        rescore(p)


if __name__ == "__main__":
    main()
