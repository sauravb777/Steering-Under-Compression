#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
rescore_v23_fliplist.py  —  Q2 of the GSM8K parser v2.3 re-score (2026-07-15).

WHAT IT DOES
  Walks the 45 confirmatory COMPLETE files, re-parses every stored GSM8K
  capability trace with the v2.3 parser (harness gsm8k_parse_answer), and
  records every ITEM whose correct/incorrect status CHANGES (a "flip").
  Output is ONE flip file (JSON) in the schema the review app reads, plus a
  companion CSV. Saurav reviews the flips in SteerQuant_flip_reviewer.html.

WHAT IT DELIBERATELY DOES NOT DO  (per the execution-queue gate)
  * Writes NO result files, touches NO file in results\\.
  * Does NOT recompute IECC / H1 / any pooled number.
  * Does NOT summarise what the flips would do to the numbers. It prints only
    per-file flip COUNTS (allowed) so you can sanity-check volume.

SCORING (identical to the harness, method note 2026-07-13):
  correct = EOS_emitted AND (v2.3_parse(text) == gold)
  The EOS flag is the immutable generation fact (capability_eos_flags), so a
  NON-TERMINATING trace stays failed no matter what v2.3 parses — v2.3 only
  rescues traces that terminated with a well-formed but marker-less answer.

GOLD RECONSTRUCTION:
  meta.resample.mmlu_item_indices is the shared capability index vector. Gold
  for item j = gsm8k_parse_answer( gsm8k_test[ indices[j] % m ]["answer"] ),
  m = len(base slice). A GUARD verifies the mapping: for every item the ORIGINAL
  run scored correct, v2.3(text) must equal the reconstructed gold (v2.3 ⊇ v2.2,
  so an old-correct item must still match). Any mismatch aborts — a wrong gold
  map would silently mis-score, so we fail closed.

USAGE (from the project folder):
    python rescore_v23_fliplist.py
    python rescore_v23_fliplist.py --out SteerQuant_v23_fliplist_2026-07-15.json
"""

import argparse
import csv
import datetime as _dt
import json
import re
import sys
from pathlib import Path

# --- v2.3 parser: import the REAL one from the harness so this can never drift.
try:
    from steerquant_phase0_harness import gsm8k_parse_answer as parse_v23
except Exception as e:  # pragma: no cover - surfaced to the user, not swallowed
    print("FATAL: could not import gsm8k_parse_answer from steerquant_phase0_harness.py")
    print("       run this from the project folder. Error:", e)
    raise SystemExit(2)

# --- v2.2 parser (display only): shows Saurav what the OLD parser extracted.
#     Precedence #### > \boxed{} > prose 'answer is|:'. NO terminal-line rule.
_H = re.compile(r"####\s*\$?\s*([-+]?[\d.,]+)")
_B = re.compile(r"\\boxed\{\s*\$?\s*([-+]?[\d.,]+)\s*\}")
_P = re.compile(r"answer\s*(?:is|:)[\s:]*\$?\s*([-+]?[\d.,]+)", re.IGNORECASE)


def _norm(s):
    s = s.strip().replace(",", "").replace("$", "").rstrip(".")
    try:
        f = float(s)
    except ValueError:
        return None
    return str(int(f)) if f == int(f) else repr(f)


def parse_v22(text):
    for rx in (_H, _B, _P):
        for m in reversed(list(rx.finditer(text))):
            v = _norm(m.group(1))
            if v is not None:
                return v
    return None


# --- confirmatory file selection ------------------------------------------
CONFIRMATORY_RE = re.compile(
    r"^SteerQuant_.+_sentiment_L\d+_r\d+_COMPLETE_.+\.json$")
DIAG_PREFIXES = ("bandsmoke_", "diag_", "v2check_", "eqsmoke_")


def is_confirmatory(p: Path) -> bool:
    n = p.name
    if p.parent.name != "results":
        return False
    if n.startswith("SteerQuant_phase0_"):
        return False           # pilots do not pool
    if any(n.startswith(x) for x in DIAG_PREFIXES):
        return False           # diagnostics do not pool
    if "_v23rescore_" in n:
        return False           # never read a rescored sibling as an original
    return bool(CONFIRMATORY_RE.match(n))


def field(meta, *names, default=None):
    for k in names:
        if k in meta:
            return meta[k]
    return default


_GOLD_CACHE = {}


def gsm8k_golds(indices):
    """Golds for a resample index vector, using the harness's slice convention:
    base = gsm8k_test[:min(n, len)]; item j -> base[ indices[j] % len(base) ]."""
    from datasets import load_dataset
    key = "gsm8k_main_test"
    if key not in _GOLD_CACHE:
        ds = load_dataset("openai/gsm8k", "main", split="test")
        _GOLD_CACHE[key] = ds
    ds = _GOLD_CACHE[key]
    # base size m: the run resampled len(indices) items from a prefix slice of
    # size min(n_questions, len(ds)); n_questions == the per-alpha item count ==
    # len(indices) for these runs. Guard below catches any violation.
    m = len(indices)
    golds = []
    for idx in indices:
        row = ds[int(idx) % m]
        golds.append(parse_v23(row["answer"]))
    return golds


def main():
    ap = argparse.ArgumentParser(description="Generate the v2.3 re-score flip list.")
    ap.add_argument("--results", default="results",
                    help="results directory (default: results)")
    ap.add_argument("--out", default=None,
                    help="output JSON path (default: SteerQuant_v23_fliplist_<date>.json)")
    ap.add_argument("--tail-chars", type=int, default=900,
                    help="chars of trace tail to include per flip (default 900)")
    args = ap.parse_args()

    day = _dt.date.today().isoformat()
    out_json = Path(args.out) if args.out else Path(f"SteerQuant_v23_fliplist_{day}.json")
    out_csv = out_json.with_suffix(".csv")

    results_dir = Path(args.results)
    if not results_dir.is_dir():
        print(f"FATAL: results dir not found: {results_dir.resolve()}")
        raise SystemExit(2)

    files = sorted(p for p in results_dir.glob("*.json") if is_confirmatory(p))
    print(f"confirmatory files matched: {len(files)}")
    if len(files) != 45:
        print(f"  WARNING: expected 45 confirmatory files, found {len(files)}.")
        print("  Review the list below before sending anything to Saurav:")
        for p in files:
            print("   ", p.name)
        print("  (Continuing so you can inspect; do NOT send if the set is wrong.)")

    flips = []
    per_file_counts = {}
    guard_failures = []

    for p in files:
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"  SKIP (unreadable): {p.name}: {e}")
            continue
        meta = data.get("meta", {})
        model = field(meta, "model", default="?")
        scheme = field(meta, "scheme", default="?")
        run = field(meta, "resample", default={}) or {}
        run_no = run.get("run", meta.get("run", "?"))
        indices = (meta.get("resample", {}) or {}).get("mmlu_item_indices")
        n_flip_this = 0

        # per-alpha blocks live under the top-level "by_alpha" list.
        rows = data.get("by_alpha") or data.get("results") or data.get("curve") or []
        if not isinstance(rows, list):
            rows = []

        golds = gsm8k_golds(indices) if indices else None

        for block in rows:
            if not isinstance(block, dict):
                continue
            texts = block.get("capability_texts")
            items = block.get("capability_mmlu_items")
            eos = block.get("capability_eos_flags")
            if not (texts and items and eos):
                continue
            alpha = block.get("alpha")
            fail = block.get("capability_failure_flags") or [None] * len(texts)
            L = min(len(texts), len(items), len(eos))
            g = golds if (golds and len(golds) >= L) else None
            for j in range(L):
                text = texts[j]
                old = int(items[j])
                e = int(eos[j])
                gold = g[j] if g else None
                pred = parse_v23(text)
                new = 1 if (e == 1 and gold is not None and pred == gold) else 0

                # GUARD: an old-correct item must re-verify under v2.3.
                if old == 1 and g is not None and not (e == 1 and pred == gold):
                    guard_failures.append(
                        (p.name, alpha, j, gold, pred, e))

                if new != old:
                    n_flip_this += 1
                    flips.append({
                        "id": f"{p.stem}|a{alpha}|i{j}",
                        "file": str(p),
                        "model": model,
                        "scheme": scheme,
                        "run": f"r{run_no}",
                        "alpha": alpha,
                        "item_idx": j,
                        "gsm8k_index": (int(indices[j]) if indices and j < len(indices) else None),
                        "gold": gold,
                        "old_score": old,
                        "new_score": new,
                        "old_parsed_answer": parse_v22(text),
                        "parsed_answer": pred,
                        "eos": e,
                        "failure_flagged": bool(pred is None or e == 0),
                        "text_tail": text[-args.tail_chars:],
                    })
        per_file_counts[p.name] = n_flip_this
        print(f"  {p.name}: {n_flip_this} flip(s)")

    if guard_failures:
        print("\nFATAL: gold-reconstruction guard failed on old-correct items.")
        print("The index->gold mapping is wrong; NOT writing a flip file.")
        for row in guard_failures[:20]:
            print("   ", row)
        print(f"   ...{len(guard_failures)} total mismatches.")
        raise SystemExit(3)

    # sort flips for a sensible review order: model, scheme, run, alpha, item
    flips.sort(key=lambda f: (str(f["model"]), str(f["scheme"]),
                              str(f["run"]), (f["alpha"] if f["alpha"] is not None else 0),
                              f["item_idx"]))

    out = {
        "meta": {
            "generated": day,
            "parser_from": "v2.2",
            "parser_to": "v2.3",
            "n_files": len(files),
            "n_flips": len(flips),
            "note": ("GSM8K v2.3 terminal-line re-score. Flips are items whose "
                     "correct/incorrect status changed. Non-terminating traces "
                     "cannot flip to correct (EOS guard). Review each: does the "
                     "highlighted v2.3 answer legitimately match the gold?"),
            "scoring": "correct = eos AND v2.3_parse(text) == gold",
        },
        "flips": flips,
    }
    out_json.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    # companion CSV (flat, for spreadsheet eyes)
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "file", "model", "scheme", "run", "alpha", "item_idx",
                    "gsm8k_index", "gold", "old_score", "new_score",
                    "old_parsed_answer", "parsed_answer", "eos", "failure_flagged"])
        for f in flips:
            w.writerow([f["id"], Path(f["file"]).name, f["model"], f["scheme"],
                        f["run"], f["alpha"], f["item_idx"], f["gsm8k_index"],
                        f["gold"], f["old_score"], f["new_score"],
                        f["old_parsed_answer"], f["parsed_answer"], f["eos"],
                        f["failure_flagged"]])

    print("\n" + "-" * 64)
    print(f"total flips: {len(flips)}  across {len(files)} files")
    print(f"wrote: {out_json}")
    print(f"wrote: {out_csv}")
    print("Load the JSON into SteerQuant_flip_reviewer.html and send both to Saurav.")


if __name__ == "__main__":
    main()
