"""
summarize_length_matrix.py

Pull per-alpha length efficacy and gsm8k capability out of the SteerQuant
length COMPLETE result files and produce:

  1. length_matrix_long.csv   -- tidy long form, one row per (cell, alpha).
                                 Good for pivoting / re-analysis in R/pandas.
  2. length_cell_rollup.csv   -- one row per cell (model x scheme x resample)
                                 with baseline, peak, collapse and safe-window
                                 summaries.
  3. a printed fp16 vs int8 vs nf4 comparison table per model, averaged over
     the 5 resamples -- the quick "does quantization break steering" eyeball.

Only *_COMPLETE_*.json are read; PARTIAL files are ignored. Safe to re-run.

Run from the project root:
    python summarize_length_matrix.py
Options:
    --results FOLDER   where the COMPLETE json files live (default: results)
    --out FOLDER       where to write the CSVs           (default: .)

Schema notes (SteerQuant harness output):
    top level: meta{...}, by_alpha[ {...}, ... ]
    each by_alpha record:
        alpha, efficacy(=median generated tokens), failure_rate(length),
        and for probed alphas: capability_mmlu(=probe accuracy, e.g. gsm8k),
        capability_failure_rate.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

# Primary capability accuracy is stored under a legacy key name; try in order.
CAP_ACC_KEYS = ("capability_mmlu", "capability_gsm8k", "capability_acc", "capability_score")
CAP_FAIL_KEYS = ("capability_failure_rate",)

SCHEME_LABEL = {"fp16": "fp16", "w8a16_bnb_int8": "int8", "w4a16_bnb_nf4": "nf4"}
LABEL_ORDER = {"fp16": 0, "int8": 1, "nf4": 2}

SAFE_TOL = 0.05  # gsm8k counts as "safe" if within this of the alpha=0 baseline


def _first(d, keys):
    for k in keys:
        v = d.get(k)
        if v is not None:
            return v
    return None


def short_model(name):
    return name.split("/")[-1] if name else name


def is_baseline(alpha):
    return alpha is not None and abs(alpha) < 1e-9


def load_cell(path):
    """Return {'meta':..., 'rows':[per-alpha dicts]} or None if unreadable."""
    try:
        with path.open(encoding="utf-8") as fh:
            data = json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"  [skip] {path.name}: unreadable ({exc})")
        return None

    rows = []
    for rec in data.get("by_alpha", []):
        rows.append({
            "alpha": rec.get("alpha"),
            "length_median": rec.get("efficacy"),
            "length_fail": rec.get("failure_rate"),
            "gsm8k": _first(rec, CAP_ACC_KEYS),
            "gsm8k_fail": _first(rec, CAP_FAIL_KEYS),
        })
    rows.sort(key=lambda r: (r["alpha"] is None, r["alpha"]))
    return {"meta": data.get("meta", {}), "rows": rows}


def cell_id(meta):
    resample = meta.get("resample") or {}
    return {
        "model": short_model(meta.get("model", "")),
        "scheme": SCHEME_LABEL.get(meta.get("scheme", ""), meta.get("scheme", "")),
        "layer": meta.get("layer"),
        "resample": resample.get("run"),
        "residual_norm": meta.get("residual_norm_at_layer"),
        "parser": meta.get("gsm8k_parser"),
        "probe": meta.get("capability_probe"),
        "run_tag": meta.get("run_tag"),
    }


def baseline_value(rows, key):
    for r in rows:
        if is_baseline(r["alpha"]):
            return r[key]
    return None


def rollup(cid, rows):
    base_len = baseline_value(rows, "length_median")
    base_cap = baseline_value(rows, "gsm8k")

    lengths = [(r["length_median"], r["alpha"]) for r in rows if r["length_median"]]
    peak_len, peak_alpha = max(lengths) if lengths else (None, None)

    # first (smallest |alpha|) fully-degenerate cell (length failure_rate == 1.0)
    first_collapse = None
    for r in sorted(rows, key=lambda r: abs(r["alpha"]) if r["alpha"] is not None else 1e9):
        if r["alpha"] and r["length_fail"] == 1.0:
            first_collapse = r["alpha"]
            break

    # widest |alpha| where capability stays within tolerance of baseline
    gsm8k_safe = 0.0
    if base_cap is not None:
        for r in rows:
            if r["gsm8k"] is not None and r["gsm8k"] >= base_cap - SAFE_TOL:
                gsm8k_safe = max(gsm8k_safe, abs(r["alpha"]))

    out = dict(cid)
    out.update({
        "baseline_len": base_len,
        "baseline_gsm8k": base_cap,
        "peak_len": peak_len,
        "peak_len_alpha": peak_alpha,
        "first_collapse_abs_alpha": first_collapse,
        "gsm8k_safe_abs_alpha": gsm8k_safe,
    })
    return out


def mean(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    return round(statistics.mean(xs), 2) if xs else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default=".")
    ap.add_argument("--glob", default="SteerQuant_*_length_L*_r*_COMPLETE_*.json")
    args = ap.parse_args()

    results = Path(args.results)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    files = [f for f in sorted(results.glob(args.glob)) if "PARTIAL" not in f.name]
    if not files:
        print(f"No COMPLETE files in {results.resolve()} matching {args.glob}")
        return

    long_rows, cell_rows = [], []
    for f in files:
        cell = load_cell(f)
        if cell is None:
            continue
        cid = cell_id(cell["meta"])
        for r in cell["rows"]:
            long_rows.append({
                "model": cid["model"], "scheme": cid["scheme"], "layer": cid["layer"],
                "resample": cid["resample"], "alpha": r["alpha"],
                "length_median": r["length_median"], "length_fail": r["length_fail"],
                "gsm8k": r["gsm8k"], "gsm8k_fail": r["gsm8k_fail"],
                "residual_norm": cid["residual_norm"], "parser": cid["parser"],
            })
        cell_rows.append(rollup(cid, cell["rows"]))

    long_path = out / "length_matrix_long.csv"
    with long_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=[
            "model", "scheme", "layer", "resample", "alpha",
            "length_median", "length_fail", "gsm8k", "gsm8k_fail",
            "residual_norm", "parser"])
        w.writeheader()
        w.writerows(long_rows)

    roll_path = out / "length_cell_rollup.csv"
    with roll_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, extrasaction="ignore", fieldnames=[
            "model", "scheme", "layer", "resample",
            "baseline_len", "baseline_gsm8k", "peak_len", "peak_len_alpha",
            "first_collapse_abs_alpha", "gsm8k_safe_abs_alpha",
            "residual_norm", "parser", "probe", "run_tag"])
        w.writeheader()
        w.writerows(sorted(cell_rows, key=lambda c: (
            c["model"], LABEL_ORDER.get(c["scheme"], 9),
            c["resample"] if c["resample"] is not None else 99)))

    # console comparison: fp16 vs int8 vs nf4, averaged over resamples
    agg = defaultdict(list)
    for c in cell_rows:
        agg[(c["model"], c["scheme"])].append(c)

    print(f"\nRead {len(cell_rows)} cells from {len(files)} files.\n")
    print("=== Length steering: fp16 vs int8 vs nf4 (mean over resamples) ===")
    hdr = (f"{'model':24} {'scheme':6} {'n':>2} {'base_len':>8} {'base_gsm':>8} "
           f"{'peak_len':>8} {'collapse|a|':>11} {'gsm_safe|a|':>11} {'resid_norm':>10}")
    print(hdr)
    print("-" * len(hdr))
    for key in sorted(agg, key=lambda k: (k[0], LABEL_ORDER.get(k[1], 9))):
        cs = agg[key]
        print(f"{key[0]:24} {key[1]:6} {len(cs):>2} "
              f"{mean([c['baseline_len'] for c in cs]):>8} "
              f"{mean([c['baseline_gsm8k'] for c in cs]):>8} "
              f"{mean([c['peak_len'] for c in cs]):>8} "
              f"{mean([c['first_collapse_abs_alpha'] for c in cs]):>11} "
              f"{mean([c['gsm8k_safe_abs_alpha'] for c in cs]):>11} "
              f"{mean([c['residual_norm'] for c in cs]):>10}")

    print(f"\nWrote:\n  {long_path.resolve()}\n  {roll_path.resolve()}")


if __name__ == "__main__":
    main()
