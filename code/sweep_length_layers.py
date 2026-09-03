#!/usr/bin/env python3
r"""
SteerQuant -- LENGTH layer sweep (site=last).  EXPLORATORY.
===========================================================
Runs the LENGTH cell at several steering layers and paired-rescores each, so the
"accept the modest layer-14 effect vs. layer-sweep for a bigger clean signal"
decision can be made from data rather than blind. Mirrors run_validation_cell.py's
streaming + tee + auto-detect-COMPLETE-json pattern.

For each layer L in --layers:
    python steerquant_phase0_harness.py --target length --site last --layer L
           --alphas <grid> --subset <mmlu>
    python rescore_length_paired.py <that layer's COMPLETE json>
Then prints a LAYER COMPARISON table: signed mean dL + fail_rate at reference
alphas, so you can eyeball which layer gives the largest CLEAN sign-consistent
signed length effect (monotone, low fail).

IMPORTANT
  * This is EXPLORATORY. The confirmatory steering layer is prereg-LOCKED at
    floor(0.5 x N_layers) (=14 for Qwen2.5-7B) in base prereg s4. Moving it would
    require a dated prereg amendment; this sweep only informs that decision.
  * Reads the SIGN-FIXED stimulus (pair order reversed 2026-06-30), so a working
    layer shows POSITIVE mean dL on the +arm and NEGATIVE on the -arm.
  * Compact alpha grid + small MMLU subset by default to keep the sweep affordable
    (length gens at 2048 tokens dominate; ~24 min/layer at 7 alphas). MMLU here is
    incidental -- the sweep is about length efficacy, not IECC.

Usage (from the project folder):
    python sweep_length_layers.py                       # layers 10..20, compact grid
    python sweep_length_layers.py --layers 12 14 16     # custom layer set
    python sweep_length_layers.py --alphas -40 -20 -10 0 10 20 40 --subset 20
    python sweep_length_layers.py --model Qwen/Qwen2.5-7B-Instruct
"""
import argparse
import datetime as dt
import os
import re
import subprocess
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
RESULTS = PROJECT / "results"
PY = sys.executable

DONE_RE = re.compile(r"->\s*(\S+\.json)")
# rescore table row:  -40.0 |    102.3      82.0 |     10 |      0.00 |        268.5
ROW_RE = re.compile(
    r"^\s*(-?\d+\.\d+)\s*\|\s*(-?\d+\.\d+|nan)\s+(-?\d+\.\d+|nan)\s*\|\s*"
    r"(\d+)\s*\|\s*(\d+\.\d+)\s*\|")

DEFAULT_LAYERS = [10, 12, 14, 16, 18, 20]
DEFAULT_ALPHAS = [-40.0, -20.0, -10.0, 0.0, 10.0, 20.0, 40.0]
REF_ALPHAS = [-40.0, -20.0, 20.0, 40.0]  # reported in the comparison table


def _child_env():
    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def run_step(name, argv, logf, critical):
    bar = "=" * 78
    header = f"\n{bar}\n[{name}]\n  cmd: {' '.join(argv)}\n{bar}\n"
    print(header, end="", flush=True)
    logf.write(header); logf.flush()
    t0 = time.time()
    captured = []
    try:
        proc = subprocess.Popen(
            argv, cwd=str(PROJECT), stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, encoding="utf-8",
            errors="replace", bufsize=1, env=_child_env())
    except FileNotFoundError as e:
        msg = f"  !! could not launch: {e}\n"
        print(msg, end="", flush=True); logf.write(msg)
        return 1, ""
    for line in proc.stdout:
        sys.stdout.write(line); sys.stdout.flush()
        logf.write(line); captured.append(line)
    proc.wait(); logf.flush()
    elapsed = time.time() - t0
    status = "OK" if proc.returncode == 0 else f"FAILED (rc={proc.returncode})"
    footer = f"  [{name}] {status} in {elapsed:.1f}s\n"
    print(footer, end="", flush=True)
    logf.write(footer); logf.flush()
    if proc.returncode != 0 and critical:
        abort = f"\n  ABORTING: critical step failed ({name}). See log.\n"
        print(abort, end="", flush=True); logf.write(abort)
        raise SystemExit(1)
    return proc.returncode, "".join(captured)


def find_done_json(text):
    m = DONE_RE.search(text)
    return m.group(1) if m else None


def parse_rescore(text):
    """rescore stdout -> {alpha: (mean_dL, fail_rate)}."""
    rows = {}
    for line in text.splitlines():
        m = ROW_RE.match(line)
        if not m:
            continue
        alpha = float(m.group(1))
        mean_dl = float("nan") if m.group(2) == "nan" else float(m.group(2))
        fail = float(m.group(5))
        rows[alpha] = (mean_dl, fail)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, nargs="+", default=DEFAULT_LAYERS,
                    help=f"steering layers to sweep (default {DEFAULT_LAYERS})")
    ap.add_argument("--alphas", type=float, nargs="+", default=DEFAULT_ALPHAS,
                    help="alpha grid passed to the harness (compact by default)")
    ap.add_argument("--subset", type=int, default=20,
                    help="MMLU items per alpha (default 20; incidental for this sweep)")
    ap.add_argument("--site", default="last", choices=["all", "last"],
                    help="intervention site (default last = frozen method)")
    ap.add_argument("--model", default=None, help="override base model")
    args = ap.parse_args()

    model_args = ["--model", args.model] if args.model else []
    alpha_str = [str(a) for a in args.alphas]

    RESULTS.mkdir(exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = PROJECT / f"sweep_length_layers_log_{stamp}.txt"

    t_all = time.time()
    per_layer = {}   # layer -> {alpha: (mean_dL, fail)}

    with open(log_path, "w", encoding="utf-8") as logf:
        intro = (
            f"SteerQuant LENGTH layer sweep (EXPLORATORY)  |  {dt.datetime.now().isoformat()}\n"
            f"  project : {PROJECT}\n"
            f"  python  : {PY}\n"
            f"  layers  : {args.layers}\n"
            f"  alphas  : {args.alphas}\n"
            f"  subset  : {args.subset}   site: {args.site}\n"
            f"  model   : {args.model or '(harness default)'}\n"
            f"  log     : {log_path}\n"
            f"  NOTE: confirmatory layer is prereg-locked at floor(0.5xN); this is\n"
            f"        exploratory data to inform accept-vs-sweep. Close LM Studio for VRAM.\n")
        print(intro, end="", flush=True)
        logf.write(intro)

        for L in args.layers:
            argv = ([PY, "steerquant_phase0_harness.py", "--target", "length",
                     "--site", args.site, "--layer", str(L),
                     "--alphas"] + alpha_str + ["--subset", str(args.subset)]
                    + model_args)
            _, out = run_step(f"length cell @ layer {L}", argv, logf, critical=True)
            fn = find_done_json(out)
            if not fn:
                warn = f"  !! could not parse COMPLETE filename for layer {L}\n"
                print(warn, end="", flush=True); logf.write(warn)
                continue
            length_json = str(RESULTS / fn)
            _, rout = run_step(f"paired rescore @ layer {L}",
                               [PY, "rescore_length_paired.py", length_json],
                               logf, critical=False)
            per_layer[L] = parse_rescore(rout)

        # ---- comparison table ----
        def cell(rows, a):
            if a not in rows:
                return "    --      "
            dl, fail = rows[a]
            dl_s = "  nan " if dl != dl else f"{dl:+7.1f}"
            flag = "*" if fail >= 0.5 else " "  # * = unreliable (>=50% fail)
            return f"{dl_s}{flag}"

        head = "  layer | " + " | ".join(f"a={a:+.0f}" for a in REF_ALPHAS)
        lines = ["\n" + "=" * 78,
                 "  LENGTH LAYER SWEEP -- signed mean dL at reference alphas",
                 "  (+arm should be POSITIVE = lengthens, -arm NEGATIVE; '*' = fail>=0.5, unreliable)",
                 "-" * 78, head, "-" * 78]
        for L in args.layers:
            rows = per_layer.get(L, {})
            cells = " | ".join(cell(rows, a) for a in REF_ALPHAS)
            lines.append(f"  {L:>5} | {cells}")
        lines += ["-" * 78,
                  "  Read: the best layer maximizes clean signed magnitude (large |dL|,",
                  "  correct sign, NO '*' flag) across the reference alphas. Compare vs",
                  "  layer 14 before proposing any prereg amendment.",
                  f"  total wall time: {(time.time()-t_all)/60:.1f} min",
                  f"  log: {log_path}",
                  "=" * 78 + "\n"]
        block = "\n".join(lines)
        print(block, flush=True)
        logf.write(block)


if __name__ == "__main__":
    main()
