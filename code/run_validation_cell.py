#!/usr/bin/env python3
r"""
SteerQuant validation-cell driver (Windows 11).
==============================================
Runs the whole frozen-method validation sequence one step after another, streaming
each step's output live AND saving a timestamped combined log. Auto-detects the
auto-incremented `..._COMPLETE_*.json` filename each harness run produces and feeds
the LENGTH cell straight into the paired rescorer -- no manual filename copying.

Sequence:
  0. logit_lens_length_vector.py            (fast, no generation; diagnostic)
  1. harness  --target sentiment --site last   (GPU)
  2. harness  --target length   --site last   (GPU)
  3. steerquant_analysis.py                 (no GPU; IECC report)
  4. rescore_length_paired.py <length json> (no GPU; paired estimator)

Frozen method is already the harness default (layer 14, chat-templated, assistant-role
extraction, EOS set); only --site last must be passed, which this driver does.

Usage (from the project folder):
    python run_validation_cell.py                 # full: --subset 200
    python run_validation_cell.py --quick         # smoke test the driver: --subset 5
    python run_validation_cell.py --subset 200 --site last --targets sentiment,length
    python run_validation_cell.py --model Qwen/Qwen2.5-7B-Instruct

Notes:
  * LM Studio is NOT needed (sentiment uses the in-process RoBERTa judge; length is
    judge-free). Close it first to free VRAM.
  * Critical GPU steps abort the run on failure; the no-GPU report/rescore steps are
    best-effort and won't abort.
  * Forces UTF-8 in child processes so the logit-lens CJK tokens don't crash on Windows.
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
PY = sys.executable  # same interpreter / venv as this script

DONE_RE = re.compile(r"->\s*(\S+\.json)")


def _child_env():
    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"          # emit UTF-8 (CJK tokens in logit-lens)
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def run_step(num, name, argv, logf, critical):
    """Run one subprocess, tee output to console + log, return (rc, captured_text)."""
    bar = "=" * 78
    header = f"\n{bar}\n[STEP {num}] {name}\n  cmd: {' '.join(argv)}\n{bar}\n"
    print(header, end="", flush=True)
    logf.write(header)
    logf.flush()

    t0 = time.time()
    captured = []
    try:
        proc = subprocess.Popen(
            argv, cwd=str(PROJECT),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
            env=_child_env())
    except FileNotFoundError as e:
        msg = f"  !! could not launch: {e}\n"
        print(msg, end="", flush=True)
        logf.write(msg)
        return 1, ""

    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        logf.write(line)
        captured.append(line)
    proc.wait()
    logf.flush()

    elapsed = time.time() - t0
    status = "OK" if proc.returncode == 0 else f"FAILED (rc={proc.returncode})"
    footer = f"  [STEP {num}] {status} in {elapsed:.1f}s\n"
    print(footer, end="", flush=True)
    logf.write(footer)
    logf.flush()

    if proc.returncode != 0 and critical:
        abort = (f"\n  ABORTING: step {num} is critical and failed. "
                 f"Downstream steps skipped. See log.\n")
        print(abort, end="", flush=True)
        logf.write(abort)
        raise SystemExit(1)

    return proc.returncode, "".join(captured)


def find_done_json(text):
    m = DONE_RE.search(text)
    return m.group(1) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", type=int, default=200,
                    help="MMLU items per alpha (default 200; --quick sets 5)")
    ap.add_argument("--quick", action="store_true",
                    help="smoke-test the driver fast (--subset 5)")
    ap.add_argument("--site", default="last", choices=["all", "last"],
                    help="steering intervention site (default last = frozen method)")
    ap.add_argument("--targets", default="sentiment,length",
                    help="comma-separated targets to run as cells")
    ap.add_argument("--model", default=None, help="override base model")
    ap.add_argument("--skip-logit-lens", action="store_true")
    args = ap.parse_args()

    subset = 5 if args.quick else args.subset
    targets = [t.strip() for t in args.targets.split(",") if t.strip()]
    model_args = ["--model", args.model] if args.model else []

    RESULTS.mkdir(exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = PROJECT / f"run_log_{stamp}.txt"

    t_all = time.time()
    completed = {}   # target -> COMPLETE json basename

    with open(log_path, "w", encoding="utf-8") as logf:
        intro = (f"SteerQuant validation-cell driver  |  {dt.datetime.now().isoformat()}\n"
                 f"  project : {PROJECT}\n"
                 f"  python  : {PY}\n"
                 f"  subset  : {subset}   site: {args.site}   targets: {targets}\n"
                 f"  model   : {args.model or '(harness default)'}\n"
                 f"  log     : {log_path}\n"
                 f"  (LM Studio not needed for sentiment/length -- close it for VRAM)\n")
        print(intro, end="", flush=True)
        logf.write(intro)

        step = 0
        # 0. logit-lens (non-critical diagnostic)
        if not args.skip_logit_lens:
            run_step(step, "logit-lens length vector",
                     [PY, "logit_lens_length_vector.py"], logf, critical=False)
            step += 1

        # 1..N. target cells (critical)
        for tgt in targets:
            argv = ([PY, "steerquant_phase0_harness.py",
                     "--target", tgt, "--subset", str(subset), "--site", args.site]
                    + model_args)
            _, out = run_step(step, f"{tgt} cell (subset={subset}, site={args.site})",
                              argv, logf, critical=True)
            fn = find_done_json(out)
            if fn:
                completed[tgt] = fn
                logf.write(f"  captured {tgt} output: {fn}\n")
            else:
                warn = f"  !! could not parse COMPLETE filename for {tgt} from output\n"
                print(warn, end="", flush=True)
                logf.write(warn)
            step += 1

        # report (non-critical)
        run_step(step, "analysis report (steerquant_analysis.py)",
                 [PY, "steerquant_analysis.py"], logf, critical=False)
        step += 1

        # paired rescore of the length cell (non-critical)
        if "length" in completed:
            length_json = str(RESULTS / completed["length"])
            run_step(step, "paired length rescore",
                     [PY, "rescore_length_paired.py", length_json], logf, critical=False)
            step += 1
        else:
            msg = "  (skipping paired rescore: no length cell completed this run)\n"
            print(msg, end="", flush=True)
            logf.write(msg)

        # summary
        total = time.time() - t_all
        summary = ["\n" + "=" * 78, "  RUN COMPLETE",
                   f"  total wall time: {total/60:.1f} min",
                   f"  log: {log_path}"]
        if completed:
            summary.append("  result files (in results\\):")
            for tgt, fn in completed.items():
                summary.append(f"    {tgt:>10}: {fn}")
        summary.append("=" * 78 + "\n")
        block = "\n".join(summary)
        print(block, flush=True)
        logf.write(block)


if __name__ == "__main__":
    main()
