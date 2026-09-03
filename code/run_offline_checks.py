#!/usr/bin/env python3
r"""
SteerQuant -- offline verification runner (2026-07-06)
=====================================================
Runs every FAST, NON-GPU check accumulated over the S1-S13 code work, in order,
and prints a PASS/FAIL summary. No model is loaded; safe to run anytime.

Usage (from the project folder):
    python run_offline_checks.py

Exit code 0 iff every check passes (so it's CI-friendly).

NOTE: the GPU smoke tests are NOT run here -- they load Qwen2.5-7B and take
minutes. Run those by hand after this passes (see the reminder printed at the end).
"""
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = sys.executable  # same interpreter you launched this with (your conda env)

# (label, argv). Each runs as: <python> <args...>  in the project folder.
CHECKS = [
    ("py_compile: harness",
     ["-m", "py_compile", "steerquant_phase0_harness.py"]),
    ("py_compile: analysis",
     ["-m", "py_compile", "steerquant_analysis.py"]),
    ("py_compile: termination detector",
     ["-m", "py_compile", "termination_failure_detector.py"]),
    ("py_compile: trajectory",
     ["-m", "py_compile", "steerquant_trajectory.py"]),
    ("py_compile: estar (shared E*/arm/crossing rule, 2026-07-11 extraction)",
     ["-m", "py_compile", "steerquant_estar.py"]),
    ("py_compile: matrix orchestrator",
     ["-m", "py_compile", "run_matrix.py"]),
    ("py_compile: judge",
     ["-m", "py_compile", "steerquant_judge.py"]),
    ("py_compile: LLM-judge scoring pass",
     ["-m", "py_compile", "steerquant_score_llm.py"]),
    ("py_compile: length stimulus",
     ["-m", "py_compile", "SteerQuant_length_stimulus_2026-06-30.py"]),
    ("py_compile: sycophancy stimulus",
     ["-m", "py_compile", "SteerQuant_sycophancy_stimulus_2026-06-30.py"]),
    ("selftest: steerquant_trajectory (S9 metrics; norm-invariance)",
     ["steerquant_trajectory.py"]),
    ("pytest: termination_failure_detector (19 tests)",
     ["-m", "pytest", "test_termination_failure_detector.py", "-q"]),
    ("selftest: steerquant_analysis (cases [1]-[18]; S11/S12/S13 + 07-07 fixes + [16] adaptive files + [17] pooling guard + [18] Option C REML+mKH combiner)",
     ["steerquant_analysis.py", "--selftest"]),
    ("selftest: steerquant_estar (adaptive capability-alpha selection; determinism/crossings/recentering/fallback)",
     ["steerquant_estar.py"]),
    ("selftest: harness adaptive plumbing (self/sibling/guard paths, no GPU)",
     ["steerquant_phase0_harness.py", "--adaptive-selftest"]),
    ("selftest: resample_plan (S10 data-level resampling; determinism/shape/range)",
     ["steerquant_phase0_harness.py", "--resample-selftest"]),
    ("selftest: gsm8k_parse_answer (capability probe deviation 2026-07-10; #### / \\boxed parsing)",
     ["steerquant_phase0_harness.py", "--gsm8k-selftest"]),
]


def run(label, args):
    print("=" * 72)
    print(f"RUN  {label}")
    print(f"     {PY} {' '.join(args)}")
    print("-" * 72)
    t0 = time.time()
    proc = subprocess.run([PY, *args], cwd=str(HERE))
    dt = time.time() - t0
    ok = proc.returncode == 0
    print("-" * 72)
    print(f"{'PASS' if ok else 'FAIL'}  {label}   ({dt:.1f}s, exit={proc.returncode})")
    return ok


def main():
    print("SteerQuant offline checks  |  python =", PY)
    print("folder =", HERE)
    results = [(label, run(label, args)) for label, args in CHECKS]

    print("\n" + "#" * 72)
    print("SUMMARY")
    print("#" * 72)
    width = max(len(label) for label, _ in results)
    for label, ok in results:
        print(f"  [{'PASS' if ok else 'FAIL'}]  {label:<{width}}")
    n_pass = sum(1 for _, ok in results if ok)
    all_ok = n_pass == len(results)
    print("-" * 72)
    print(f"  {n_pass}/{len(results)} checks passed  ->  {'ALL GREEN' if all_ok else 'SOME FAILED'}")

    print("\nGPU smokes (run by hand, NOT included above -- they load the model):")
    print("  python steerquant_phase0_harness.py --target sycophancy --subset 2   # S6 (already run 2026-07-06)")
    print("  python steerquant_phase0_harness.py --target length --subset 5       # re-smoke length under S3 token metric")
    print("  python steerquant_phase0_harness.py --trajectory --subset 2          # S9 smoke: text byte-identical, trajectory_mean populated")
    print("  python steerquant_phase0_harness.py --target sentiment --subset 5 --resample-run 1   # S10 smoke: meta.resample block, indices recorded")
    print("  python steerquant_phase0_harness.py --capability gsm8k --subset 3 --alphas 0 -25   # GSM8K probe smoke: capability_failure_rate populated, meta.capability_probe=gsm8k")
    print("  python run_matrix.py --schemes fp16 --targets sentiment --alphas -20 0 20 --subset 5 --runs 2   # S10: two DISTINCT resampled r1/r2 cells")
    print("  (move 06-30 length dev files to D:\\Claude\\Trash before the real length run)")
    print("  -- 2026-07-11 economy levers: TWO equivalence smokes REQUIRED before the matrix --")
    print("  python steerquant_phase0_harness.py --subset 20 --alphas 0 -25 --capability-batch-size 1  --run-tag eqsmoke_b1    # A: per-item reference")
    print("  python steerquant_phase0_harness.py --subset 20 --alphas 0 -25 --capability-batch-size 16 --run-tag eqsmoke_b16   # A: batched; diff capability_mmlu_items+capability_failure_flags vs b1 (expect row-for-row match; investigate ANY mismatch)")
    print("  python steerquant_phase0_harness.py --subset 20 --capability-alpha-mode adaptive --run-tag eqsmoke_adaptive       # B: fp16 self path; check meta.capability_alpha_selection + capability keys ONLY at selected alphas; alpha* must match steerquant_analysis on a full-grid file")
    print("  (eqsmoke_* files are DIAGNOSTICS -- move to D:\\Claude\\Trash after checking; never pool)")

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
