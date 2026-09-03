#!/usr/bin/env python3
"""
SteerQuant -- LLM-judge scoring pass (Backend 2 driver)
=======================================================
Reads a harness results JSON (which already stores the generations per alpha),
scores each generation with the rubric-based LLM judge (LM Studio) for the given
target, and writes the scores back as `efficacy_llm`.

WHY A SEPARATE PASS: the generation harness holds the steered model on the GPU via
transformers; an LLM judge needs its own model loaded in LM Studio. Running both at
once contends for the 4090's VRAM. So the workflow is sequential:

    1. (LM Studio model UNLOADED) run steerquant_phase0_harness.py  -> saves generations
    2. (load a judge model in LM Studio) run THIS script over that results JSON

Deterministic (temperature 0, structured output). stdlib + numpy only.

Usage:
    python steerquant_score_llm.py --results results\\SteerQuant_phase0_fp16_3_COMPLETE_20260624.json --target sycophancy
"""
import argparse
import json
from pathlib import Path

import numpy as np

from steerquant_judge import get_judge


def main():
    ap = argparse.ArgumentParser(description="Score saved generations with the LLM judge.")
    ap.add_argument("--results", required=True, help="path to a harness *_COMPLETE_*.json")
    ap.add_argument("--target", required=True, help="behavioral target with a locked rubric (e.g. sycophancy)")
    args = ap.parse_args()

    path = Path(args.results)
    if not path.exists():
        raise SystemExit(f"results file not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))

    judge = get_judge(args.target)
    print(f"Judge: {judge.manifest()}")
    print("=" * 60)

    for cell in data.get("by_alpha", []):
        texts = [g["text"] for g in cell.get("generations", [])]
        if not texts:
            continue
        scores = judge.score(texts)
        cell["efficacy_llm"] = round(float(np.mean(scores)), 3)
        cell["efficacy_llm_per"] = [round(float(s), 3) for s in scores]
        print(f"  alpha={cell['alpha']:5.1f}  efficacy_llm={cell['efficacy_llm']:+.3f}")

    data.setdefault("meta", {})["llm_judge"] = judge.manifest()
    out = path.with_name(f"{path.stem}_llm-{args.target}.json")
    out.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print("=" * 60)
    print(f"Wrote {out.name}")


if __name__ == "__main__":
    main()
