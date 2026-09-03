#!/usr/bin/env python3
r"""
SteerQuant -- confirmatory MATRIX orchestrator (resumable + shardable).
=======================================================================
Enumerates the confirmatory cells (model x scheme x target x run) and runs each as
one harness invocation with a DETERMINISTIC --run-tag. A cell whose COMPLETE json
already exists is SKIPPED. This makes the matrix its own final test: if a cell dies
(or a whole box dies), just re-launch -- finished cells are skipped, only the
missing ones run. No separate dress rehearsal.

Resumability contract
  * Each cell writes  results/SteerQuant_<tag>_COMPLETE_<date>.json  (harness --run-tag).
  * tag = <model_slug>_<scheme>_<target>_L<layer>_r<run>  (stable across days/boxes).
  * Skip rule: a SteerQuant_<tag>_COMPLETE_*.json marks the cell done ONLY if its
    meta.capability_probe matches this launch's --capability (2026-07-12 guard:
    stale/unstamped files never silently satisfy a skip).

Parallel across boxes/GPUs
  * --shard i/N keeps only cells with (index % N == i-1); launch N processes, each
    with a different i, to cover the matrix with no overlap. Skips + shards compose,
    so a re-launch after a crash is safe.
  * --gpu K sets CUDA_VISIBLE_DEVICES=K for the child harness (pin one shard per GPU).

Weight-only v1 scope (per SteerQuant_vast_build_plan_2026-06-30.md): fp16 + bnb +
GPTQ/AWQ. Activation-quant (W8A8/W4A4) is deferred; the harness will raise
NotImplementedError if asked, and this orchestrator reports it as a failed cell.

Usage (from the project folder):
    # smoke test the plumbing (runnable TODAY: Qwen + fp16 + sentiment/length):
    python run_matrix.py --models Qwen/Qwen2.5-7B-Instruct --schemes fp16 \
        --targets sentiment length --alphas -20 0 20 --subset 5 --runs 1

    # full weight-only matrix on one box (defaults = confirmatory grid, 5 runs):
    python run_matrix.py

    # parallel: 4 boxes, box #2 of 4, pinned to GPU 0:
    python run_matrix.py --shard 2/4 --gpu 0
"""
import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
RESULTS = PROJECT / "results"
PY = sys.executable

# Per-model steering layer = floor(0.5 x N_layers), prereg s4. A model MUST have an
# entry here (we can't compute it without loading the model). Extend as models are added.
MODEL_LAYERS = {
    "Qwen/Qwen2.5-7B-Instruct": 14,   # 28 layers
    "meta-llama/Meta-Llama-3-8B-Instruct": 16,   # 32 layers (gated: needs HF token)
    "meta-llama/Meta-Llama-3.1-8B-Instruct": 16, # 32 layers (recommended: clean quant repos)
    "mistralai/Mistral-7B-Instruct-v0.3": 16,    # 32 layers
    "google/gemma-2-9b-it": 21,                  # 42 layers = floor(0.5*42); 4th family (2026-07-15 smoke)
    "meta-llama/Meta-Llama-3-70B-Instruct": 40,  # 80 layers (A100; v2/stretch)
}

# Defaults. v1 = weight-only. fp16 + bnb run today; GPTQ/AWQ need CHECKPOINTS filled.
DEFAULT_MODELS = ["Qwen/Qwen2.5-7B-Instruct"]   # add families as checkpoints/tokens land
DEFAULT_SCHEMES = ["fp16", "w8a16_bnb_int8", "w4a16_bnb_nf4"]  # + w8a16_gptq w4a16_gptq w4a16_awq when ready
DEFAULT_TARGETS = ["sentiment", "length"]        # + sycophancy truthfulness refusal when wired
# Confirmatory alpha grid = the harness default (dense near 0, coarse far).
# Densified negative mid-band (-25/-30/-35) 2026-07-07 pre-data (see harness ALPHAS
# note): layer-14 residual norm ~409 puts the E* crossing on the negative arm in the
# old -20..-40 gap. Mirror of the harness default -- keep the two in sync.
DEFAULT_ALPHAS = [-80, -60, -40, -35, -30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 40, 60, 80]

# Per-model confirmatory grids (dated method note 2026-07-13; Saurav approved).
# Residual norms at the steering layer differ ~20-30x across families (Qwen L14
# ~409; Llama-3.1 L16 ~13.0; Mistral L16 ~20.6 -- 07-13 band smokes), so one
# grid cannot serve all: E* crossings sit near alpha* ~ -25 (Qwen), ~ -7
# (Llama), ~ -2.5 (Mistral). Each grid is symmetric, dense around its model's
# crossing, with coarse far points to capture the capability collapse. A model
# not listed here falls back to DEFAULT_ALPHAS (Qwen-calibrated).
MODEL_ALPHAS = {
    "Qwen/Qwen2.5-7B-Instruct": DEFAULT_ALPHAS,
    "meta-llama/Meta-Llama-3.1-8B-Instruct":
        [-40, -30, -20, -15, -12, -10, -8, -6, -4, -2, 0,
         2, 4, 6, 8, 10, 12, 15, 20, 30, 40],
    "mistralai/Mistral-7B-Instruct-v0.3":
        [-30, -20, -15, -10, -7, -5, -4, -3, -2, -1, 0,
         1, 2, 3, 4, 5, 7, 10, 15, 20, 30],
    # Gemma-2-9B (2026-07-15 smoke): residual norm @ L21 ~657 -- an order of
    # magnitude larger than Llama/Mistral (~13-20) and bigger than Qwen (~409).
    # Sentiment crossing ~|a|70-120; FULL degeneration (gsm8k fail=1.0) at
    # |a|>=400, so the grid is CAPPED at +-300 to avoid spending 2048-token capped
    # generations on all-degenerate alphas. PROVISIONAL -- confirm the LENGTH
    # crossing with a length band smoke before launching Gemma's matrix.
    "google/gemma-2-9b-it":
        [-300, -250, -200, -160, -120, -100, -80, -60, -40, -20, 0,
         20, 40, 60, 80, 100, 120, 160, 200, 250, 300],
}


# --- Rough ETA machinery (2026-07-16) ---------------------------------------
# Cost multipliers relative to fp16, from the sentiment-run empirical ratios
# (int8 ~3-4x fp16, nf4 ~1.5x). Used ONLY for the [eta] progress line printed
# after each cell -- affects nothing that runs and no result file.
SCHEME_TIME_MULT = {"fp16": 1.0, "w8a16_bnb_int8": 3.5, "w4a16_bnb_nf4": 1.5}


def _fmt_dur(seconds):
    h, rem = divmod(int(seconds), 3600)
    m = rem // 60
    return f"{h}h{m:02d}m" if h else f"{m}m"


def eta_estimate(observed, remaining):
    """Rough remaining-time line (string, newline-terminated).
    observed: {(model, scheme): [elapsed_sec, ...]} for cells finished THIS process.
    remaining: [(model, scheme), ...] shard cells not yet done.
    Base per cell: its own (model, scheme) mean if seen; else the model's fp16
    mean x scheme multiplier; else the cross-model fp16 mean x multiplier.
    Cells with no base at all are reported as unknown. ROUGH by design."""
    if not remaining:
        return "  [eta] shard done -- no cells remaining.\n"
    mean = lambda xs: sum(xs) / len(xs)
    fp16_means = {m: mean(t) for (m, s), t in observed.items() if s == "fp16"}
    cross = mean(list(fp16_means.values())) if fp16_means else None
    total, unknown = 0.0, 0
    for model, scheme in remaining:
        mult = SCHEME_TIME_MULT.get(scheme, 3.5)
        if (model, scheme) in observed:
            total += mean(observed[(model, scheme)])
        elif model in fp16_means:
            total += fp16_means[model] * mult
        elif cross is not None:
            total += cross * mult
        else:
            unknown += 1
    if unknown == len(remaining):
        return (f"  [eta] {len(remaining)} cell(s) remaining this shard; no timing "
                f"observed yet -- estimate appears after the first cell completes.\n")
    line = (f"  [eta] {len(remaining)} cell(s) remaining this shard; rough remaining "
            f"~{_fmt_dur(total)} (int8 x3.5 / nf4 x1.5 vs fp16; re-estimates as "
            f"cells finish -- ROUGH)")
    if unknown:
        line += f"; +{unknown} cell(s) with no timing base yet"
    return line + "\n"


def slug(model_name: str) -> str:
    return model_name.split("/")[-1].replace(".", "-")


def cell_tag(model, scheme, target, layer, run):
    return f"{slug(model)}_{scheme}_{target}_L{layer}_r{run}"


# Harness stamp written to meta.capability_probe for each --capability choice
# (the harness ALWAYS stamps since 2026-07-11; earlier files carry no stamp).
PROBE_STAMPS = {"gsm8k": "gsm8k", "mmlu": "single_pass_mmlu"}


def _file_meta(path):
    """meta dict of a COMPLETE json, or None if unreadable/corrupt."""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8")).get("meta", {})
    except Exception:
        return None


def _probe_matches(path, expected_stamp):
    meta = _file_meta(path)
    return meta is not None and meta.get("capability_probe") == expected_stamp


def is_done(tag: str, expected_stamp: str):
    """(done, notes). 2026-07-12 stale-skip guard: a tag-matching COMPLETE file
    counts as done ONLY if its meta.capability_probe matches the probe THIS
    launch uses. Before this guard ANY tag-matching file satisfied the skip
    rule -- on 2026-07-12 a stale 07-07 dose-blind single-pass-MMLU file
    silently kept fp16 r1 from re-running with the GSM8K probe. Unstamped
    (pre-2026-07-11) or unreadable files also do NOT count."""
    matches = sorted(RESULTS.glob(f"SteerQuant_{tag}_COMPLETE_*.json"))
    good = [p for p in matches if _probe_matches(p, expected_stamp)]
    stale = [p for p in matches if p not in good]
    notes = []
    for p in stale:
        meta = _file_meta(p)
        got = ("<unreadable json>" if meta is None
               else repr(meta.get("capability_probe")))
        notes.append(f"  !! [probe-guard] {tag}: {p.name} capability_probe={got} "
                     f"!= {expected_stamp!r} -> does NOT count as done.")
    if stale and not good:
        notes.append(f"  !! [probe-guard] {tag}: move the stale file(s) above "
                     "(json+txt+npy trio) to Trash BEFORE re-running -- a same-day "
                     "rerun auto-increments the run-tag (breaking skip/resume and the "
                     "--e-star-from sibling lookup), and a stale file left in results/ "
                     "still poisons analysis globs.")
    return bool(good), notes


def sibling_fp16_complete(model, target, layer, run, expected_stamp):
    """Latest VALID COMPLETE json of the fp16 sibling cell (same model/target/
    layer/run -- run matters: run r shares the (SEED, r) data resample across
    schemes, so the pairing premise holds). Valid = probe stamp matches this
    launch (2026-07-12 guard); a stale-probe sibling is treated as ABSENT, so
    the quantized cell goes pending-on-dependency until the fp16 cell re-runs
    with the right probe. None if no valid sibling exists."""
    tag = cell_tag(model, "fp16", target, layer, run)
    matches = sorted(RESULTS.glob(f"SteerQuant_{tag}_COMPLETE_*.json"))
    good = [p for p in matches if _probe_matches(p, expected_stamp)]
    return good[-1] if good else None


def _child_env(gpu):
    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    return env


def run_cell(tag, argv, logf, gpu):
    bar = "=" * 78
    header = f"\n{bar}\n[CELL {tag}]\n  cmd: {' '.join(argv)}\n{bar}\n"
    print(header, end="", flush=True); logf.write(header); logf.flush()
    t0 = time.time()
    try:
        proc = subprocess.Popen(
            argv, cwd=str(PROJECT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1, env=_child_env(gpu))
    except FileNotFoundError as e:
        msg = f"  !! could not launch: {e}\n"
        print(msg, end="", flush=True); logf.write(msg)
        return 1, 0.0
    for line in proc.stdout:
        sys.stdout.write(line); sys.stdout.flush(); logf.write(line)
    proc.wait(); logf.flush()
    elapsed = time.time() - t0
    status = "OK" if proc.returncode == 0 else f"FAILED (rc={proc.returncode})"
    footer = f"  [CELL {tag}] {status} in {elapsed:.1f}s ({_fmt_dur(elapsed)})\n"
    print(footer, end="", flush=True); logf.write(footer); logf.flush()
    return proc.returncode, elapsed


def parse_shard(s):
    if not s:
        return (1, 1)
    i, n = s.split("/")
    i, n = int(i), int(n)
    if not (1 <= i <= n):
        raise SystemExit(f"--shard i/N requires 1<=i<=N (got {s})")
    return (i, n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--schemes", nargs="+", default=DEFAULT_SCHEMES)
    ap.add_argument("--targets", nargs="+", default=DEFAULT_TARGETS)
    ap.add_argument("--runs", type=int, default=5,
                    help="resampled runs per cell (prereg s4 = 5). With --resample (default), "
                         "run r is a REAL data-level resample seeded from (SEED, r) in the "
                         "harness (S10). With --no-resample, runs>1 repeat the identical "
                         "deterministic run (plumbing tests only -> fake replication).")
    ap.add_argument("--resample", dest="resample", action="store_true", default=True,
                    help="S10 (prereg s4): pass --resample-run r to each cell so every run "
                         "is a real data-level resample (contrast pairs + eval/benchmark "
                         "items, with replacement). ON by default.")
    ap.add_argument("--no-resample", dest="resample", action="store_false",
                    help="disable S10 resampling: runs>1 repeat the identical deterministic "
                         "run (plumbing/skip-logic tests only; produces fake replication).")
    ap.add_argument("--alphas", type=float, nargs="+", default=None,
                    help="explicit alpha grid applied to EVERY cell (override). "
                         "Default None = each cell uses its model's grid from "
                         "MODEL_ALPHAS (method note 2026-07-13), falling back to "
                         "DEFAULT_ALPHAS for unlisted models.")
    ap.add_argument("--subset", type=int, default=200, help="capability items per alpha (prereg=200)")
    ap.add_argument("--site", default="last", choices=["all", "last"])
    ap.add_argument("--capability", choices=["mmlu", "gsm8k"], default="gsm8k",
                    help="capability probe passed to every cell. DEFAULT gsm8k = the "
                         "prereg-deviation 2026-07-10 generation-consistent PRIMARY probe. "
                         "'mmlu' is the legacy single-pass probe, DOSE-BLIND under "
                         "site=last (07-07 adjudicator) -- diagnostics only, NEVER "
                         "confirmatory.")
    ap.add_argument("--secondary-mmlu", action="store_true",
                    help="with gsm8k, also run single-pass MMLU per alpha as the SECONDARY "
                         "probe (Saurav 2026-07-10; dissociation figure). Adds ~no cost "
                         "(single forward pass per item).")
    ap.add_argument("--capability-max-new-tokens", type=int, default=None,
                    help="token cap for GSM8K capability generations (harness default 512, "
                         "confirmed by the 07-10 adjudicator: no in-band truncation).")
    ap.add_argument("--capability-alpha-mode", choices=["full", "adaptive"],
                    default="adaptive",
                    help="DEFAULT adaptive (2026-07-11 economy lever): efficacy at every "
                         "grid alpha, capability only at baseline + a window around the "
                         "prereg sec.7 E* crossings (fp16 self-referential; other schemes "
                         "read the sibling fp16 file via --e-star-from, wired here "
                         "automatically). Cells whose fp16 sibling isn't COMPLETE yet are "
                         "held as pending-on-dependency, NOT launched blind -- re-run the "
                         "same command after the fp16 cells finish. 'full' = probe every "
                         "alpha (legacy cost, no dependency).")
    ap.add_argument("--capability-alpha-neighbors", type=int, default=2,
                    help="adaptive mode: grid neighbors kept on EACH side of the "
                         "nearest-to-crossing alpha (default 2).")
    ap.add_argument("--capability-batch-size", type=int, default=16,
                    help="GSM8K probe generation batch size passed to every cell "
                         "(2026-07-11 economy lever; DEFAULT 16 so batching is UNIFORM "
                         "matrix-wide, per the dated pre-data method note). Pass 1 for "
                         "the legacy per-item path. Run the GPU equivalence smoke before "
                         "confirmatory batched use.")
    ap.add_argument("--shard", default=None, help="i/N: run only cells with index %% N == i-1")
    ap.add_argument("--gpu", default=None, help="CUDA_VISIBLE_DEVICES for child harness")
    ap.add_argument("--dry-run", action="store_true", help="list cells + done/pending, run nothing")
    args = ap.parse_args()

    shard_i, shard_n = parse_shard(args.shard)
    probe_stamp = PROBE_STAMPS[args.capability]

    # Build the ordered cell list (stable order = stable sharding).
    cells = []
    for model in args.models:
        if model not in MODEL_LAYERS:
            raise SystemExit(f"no layer registered for {model!r}; add it to MODEL_LAYERS.")
        layer = MODEL_LAYERS[model]
        for scheme in args.schemes:
            for target in args.targets:
                for run in range(1, args.runs + 1):
                    cells.append((model, scheme, target, layer, run))

    RESULTS.mkdir(exist_ok=True)

    # ETA precompute (2026-07-16): note which of this shard's cells are already
    # done at launch, so the [eta] line only counts genuinely-remaining work.
    # Guard notes are emitted by the authoritative is_done call in the main loop.
    shard_cells = [(i, c) for i, c in enumerate(cells) if i % shard_n == (shard_i - 1)]
    done_start = {i: is_done(cell_tag(c[0], c[1], c[2], c[3], c[4]), probe_stamp)[0]
                  for i, c in shard_cells}
    observed = {}   # (model, scheme) -> [elapsed seconds, ...] this process

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    shard_label = f"shard{shard_i}of{shard_n}" if args.shard else "single"
    log_path = PROJECT / f"run_matrix_log_{stamp}_{shard_label}.txt"

    ran = skipped = failed = pending_dep = 0
    failed_tags, pending_dep_tags = [], []

    with open(log_path, "w", encoding="utf-8") as logf:
        intro = (
            f"SteerQuant matrix orchestrator  |  {dt.datetime.now().isoformat()}\n"
            f"  models  : {args.models}\n"
            f"  schemes : {args.schemes}\n"
            f"  targets : {args.targets}\n"
            f"  runs    : {args.runs}   subset: {args.subset}   site: {args.site}   resample: {args.resample}\n"
            f"  capability: {args.capability}   secondary-mmlu: {args.secondary_mmlu}   "
            f"cap-max-new: {args.capability_max_new_tokens or 'harness default (512)'}\n"
            f"  cap-alpha-mode: {args.capability_alpha_mode} "
            f"(neighbors={args.capability_alpha_neighbors})   "
            f"cap-batch-size: {args.capability_batch_size}\n"
            f"  alphas  : {args.alphas if args.alphas is not None else 'per-model (MODEL_ALPHAS, method note 2026-07-13)'}\n"
            f"  shard   : {shard_i}/{shard_n}   gpu: {args.gpu}\n"
            f"  cells   : {len(cells)} total (before shard/skip)\n"
            f"  log     : {log_path}\n")
        print(intro, end="", flush=True); logf.write(intro)

        for idx, (model, scheme, target, layer, run) in enumerate(cells):
            if idx % shard_n != (shard_i - 1):
                continue   # not this shard
            tag = cell_tag(model, scheme, target, layer, run)
            done, guard_notes = is_done(tag, probe_stamp)
            for note in guard_notes:
                print(note, flush=True); logf.write(note + "\n")
            if done:
                skipped += 1
                msg = f"  [skip] {tag} (COMPLETE exists, probe={args.capability})\n"
                print(msg, end="", flush=True); logf.write(msg)
                continue
            # Adaptive dependency (2026-07-11): a non-fp16 cell needs its fp16
            # sibling's COMPLETE file for the E* reference. Missing -> the cell
            # is PENDING-ON-DEPENDENCY (a new bucket beside skip/ran/failed),
            # never launched blind; a re-run picks it up once fp16 lands. On a
            # single unsharded box the fp16 cells run first (scheme is the
            # second loop), so this only bites cross-shard/cross-box launches.
            sibling = None
            if args.capability_alpha_mode == "adaptive" and scheme != "fp16":
                sibling = sibling_fp16_complete(model, target, layer, run, probe_stamp)
                if sibling is None:
                    pending_dep += 1; pending_dep_tags.append(tag)
                    sib_tag = cell_tag(model, "fp16", target, layer, run)
                    msg = (f"  [pending-dep] {tag} (needs fp16 sibling {sib_tag} "
                           f"COMPLETE for --e-star-from; re-run after it lands)\n")
                    print(msg, end="", flush=True); logf.write(msg)
                    continue
            if args.dry_run:
                msg = f"  [pending] {tag}\n"
                print(msg, end="", flush=True); logf.write(msg)
                continue
            argv = [PY, "steerquant_phase0_harness.py",
                    "--model", model, "--scheme", scheme, "--target", target,
                    "--layer", str(layer), "--site", args.site,
                    "--subset", str(args.subset), "--run-tag", tag,
                    "--capability", args.capability,
                    "--capability-alpha-mode", args.capability_alpha_mode,
                    "--capability-alpha-neighbors", str(args.capability_alpha_neighbors),
                    "--capability-batch-size", str(args.capability_batch_size)]
            if sibling is not None:
                argv += ["--e-star-from", str(sibling)]
            if args.secondary_mmlu:
                argv += ["--secondary-mmlu"]
            if args.capability_max_new_tokens is not None:
                argv += ["--capability-max-new-tokens", str(args.capability_max_new_tokens)]
            if args.resample:
                argv += ["--resample-run", str(run)]  # S10: real per-run data-level resample
            cell_alphas = (args.alphas if args.alphas is not None
                           else MODEL_ALPHAS.get(model, DEFAULT_ALPHAS))
            argv += ["--alphas"] + [str(a) for a in cell_alphas]
            rc, elapsed = run_cell(tag, argv, logf, args.gpu)
            if rc == 0:
                ran += 1
                observed.setdefault((model, scheme), []).append(elapsed)
            else:
                failed += 1; failed_tags.append(tag)   # non-fatal: keep going, re-launch later
            remaining = [(c[0], c[1]) for i, c in shard_cells
                         if i > idx and not done_start.get(i, False)]
            eta = eta_estimate(observed, remaining)
            print(eta, end="", flush=True); logf.write(eta)

        summary = ["\n" + "=" * 78, "  MATRIX SHARD COMPLETE",
                   f"  ran: {ran}   skipped(done): {skipped}   failed: {failed}   "
                   f"pending-on-dependency: {pending_dep}",
                   f"  shard: {shard_i}/{shard_n}   log: {log_path}"]
        if failed_tags:
            summary.append("  failed cells (re-launch to retry):")
            summary += [f"    {t}" for t in failed_tags]
        if pending_dep_tags:
            summary.append("  pending-on-dependency cells (NOT failures; waiting on their "
                           "fp16 sibling's COMPLETE file):")
            summary += [f"    {t}" for t in pending_dep_tags]
        remaining = failed + pending_dep  # still-missing cells for this shard
        summary.append(f"  -> re-run the SAME command to retry {remaining} missing cell(s); "
                       f"done cells will skip.")
        summary.append("=" * 78 + "\n")
        block = "\n".join(summary)
        print(block, flush=True); logf.write(block)


if __name__ == "__main__":
    main()
