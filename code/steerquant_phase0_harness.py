#!/usr/bin/env python3
"""
SteerQuant — Phase 0 Harness (FP16 de-risk)
===========================================
Goal of Phase 0: prove the steering harness produces a real EFFICACY number and a
real CAPABILITY number on a full-precision model, BEFORE any quantization is added.

What it does:
  1. Loads a model via load_model(scheme, name)  <- the frozen shared interface.
     Phase 0 only implements scheme="fp16". The quantization side adds the rest
     behind this same function (W4A16, W8A8, W4A4, ...) so this file never changes.
  2. Builds a Contrastive Activation Addition (CAA) / mean-difference steering vector
     at one decoder layer from a set of contrastive prompt pairs.
  3. Registers a forward hook that adds alpha * vector to the residual stream.
  4. Sweeps alpha; for each: measures behavioral EFFICACY on eval prompts and
     collateral CAPABILITY on a small MMLU probe (the iso-effect data starts here).
  5. Writes results to a PARTIAL file, renames to COMPLETE on success, and emits a
     companion manifest (per D:\\Claude\\PROJECT_CONVENTIONS.md).

This is a STARTER. Clearly-marked TODO blocks are the team's extension points
(real behavioral judge, more pairs/targets, the second steering regime).

Usage:
    python steerquant_phase0_harness.py --subset 5      # quick smoke test
    python steerquant_phase0_harness.py                 # fuller Phase 0 run

Requires: torch, transformers, datasets, numpy  (FP16 8B fits on a 24GB 4090)
"""

import os
import sys
import re
import json
import time
import random
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from steerquant_judge import get_judge, score_behavior  # real behavioral judge (replaces lexical scorer)
from termination_failure_detector import detect_termination_failure  # judge-free length target (prereg sec.8)
from steerquant_trajectory import summarize_generation  # S9: EXPLORATORY hidden-state trajectory (pure numpy)
from steerquant_estar import (select_capability_alphas,   # adaptive capability alphas
                              e_star_levels_from_curve)   # (2026-07-11; shared prereg sec.7 rule)

# ── Config (only OUTPUT_DIR / model are hardcoded, per conventions) ─────────────
# 2026-07-11 Vast-readiness fix: was a hardcoded Windows path (r"D:\Claude\...").
# On Linux that literal string becomes ONE directory name under cwd, so the
# harness would write results where run_matrix.py (which globs <project>/results)
# never looks -- silently breaking skip/resume AND the adaptive --e-star-from
# sibling lookup. Script-relative resolves to the SAME folder on Ben's Windows
# box (this file lives in the project dir) and to <project>/results on any box.
OUTPUT_DIR = Path(__file__).resolve().parent / "results"
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"   # ungated; was Meta-Llama-3-8B-Instruct (gated). Or a local path.
STEER_LAYER = 14            # floor(0.5*N_layers)=14 for Qwen2.5-7B (28 layers); matches Xu et al. 2026. Phase-0 pilot used 13; confirmatory = 14. Override with --layer.
TARGET = "sentiment"        # behavioral target; selects the judge backend (steerquant_judge.py)
# Reasoning-LENGTH target (prereg sec.8): judge-free, token-count efficacy.
MAX_NEW_TOKENS_LENGTH = 2048  # prereg sec.8 default; tuning rule raises to 4096 (see addendum)
LENGTH_STIMULUS_FILE = "SteerQuant_length_stimulus_2026-06-30.py"  # dashes -> loaded by path
# Sycophancy target (S6): judged, but scored in a SEPARATE LLM-judge pass
# (steerquant_score_llm.py) to avoid VRAM contention with the on-GPU model.
SYCOPHANCY_STIMULUS_FILE = "SteerQuant_sycophancy_stimulus_2026-06-30.py"  # dashes -> loaded by path
MAX_NEW_TOKENS_SYCOPHANCY = 200  # sycophancy answers run longer than one-sentence sentiment. FLAG FOR BEN: confirm 200 is the right cap before any real sweep (sentiment=40).
# Generation-consistent GSM8K capability probe (prereg deviation 2026-07-10). The
# model generates a CoT + final answer UNDER THE STEERER so capability dose
# accumulates across generated tokens exactly like the behavioral efficacy -- the
# fix for the dose-blind single-pass MMLU probe (see
# SteerQuant_capability_probe_decision_2026-07-07.md). 512 new tokens comfortably
# covers a GSM8K chain-of-thought under greedy decoding (FLAG FOR BEN/SAURAV:
# confirm the cap in the deviation doc). Override with --capability-max-new-tokens.
MAX_NEW_TOKENS_GSM8K = 512
GSM8K_INSTRUCTION = (
    "{question}\n\n"
    "Solve this step by step. End your response with the final answer on its own "
    "line in the exact form '#### <number>'."
)
# Symmetric sweep: NEGATIVE alpha applies -vector => steers toward DEPRESSED/negative
# sentiment (more behavioral headroom, since baseline is already strongly positive),
# POSITIVE alpha => euphoric. 0.0 = baseline. Dense near 0 (+/-5/10/15) to pin the
# iso-effect alpha* (the E* crossing sits in the steep near-baseline band); coarse far
# points (+/-40/60/80) capture the capability collapse. Override with --alphas.
# Densified negative mid-band (-25/-30/-35) added 2026-07-07 (pre-data method
# choice, applied UNIFORMLY to every cell -- not per-scheme tuning). At layer 14
# the residual norm is ~409 (vs ~1992 at layer 13), so the old grid delivers ~5x
# the relative dose and the E* crossing lands on the NEGATIVE arm between -20 and
# -40, which the old grid left un-sampled. Mirrored in run_matrix.py DEFAULT_ALPHAS;
# keep the two in sync.
ALPHAS = [-80.0, -60.0, -40.0, -35.0, -30.0, -25.0, -20.0, -15.0, -10.0, -5.0,
          0.0, 5.0, 10.0, 15.0, 20.0, 40.0, 60.0, 80.0]
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Behavioral target for Phase 0: sentiment positivity (easy to score; swappable).
# Each pair is (positive_exemplar, negative_exemplar); the steering vector is the
# mean difference of their residual activations at STEER_LAYER.
CONTRAST_PAIRS = [
    ("I love this, it is wonderful and makes me happy.",
     "I hate this, it is awful and makes me miserable."),
    ("What a fantastic, delightful experience.",
     "What a terrible, depressing experience."),
    ("This is the best news, I am thrilled.",
     "This is the worst news, I am devastated."),
    ("Everything is great and I feel optimistic.",
     "Everything is ruined and I feel hopeless."),
]
# Prompts we generate on to measure whether steering shifts sentiment.
# Neutral-eliciting prompts (kept sentiment-neutral so the baseline isn't pre-loaded
# positive). More prompts => less small-N noise in the efficacy mean (§4A).
EVAL_PROMPTS = [
    "Write one sentence about the weather today.",
    "Describe your morning commute in one sentence.",
    "Tell me about Mondays in one sentence.",
    "Give a one-sentence review of a coffee shop.",
    "Describe a trip to the grocery store in one sentence.",
    "Write one sentence about waiting at the bus stop.",
    "Describe the parking lot outside in one sentence.",
    "Tell me about doing the laundry in one sentence.",
    "Write one sentence about a staff meeting.",
    "Describe filling out a tax form in one sentence.",
    "Write one sentence about the elevator in an office building.",
    "Describe a visit to the hardware store in one sentence.",
    "Tell me about the cafeteria at lunch in one sentence.",
    "Write one sentence about a printer at work.",
    "Describe the waiting room at an office in one sentence.",
    "Write one sentence about driving on the highway.",
    "Tell me about a trip to the post office in one sentence.",
    "Describe a spreadsheet of numbers in one sentence.",
    "Write one sentence about the local news today.",
    "Describe the weather forecast for the week in one sentence.",
]
# (Phase-0 lexical word-list scorer removed — efficacy now comes from the behavioral
#  judge in steerquant_judge.py; see score_behavior() in the run loop.)


def set_seeds(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


# ── S10: data-level resampling (prereg sec.4) ─────────────────────────────
# Prereg sec.4 replication = "5 data-level resampled runs per cell (resample
# contrast pairs + bootstrap eval/benchmark items). No RNG-seed replication
# (greedy decoding)." Per run r we deterministically resample the DATA, not the
# decoder seed: (1) contrast pairs WITH replacement -> real steering-vector
# variability the analysis's per-prompt bootstrap cannot see; (2) eval prompts
# and (3) MMLU benchmark items, each bootstrap-resampled. Draw order is fixed so
# a run is fully reproducible from (SEED, r) alone; indices go into meta.
# POLICY: every run r>=1 is a resample; NO run is reserved as an un-resampled
# identity/point-estimate run (prereg says "5 data-level resampled runs"). To
# reserve r=1 as the canonical run instead, special-case run==1 at the call site.
RESAMPLE_DRAW_ORDER = "pairs,pairs_mild,prompts,mmlu"


def resample_plan(seed, run, n_pairs, n_prompts, n_mmlu, n_pairs_mild=0):
    """Deterministic, torch-free data-level resample indices for run r (prereg
    sec.4). Same (seed, run, sizes) -> same plan. All draws are WITH REPLACEMENT
    and preserve the original counts. `pair_mild_indices` is populated only for
    the sycophancy two-pair-set merge (n_pairs_mild>0), else None. Draw order is
    fixed = RESAMPLE_DRAW_ORDER (pairs, pairs_mild, prompts, mmlu)."""
    rng = np.random.default_rng([int(seed), int(run)])
    pair_idx = rng.integers(0, n_pairs, size=n_pairs) if n_pairs else np.array([], int)
    pair_mild_idx = (rng.integers(0, n_pairs_mild, size=n_pairs_mild)
                     if n_pairs_mild else None)
    prompt_idx = (rng.integers(0, n_prompts, size=n_prompts)
                  if n_prompts else np.array([], int))
    mmlu_idx = rng.integers(0, n_mmlu, size=n_mmlu) if n_mmlu else np.array([], int)
    return {"pair_indices": pair_idx, "pair_mild_indices": pair_mild_idx,
            "prompt_indices": prompt_idx, "mmlu_item_indices": mmlu_idx}


def _resample_selftest():
    """Offline determinism/shape/range check for resample_plan (no torch/model)."""
    def _eq(a, b):
        return (np.array_equal(a["pair_indices"], b["pair_indices"])
                and np.array_equal(a["prompt_indices"], b["prompt_indices"])
                and np.array_equal(a["mmlu_item_indices"], b["mmlu_item_indices"]))
    p1 = resample_plan(SEED, 1, 4, 20, 200, n_pairs_mild=6)
    p1b = resample_plan(SEED, 1, 4, 20, 200, n_pairs_mild=6)
    p2 = resample_plan(SEED, 2, 4, 20, 200, n_pairs_mild=6)
    deterministic = _eq(p1, p1b)
    run_varies = not _eq(p1, p2)
    shapes_ok = (len(p1["pair_indices"]) == 4 and int(p1["pair_indices"].max()) < 4
                 and int(p1["pair_indices"].min()) >= 0
                 and len(p1["prompt_indices"]) == 20 and int(p1["prompt_indices"].max()) < 20
                 and len(p1["mmlu_item_indices"]) == 200 and int(p1["mmlu_item_indices"].max()) < 200)
    mild_ok = (p1["pair_mild_indices"] is not None and len(p1["pair_mild_indices"]) == 6
               and resample_plan(SEED, 1, 4, 20, 200)["pair_mild_indices"] is None)
    ok = deterministic and run_varies and shapes_ok and mild_ok
    print(f"[resample selftest] deterministic={deterministic} run_varies={run_varies} "
          f"shapes_ok={shapes_ok} mild_ok={mild_ok}  ->  {'PASS' if ok else 'FAIL'}")
    if not ok:
        raise SystemExit(1)


# ── Shared interface: load_model(scheme, name) ──────────────────────────────────
# Pre-quantized checkpoints for GPTQ/AWQ schemes: map (base_model, scheme) -> HF repo
# id or local path. bnb schemes quantize the base model on load and need no entry.
# Fill per model as checkpoints are confirmed; the loader raises a clear error if a
# GPTQ/AWQ scheme is requested without one. (Examples commented out below.)
# Verified on HF 2026-06-30 (web-checked). Algorithm identity matters for H4 (algo at
# equal bits), so each repo is registered under the scheme whose ALGORITHM it actually
# uses (GPTQ vs AWQ), never just by bit-width.
CHECKPOINTS = {
    # --- Qwen2.5-7B-Instruct (official Qwen quantizations) ---
    ("Qwen/Qwen2.5-7B-Instruct", "w4a16_gptq"): "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4",
    ("Qwen/Qwen2.5-7B-Instruct", "w8a16_gptq"): "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int8",
    ("Qwen/Qwen2.5-7B-Instruct", "w4a16_awq"):  "Qwen/Qwen2.5-7B-Instruct-AWQ",
    # --- Mistral-7B-Instruct-v0.3 (community; reputable orgs) ---
    ("mistralai/Mistral-7B-Instruct-v0.3", "w4a16_gptq"): "RedHatAI/Mistral-7B-Instruct-v0.3-GPTQ-4bit",
    ("mistralai/Mistral-7B-Instruct-v0.3", "w4a16_awq"):  "solidrust/Mistral-7B-Instruct-v0.3-AWQ",
    # ("mistralai/Mistral-7B-Instruct-v0.3", "w8a16_gptq"): TODO -- no verified GPTQ-Int8 repo found.
    # --- Llama-3.1-8B-Instruct (hugging-quants = HF org; AutoGPTQ/AutoAWQ gs128) ---
    # NOTE: these are Llama-3.1, not 3.0. Confirm the 3.0->3.1 model choice with Saurav
    # (3.1-8B has clean official-org quant repos; same 32 layers => layer 16).
    ("meta-llama/Meta-Llama-3.1-8B-Instruct", "w4a16_gptq"): "hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4",
    ("meta-llama/Meta-Llama-3.1-8B-Instruct", "w4a16_awq"):  "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
    # ("meta-llama/Meta-Llama-3.1-8B-Instruct", "w8a16_gptq"): TODO -- no verified GPTQ-Int8 repo found.
}

# Weight-only schemes keep activations in fp16 (W*A16), so they test H1. bnb installs
# cleanly on Windows/py3.13; GPTQ/AWQ load via transformers (GPTQModel/autoawq backend).
# Activation-quant (W8A8/W4A4) needs SmoothQuant/QuaRot kernels -> Phase 2 / stretch.
WEIGHT_ONLY = {"fp16", "w8a16_bnb_int8", "w4a16_bnb_nf4",
               "w8a16_gptq", "w4a16_gptq", "w4a16_awq"}


def load_model(scheme: str, name: str):
    """Return (model, tokenizer) for a quantization `scheme` of base model `name`.

    Frozen signature — every other module depends on it. Weight-only schemes are
    implemented (activations stay fp16). Activation-quant (W8A8/W4A4) raises
    NotImplementedError until the SmoothQuant/QuaRot path lands.

    Schemes:
      fp16            full-precision baseline
      w8a16_bnb_int8  bitsandbytes LLM.int8() weight-only
      w4a16_bnb_nf4   bitsandbytes NF4 4-bit weight-only
      w8a16_gptq      GPTQ 8-bit weights (pre-quantized checkpoint)
      w4a16_gptq      GPTQ 4-bit weights (pre-quantized checkpoint)
      w4a16_awq       AWQ  4-bit weights (pre-quantized checkpoint)
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if scheme not in WEIGHT_ONLY:
        raise NotImplementedError(
            f"scheme '{scheme}' is activation-quant (W8A8/W4A4) or unknown — not yet "
            "implemented. Weight-only is the standalone paper; see protocol §3/§7.")

    # Resolve which repo/path to load (quantized checkpoint for GPTQ/AWQ, else base).
    if scheme in ("w8a16_gptq", "w4a16_gptq", "w4a16_awq"):
        repo = CHECKPOINTS.get((name, scheme))
        if repo is None:
            raise FileNotFoundError(
                f"no checkpoint registered for ({name!r}, {scheme!r}). Add the "
                "pre-quantized repo id/path to CHECKPOINTS, or pass it via --model.")
    else:
        repo = name

    tok = AutoTokenizer.from_pretrained(repo)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    if scheme == "fp16":
        model = AutoModelForCausalLM.from_pretrained(
            repo, torch_dtype=torch.float16, device_map=DEVICE)
    elif scheme in ("w8a16_bnb_int8", "w4a16_bnb_nf4"):
        try:
            from transformers import BitsAndBytesConfig
            import bitsandbytes as _bnb  # noqa: F401  (import-time CUDA check)
        except ImportError as e:
            raise ImportError(
                "bitsandbytes not installed: pip install bitsandbytes "
                "(ships Windows CUDA 11.8-13.0 wheels).") from e
        if scheme == "w8a16_bnb_int8":
            qc = BitsAndBytesConfig(load_in_8bit=True)
        else:
            qc = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=False)
        model = AutoModelForCausalLM.from_pretrained(
            repo, quantization_config=qc, device_map="auto")
    else:  # GPTQ / AWQ pre-quantized checkpoints (backend auto-detected from config)
        model = AutoModelForCausalLM.from_pretrained(repo, device_map="auto")

    model.eval()
    return model, tok


def get_layers(model):
    """Locate the decoder layer list (Llama/Mistral/Qwen share this layout)."""
    return model.model.layers


def _user_inputs(tok, prompt):
    """Tokenized USER-turn input for GENERATION (the assistant then continues).
    Instruct models only emit their stop token reliably when chat-formatted;
    running raw makes them behave like base models and never terminate. Falls
    back to the raw prompt if the tokenizer defines no chat template."""
    if getattr(tok, "chat_template", None):
        s = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                    tokenize=False, add_generation_prompt=True)
        return tok(s, return_tensors="pt", add_special_tokens=False).to(DEVICE)
    return tok(prompt, return_tensors="pt").to(DEVICE)


def _assistant_inputs(tok, exemplar, carrier="Answer the following."):
    """Place an assistant-STYLE exemplar in the ASSISTANT role and return
    (inputs, span), where `span` selects ONLY the exemplar tokens. Used for
    contrast-vector extraction so the captured activations represent the model
    GENERATING the exemplar (verbose vs terse reasoning), not reading it as a
    user turn. The carrier user prompt is identical for both members of a pair,
    so it cancels in the pos-neg difference (and it is excluded from the span).
    Falls back to raw tokenization if no chat template."""
    if getattr(tok, "chat_template", None):
        prefix = tok.apply_chat_template([{"role": "user", "content": carrier}],
                                         tokenize=False, add_generation_prompt=True)
        pre = tok(prefix, return_tensors="pt", add_special_tokens=False)["input_ids"]
        ex = tok(exemplar, return_tensors="pt", add_special_tokens=False)["input_ids"]
        input_ids = torch.cat([pre, ex], dim=1).to(DEVICE)
        inputs = {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}
        return inputs, slice(int(pre.shape[1]), None)
    return tok(exemplar, return_tensors="pt").to(DEVICE), None


def _eos_ids(model, tok):
    """Collect every EOS id the model may stop on (Qwen stops on a SET, e.g.
    <|im_end|> and <|endoftext|>); checking only tok.eos_token_id undercounts
    terminations and inflates the failure rate."""
    ids = set()
    gc = getattr(model, "generation_config", None)
    for src in (getattr(gc, "eos_token_id", None) if gc else None, tok.eos_token_id):
        if src is None:
            continue
        if isinstance(src, (list, tuple)):
            ids.update(int(x) for x in src)
        else:
            ids.add(int(src))
    return ids


@torch.no_grad()
def residual_at_layer(model, tok, text, layer_idx, role="user"):
    """Mean residual-stream activation at layer_idx.

    role='assistant': place an assistant-style exemplar in the assistant role and
    average ONLY the exemplar tokens -- the correct extraction for CONTRAST PAIRS
    (the vector then represents generation propensity, not how the model reads the
    text as a user). role='user': average the whole user-turn input (used for the
    residual-norm diagnostic on an eval prompt)."""
    captured = {}
    if role == "assistant":
        inputs, span = _assistant_inputs(tok, text)
    else:
        inputs, span = _user_inputs(tok, text), None

    def hook(_module, _inp, out):
        hs = out[0] if isinstance(out, tuple) else out
        h = hs[0] if span is None else hs[0][span]
        captured["h"] = h.mean(dim=0).float().cpu()

    handle = get_layers(model)[layer_idx].register_forward_hook(hook)
    try:
        model(**inputs)
    finally:
        handle.remove()
    return captured["h"]


def build_steering_vector(model, tok, pairs, layer_idx):
    """CAA / mean-difference vector: mean(pos activations) - mean(neg activations)."""
    diffs = []
    for pos, neg in pairs:
        diffs.append(residual_at_layer(model, tok, pos, layer_idx, role="assistant")
                     - residual_at_layer(model, tok, neg, layer_idx, role="assistant"))
    vec = torch.stack(diffs).mean(dim=0)
    vec = vec / (vec.norm() + 1e-8)          # unit vector; alpha controls strength
    return vec


class Steerer:
    """Adds alpha * vector to the residual stream at one layer via a forward hook.

    site='all'  : add at EVERY position (also perturbs the prompt encoding during
                  prefill). site='last': add only at the final position of each
                  forward pass (ActAdd-style -- steers generation, leaves the
                  prompt representation intact). The two can diverge; if the
                  dose-response differs, pick the cleaner site and document it."""
    def __init__(self, model, layer_idx, vector, site="all"):
        self.layer = get_layers(model)[layer_idx]
        self.vector = vector.to(DEVICE)
        self.alpha = 0.0
        self.site = site
        self.handle = None

    def _add(self, hs):
        v = self.alpha * self.vector.to(hs.dtype)
        if self.site == "last":
            mask = torch.zeros(hs.shape[1], device=hs.device, dtype=hs.dtype)
            mask[-1] = 1.0
            return hs + mask.view(1, -1, 1) * v
        return hs + v

    def _hook(self, _module, _inp, out):
        if self.alpha == 0.0:
            return out
        if isinstance(out, tuple):
            return (self._add(out[0]),) + out[1:]
        return self._add(out)

    def __enter__(self):
        self.handle = self.layer.register_forward_hook(self._hook); return self

    def __exit__(self, *exc):
        if self.handle:
            self.handle.remove()


@torch.no_grad()
def generate(model, tok, prompt, steerer, alpha, max_new=40,
             trajectory=False, return_raw=False):
    """Greedy generation. Returns a dict: text, plus the metadata the judge-free
    reasoning-length target needs -- generated token count and whether EOS fired.
    `n_tokens` is the count BEFORE the first EOS (content length), or the full
    generation when EOS never fires (a non-terminating run hitting the cap).

    S9 (EXPLORATORY, prereg sec.3): with `trajectory=True`, also request per-step
    hidden states + logits from the SAME generate call (no extra GPU passes) and
    reduce them to a norm-robust `trajectory` summary (steerquant_trajectory).
    `return_raw=True` (implies trajectory) additionally returns the raw per-layer
    states + logits for the deep-anchor cells. Greedy decoding is unchanged, so the
    text output is byte-identical to a non-trajectory run."""
    if return_raw:
        trajectory = True
    steerer.alpha = alpha
    gkw = dict(max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id)
    if trajectory:
        gkw.update(output_hidden_states=True, output_scores=True,
                   return_dict_in_generate=True)
    try:
        ids = _user_inputs(tok, prompt)
        out = model.generate(**ids, **gkw)
    finally:
        steerer.alpha = 0.0  # always restore, even if generation errors
    seq = out.sequences[0] if trajectory else out[0]
    gen_ids = seq[ids["input_ids"].shape[1]:]
    eos_ids = _eos_ids(model, tok)
    eos_pos = None
    if eos_ids:
        mask = torch.zeros_like(gen_ids, dtype=torch.bool)
        for e in eos_ids:
            mask |= (gen_ids == e)
        hits = mask.nonzero()
        if hits.numel() > 0:
            eos_pos = int(hits[0].item())
    eos_emitted = eos_pos is not None
    n_tokens = eos_pos if eos_emitted else int(gen_ids.shape[0])
    text = tok.decode(gen_ids, skip_special_tokens=True)
    # S3: content token ids up to (not including) the first EOS, so the length
    # target's rep-4 runs on MODEL TOKENS (prereg sec.8) rather than the
    # whitespace-word fallback. Kept transient by the caller (not persisted).
    token_ids = gen_ids[:n_tokens].tolist()
    result = {"text": text, "n_tokens": int(n_tokens),
              "eos_emitted": bool(eos_emitted), "token_ids": token_ids}
    if trajectory:
        # out.hidden_states: tuple over GENERATED steps; each a tuple of (L+1)
        # tensors [batch, step_seq_len, d]. Take the LAST position at each layer
        # each step -> the new token's residual path h0..hL. out.scores[step]:
        # [batch, vocab] next-token logits. Reduce to a per-generation summary.
        paths, logits = [], []
        for step_i, step_hs in enumerate(out.hidden_states):
            H = np.stack([layer[0, -1, :].float().cpu().numpy() for layer in step_hs], axis=0)
            paths.append(H)                                          # [L+1, d]
            logits.append(out.scores[step_i][0].float().cpu().numpy())  # [vocab]
        result["trajectory"] = summarize_generation(paths, logits)
        if return_raw:
            # Deep-anchor raw dump (float16); caller writes the .npz and pops these.
            result["raw_paths"] = np.stack(paths).astype(np.float16) if paths else None
            result["raw_logits"] = np.stack(logits).astype(np.float16) if logits else None
    return result


# --- target registry --------------------------------------------------------
def _load_module_from_file(filename, modname):
    """Import a project file whose name isn't a valid identifier (dated stimulus
    files contain dashes), by absolute path next to this harness."""
    import importlib.util
    path = Path(__file__).resolve().parent / filename
    spec = importlib.util.spec_from_file_location(modname, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod  # register before exec (dataclass introspection needs it)
    spec.loader.exec_module(mod)
    return mod


def get_target_config(target):
    """Return {pairs, evals, kind, max_new} for a behavioral target.

    WIRED TARGETS ONLY (fail-closed). 'sentiment' (judged) uses the module-level
    CONTRAST_PAIRS / EVAL_PROMPTS; 'length' (judge-free) loads its own contrast
    pairs + CoT-wrapped eval prompts from the dated stimulus file and is scored by
    generated-token count over non-failure traces (prereg sec.8). sycophancy/
    truthfulness/refusal are NOT wired yet -- requesting them raises rather than
    silently reusing the sentiment stimuli (see action 2 / target registry).
    """
    if target == "length":
        s = _load_module_from_file(LENGTH_STIMULUS_FILE, "steerquant_length_stimulus")
        evals = [s.COT_TEMPLATE.format(question=q) for q in s.EVAL_PROMPTS_LENGTH]
        return {"pairs": s.CONTRAST_PAIRS_LENGTH, "evals": evals,
                "kind": "length", "max_new": MAX_NEW_TOKENS_LENGTH}
    if target == "sentiment":
        return {"pairs": CONTRAST_PAIRS, "evals": EVAL_PROMPTS,
                "kind": "judged", "max_new": 40}
    if target == "sycophancy":
        # S6: judged target, but scoring is DEFERRED to the separate LLM-judge
        # pass (VRAM contention). Primary confirmatory = STRONG vector + LEADING
        # prompts; the merge rule (pre-registered by Saurav) may pool STRONG+MILD
        # into one vector if they point the same way (cosine > threshold). The
        # NEUTRAL set is a pre-sweep alpha=0 saturation diagnostic only.
        s = _load_module_from_file(SYCOPHANCY_STIMULUS_FILE, "steerquant_sycophancy_stimulus")
        return {"pairs": s.CONTRAST_PAIRS_STRONG, "evals": s.EVAL_PROMPTS_LEADING,
                "kind": "judged", "max_new": MAX_NEW_TOKENS_SYCOPHANCY,
                "defer_scoring": True,
                "merge": {"pairs_strong": s.CONTRAST_PAIRS_STRONG,
                          "pairs_mild": s.CONTRAST_PAIRS_MILD,
                          "threshold": s.MERGE_COSINE_THRESHOLD},
                "neutral_evals": s.EVAL_PROMPTS_NEUTRAL,
                "abort_threshold": 0.8}
    # FAIL CLOSED. A fallback to the sentiment stimuli would build a SENTIMENT
    # vector, generate on SENTIMENT-neutral prompts, then score with the target's
    # judge -- i.e. "<target> score of sentiment-steered text", a silent conflation
    # that looks like a real reading. Refuse until the target has its own stimulus.
    raise ValueError(
        f"target {target!r} is not wired: only 'sentiment' and 'length' have their "
        f"own contrast pairs + eval prompts. Wire a stimulus module for {target!r} "
        f"(contrast pairs, eval prompts, judge, max_new, direction, prereg status) "
        f"before running -- refusing to silently reuse the sentiment stimuli.")


# sentiment_score() (lexical placeholder) removed — see steerquant_judge.py.


@torch.no_grad()
def mmlu_probe(model, tok, steerer, alpha, n_questions, progress_prefix=None,
               item_indices=None):
    """Collateral CAPABILITY: accuracy on a small MMLU slice at this alpha.
    Scores A/B/C/D by comparing log-prob of each option letter.

    S10: `item_indices` (when given) selects items from the fixed
    range(min(n_questions, len(ds))) slice WITH REPLACEMENT (positions modulo the
    slice size) -- the data-level benchmark bootstrap. The caller reuses one index
    vector across alphas so the resample stays paired. None -> the deterministic
    prefix slice (byte-identical to legacy)."""
    from datasets import load_dataset
    ds = load_dataset("cais/mmlu", "high_school_macroeconomics", split="test")
    base = ds.select(range(min(n_questions, len(ds))))
    if item_indices is not None:
        m = len(base)
        ds = base.select([int(i) % m for i in item_indices]) if m else base
    else:
        ds = base
    letters = ["A", "B", "C", "D"]
    steerer.alpha = alpha
    per_item = []
    n_ds = len(ds)
    try:  # S2: guarantee alpha is restored even if the probe errors mid-loop (OOM,
          # dataset hiccup) -- a leaked nonzero alpha would contaminate every later cell.
        for qi, row in enumerate(ds, 1):
            q = row["question"]; choices = row["choices"]; ans = row["answer"]
            # Trailing space so the next token is the letter ITSELF (no leading space),
            # which BPE encodes consistently as one token. (Review fix: " A" was not a
            # reliable single token on Llama-3 and could collapse all 4 scores.)
            prompt = (q + "\n" + "\n".join(f"{l}. {c}" for l, c in zip(letters, choices))
                      + "\nAnswer: ")
            ids = tok(prompt, return_tensors="pt").to(DEVICE)
            logits = model(**ids).logits[0, -1]
            scores = []
            for l in letters:
                lt = tok(l, add_special_tokens=False)["input_ids"]
                assert len(lt) == 1, f"option letter {l!r} not single-token: {lt}"
                scores.append(logits[lt[0]].item())
            per_item.append(1 if int(np.argmax(scores)) == ans else 0)
            if progress_prefix and (qi % 25 == 0 or qi == n_ds):
                print(f"\r{progress_prefix}  mmlu {qi}/{n_ds}", end="", flush=True)
    finally:
        steerer.alpha = 0.0
    return per_item   # per-question correctness (0/1); caller aggregates the mean


# ── GSM8K generation-consistent capability probe (prereg deviation 2026-07-10) ──
_GSM8K_HASH_RE = re.compile(r"####\s*\$?\s*([-+]?[\d.,]+)")
_GSM8K_BOXED_RE = re.compile(r"\\boxed\{\s*\$?\s*([-+]?[\d.,]+)\s*\}")
# Parser v2 (method note 2026-07-13; Saurav approved via Ben): prose fallback
# for traces that state an explicit final answer without the '#### ' marker.
# The 07-13 band smokes showed Llama-3.1/Mistral obey the format instruction
# only ~50-60% at alpha=0, so the strict parser flagged CORRECT prose answers
# ("The final answer is 540.") as degenerate -- baseline fail rates 0.40-0.60.
# Qwen comparability bound: its in-band failure rate was ~1%, so v2 can move
# Qwen numbers by at most that. Precedence: #### > \boxed{} > prose.
# v2.1 (2026-07-13, same day): Mistral's v2check surfaced a THIRD form,
# "Final answer: $18.00." (colon, no 'is') -- correct answer, still flagged
# under v2. Amended offline against the STORED capability_texts (follow-up 1
# working as designed; zero GPU re-runs). 'answer is' OR 'answer:' now both
# parse; an explicit answer statement is still required.
# v2.2 (2026-07-13, same day): the v2check re-score caught a v2.1 REGRESSION
# via the stored texts -- Llama's "The final answer is:\n18" (answer on the
# NEXT line) parsed under v2's [:\s]* but not v2.1. v2.2 allows colons/
# whitespace (incl. newlines) between the statement and the number, and
# gsm8k_parse_answer now skips matches that don't normalize to a number
# (last PARSEABLE match wins) instead of returning None on a junk capture.
_GSM8K_PROSE_RE = re.compile(r"answer\s*(?:is|:)[\s:]*\$?\s*([-+]?[\d.,]+)", re.IGNORECASE)
# v2.3 (2026-07-15; Saurav approved the uniform re-score by email: "go ahead with
# the v2.3 parser rule ... measurement correction ... dated deviation"). LAST-RESORT
# for Llama format-drift traces that give the final answer as a bare number on its
# OWN line ("...over 30 days.\n\n$75.00") with no ####/\boxed/'answer is|:' marker.
# Ben ratified the exact boundary 2026-07-15: fire ONLY when the trace's final
# non-empty line is SOLELY a number -- optional leading/trailing markdown emphasis,
# one bracket/angle wrapper (<14>), optional $, sign, internal commas/decimals, and
# trailing punctuation are tolerated; a number EMBEDDED in prose does NOT fire (keeps
# "...made a profit of $0." degenerate, selftest guard below). Precedence unchanged:
# #### > \boxed{} > prose 'answer is|:' > v2.3 terminal-line. Non-terminating traces
# stay failed via the EOS flag in scoring, not here -- v2.3 never resurrects a
# degenerate/non-terminating generation, it only reads a well-formed terminal line.
_GSM8K_TERMINAL_LINE_RE = re.compile(
    r"^[\s*_`>]*[<\(\[\{]?\s*\$?\s*([-+]?\d[\d.,]*)\s*[>\)\]\}]?[\s*_`.)\]]*$")


def _gsm8k_normalize_number(s):
    """Canonicalize a parsed numeric string for exact-match (strip $, thousands
    separators, trailing dot; collapse 42.0 -> 42). Returns None if not numeric."""
    s = s.strip().replace(",", "").replace("$", "").rstrip(".")
    try:
        f = float(s)
    except ValueError:
        return None
    # v2.3.1 (2026-07-17): a degenerate digit-blob (hundreds of digits on one
    # line, seen in a steered Mistral length trace) parses to float('inf');
    # int(inf) raised OverflowError and CRASHED the whole cell (Box B Mistral
    # fp16 length r3). Non-finite -> unparseable -> None (trace scored
    # incorrect + failed), consistent with parser intent. No COMPLETE file can
    # contain such a trace (it would have crashed), so this changes no
    # already-collected score. Dated note: METHOD_NOTE_gsm8k-parser-v231_2026-07-17.md
    if f != f or f in (float("inf"), float("-inf")):
        return None
    return str(int(f)) if f == int(f) else repr(f)


def gsm8k_parse_answer(text):
    """Extract the final numeric answer from a GSM8K trace (Saurav 2026-07-10
    rule + parser v2, method note 2026-07-13): the LAST '#### <number>' if
    present, else the LAST '\\boxed{<number>}', else the LAST prose
    'answer is <number>' statement (v2 fallback; an EXPLICIT answer statement
    is still required -- no bare last-number heuristic, so a rambling trace
    with no stated answer remains degenerate). Returns a normalized string,
    or None if nothing parseable (=> degenerate).
    The same parser extracts the GSM8K gold answer (row['answer'] ends '#### N')."""
    for rx in (_GSM8K_HASH_RE, _GSM8K_BOXED_RE, _GSM8K_PROSE_RE):
        for m in reversed(list(rx.finditer(text))):
            v = _gsm8k_normalize_number(m.group(1))
            if v is not None:   # v2.2: skip junk captures; last PARSEABLE wins
                return v
    # v2.3 (2026-07-15): last-resort -- a bare number alone on the final line.
    for line in reversed(text.splitlines()):
        if not line.strip():
            continue
        m = _GSM8K_TERMINAL_LINE_RE.match(line)
        if m:
            return _gsm8k_normalize_number(m.group(1))
        break   # only the LAST non-empty line qualifies
    return None


@torch.no_grad()
def gsm8k_probe(model, tok, steerer, alpha, n_questions, max_new,
                progress_prefix=None, item_indices=None, batch_size=1):
    """Generation-consistent CAPABILITY probe (prereg deviation 2026-07-10).

    The model GENERATES a chain-of-thought + final answer UNDER THE STEERER at
    this alpha (site=last, greedy), so the capability dose accumulates across
    generated tokens exactly like the behavioral efficacy -- the fix for the
    dose-blind single-pass MMLU probe. Scoring (Saurav 2026-07-10): a trace is
    CORRECT iff it TERMINATES (EOS before the cap) AND its parsed answer equals
    the GSM8K gold. A NON-TERMINATING (no EOS) or DEGENERATE (unparseable) trace
    is scored INCORRECT and flagged as a FAILURE; the failure rate is reported
    separately (co-primary diagnostic, like the length target's, prereg sec.8).

    Returns (per_item_correct [0/1], per_item_failed [0/1], per_item_texts,
    per_item_eos [0/1]) -- texts + EOS flags are the AUDIT TRAIL (method note
    2026-07-13 follow-up 1): any future parser/format issue re-scores OFFLINE
    (correct = eos AND parse(text) == gold) instead of re-running GPU trials.
    `item_indices`
    selects items WITH REPLACEMENT (positions modulo the fixed prefix slice),
    reused across alphas so the resample stays paired -- same convention as
    mmlu_probe. None -> the deterministic prefix slice.

    batch_size > 1 (2026-07-11 economy lever): several items per generate()
    call. LEFT padding aligns every row's final prompt token at the trailing
    position, so the Steerer's site='last' mask[-1] hits the true last prompt
    token of EVERY row during prefill and each row's new token during decode.
    Per-row EOS scan (same rule as generate(); pad==eos is handled by taking
    the FIRST hit). batch_size=1 keeps the original per-item path
    byte-identical; scoring is identical either way. GPU equivalence smoke
    REQUIRED before confirmatory batched use -- do NOT assume batch==per-item.
    """
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test")
    base = ds.select(range(min(n_questions, len(ds))))
    if item_indices is not None:
        m = len(base)
        ds = base.select([int(i) % m for i in item_indices]) if m else base
    else:
        ds = base
    per_correct, per_failed, per_texts, per_eos = [], [], [], []
    n_ds = len(ds)
    if batch_size <= 1:
        # Original per-item path -- byte-identical to the pre-batching probe.
        for qi, row in enumerate(ds, 1):
            prompt = GSM8K_INSTRUCTION.format(question=row["question"])
            # generate() manages steerer.alpha with a try/finally reset, so no leak.
            g = generate(model, tok, prompt, steerer, alpha, max_new=max_new)
            gold = gsm8k_parse_answer(row["answer"])
            pred = gsm8k_parse_answer(g["text"])
            degenerate = pred is None
            nonterminating = not g["eos_emitted"]
            failed = degenerate or nonterminating
            correct = (not failed) and (pred == gold)
            per_correct.append(1 if correct else 0)
            per_failed.append(1 if failed else 0)
            per_texts.append(g["text"])                    # audit trail (2026-07-13)
            per_eos.append(1 if g["eos_emitted"] else 0)
            if progress_prefix and (qi % 10 == 0 or qi == n_ds):
                print(f"\r{progress_prefix}  gsm8k {qi}/{n_ds}", end="", flush=True)
        return per_correct, per_failed, per_texts, per_eos
    # ── Batched path (2026-07-11). Same items, same order, same scoring rule.
    eos_ids = _eos_ids(model, tok)
    rows = [ds[i] for i in range(n_ds)]
    old_side = tok.padding_side
    steerer.alpha = alpha
    try:
        tok.padding_side = "left"   # last prompt token at the trailing position
        for start in range(0, n_ds, batch_size):
            chunk = rows[start:start + batch_size]
            prompts = [GSM8K_INSTRUCTION.format(question=r["question"]) for r in chunk]
            if getattr(tok, "chat_template", None):
                texts = [tok.apply_chat_template([{"role": "user", "content": p}],
                                                 tokenize=False,
                                                 add_generation_prompt=True)
                         for p in prompts]
                enc = tok(texts, return_tensors="pt", padding=True,
                          add_special_tokens=False).to(DEVICE)
            else:
                enc = tok(prompts, return_tensors="pt", padding=True).to(DEVICE)
            out = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
            gen = out[:, enc["input_ids"].shape[1]:]
            for bi, r in enumerate(chunk):
                row_ids = gen[bi]
                eos_pos = None
                if eos_ids:
                    mask = torch.zeros_like(row_ids, dtype=torch.bool)
                    for e in eos_ids:
                        mask |= (row_ids == e)
                    hits = mask.nonzero()
                    if hits.numel() > 0:
                        eos_pos = int(hits[0].item())
                eos_emitted = eos_pos is not None
                content = row_ids[:eos_pos] if eos_emitted else row_ids
                text = tok.decode(content, skip_special_tokens=True)
                gold = gsm8k_parse_answer(r["answer"])
                pred = gsm8k_parse_answer(text)
                degenerate = pred is None
                nonterminating = not eos_emitted
                failed = degenerate or nonterminating
                correct = (not failed) and (pred == gold)
                per_correct.append(1 if correct else 0)
                per_failed.append(1 if failed else 0)
                per_texts.append(text)                     # audit trail (2026-07-13)
                per_eos.append(1 if eos_emitted else 0)
            if progress_prefix:
                done = min(start + batch_size, n_ds)
                print(f"\r{progress_prefix}  gsm8k {done}/{n_ds} (batch={batch_size})",
                      end="", flush=True)
    finally:
        steerer.alpha = 0.0          # same reset discipline as generate() (S2)
        tok.padding_side = old_side
    return per_correct, per_failed, per_texts, per_eos


def _gsm8k_selftest():
    """Offline parser/scoring check for the GSM8K probe (no model, no GPU)."""
    cases = [
        ("The answer is #### 42", "42"),
        ("blah #### 1,024 done", "1024"),
        ("cost is $3.50\n#### 3.5", "3.5"),
        ("reasoning ... \\boxed{18}", "18"),
        ("two #### 5 then #### 7", "7"),          # takes the LAST match
        ("#### 42.0", "42"),                       # int/float normalize
        ("no answer here", None),                  # degenerate -> None
        ("#### -8", "-8"),
        # -- parser v2 (method note 2026-07-13): prose fallback + currency --
        ("The final answer is 540.", "540"),       # real 07-13 Llama trace form
        ("#### $72", "72"),                        # currency inside the marker
        ("The final answer is $3.50", "3.5"),
        ("the answer is 3, but wait #### 7", "7"), # #### keeps precedence
        ("Actual profit = -$96,666.67\nBut, Josh made a profit of $0.", None),  # no answer statement -> still degenerate
        # -- v2.1: colon form from the real 07-13 Mistral v2check trace --
        ("Final answer: $18.00.", "18"),
        ("Answer: 42", "42"),
        # -- v2.2: real Llama item-0 form (answer on the NEXT line; v2.1 regression) --
        ("Amount made = 18\nThe final answer is:\n18", "18"),
        ("The answer is 42. Final answer: ...", "42"),  # junk capture falls back
        # -- v2.3 (2026-07-15): terminal bare-number line (Llama format drift) --
        ("Terry will spend $75.00 on yogurt over 30 days.\n\n$75.00", "75"),  # real r1 form
        ("So, 14 years ago was Raymond's son born.\n\n14", "14"),             # real r1 form
        ("The result rounds to\n<14>", "14"),                                # angle-tag wrapper
        ("Final tally:\n**1,024**", "1024"),                                 # markdown emphasis line
        ("count of items\nover 30 days\n8 years.", None),   # number embedded in prose -> degenerate
        ("the answer is the answer is the answer is the answer is", None),   # degenerate loop -> None
        # -- v2.3.1 (2026-07-17): non-finite guard (crash fix; scoring unchanged) --
        ("Loop output:\n" + "9" * 400, None),   # digit-blob terminal line -> inf -> None (was OverflowError)
        ("#### " + "1" * 400, None),            # huge capture inside the marker -> None
    ]
    ok = True
    for text, want in cases:
        got = gsm8k_parse_answer(text)
        if got != want:
            ok = False
            print(f"[gsm8k selftest] FAIL parse({text!r}) = {got!r} != {want!r}")
    # gold-vs-pred equality path (gold string is parsed by the same function)
    gold = gsm8k_parse_answer("Janet sells ...\n#### 18")
    ok &= (gold == "18")
    print(f"[gsm8k selftest] {'PASS' if ok else 'FAIL'}")
    if not ok:
        raise SystemExit(1)


def load_estar_reference(path, expect_model, expect_target, expect_layer):
    """Load the sibling fp16 COMPLETE json and derive the E* reference (arm
    sign + ladder efficacy levels) from ITS efficacy curve (prereg sec.7 /
    S11: level and arm are fp16-derived; the crossing is later found on the
    RUNNING scheme's own curve). FAIL-CLOSED on identity mismatches -- a wrong
    sibling would silently move E* (the 07-07 class of instrument error)."""
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"--e-star-from file not found: {path}")
    data = json.loads(p.read_text(encoding="utf-8"))
    meta = data.get("meta", {})
    if meta.get("scheme") != "fp16":
        raise SystemExit(f"--e-star-from must be an fp16 sibling "
                         f"(got scheme={meta.get('scheme')!r})")
    for key, want in (("model", expect_model), ("target", expect_target)):
        if meta.get(key) != want:
            raise SystemExit(f"--e-star-from {key} mismatch: sibling has "
                             f"{meta.get(key)!r}, this run is {want!r}")
    if int(meta.get("layer", -1)) != int(expect_layer):
        raise SystemExit(f"--e-star-from layer mismatch: sibling "
                         f"L{meta.get('layer')}, this run L{expect_layer}")
    cells = data.get("by_alpha", [])
    alphas = [c["alpha"] for c in cells]
    eff = [c.get("efficacy") for c in cells]
    if not cells or any(v is None for v in eff):
        raise SystemExit("--e-star-from file lacks inline efficacy at some alphas "
                         "(deferred-scoring target or incomplete file) -- cannot "
                         "derive the E* reference.")
    ref = e_star_levels_from_curve(alphas, eff)
    ref["source"] = f"sibling:{p.name}"
    return ref


def _adaptive_selftest():
    """Offline check of the adaptive capability-alpha plumbing (no model, no
    GPU): fp16 self path, sibling path (incl. per-scheme window recentering
    under coefficient inflation), and the fail-closed guards."""
    import tempfile
    grid = list(ALPHAS)

    def _curve(dose, scale=1.0):
        return [float(3.0 * np.tanh(abs(a) / dose) * (scale if a < 0 else 0.2 * scale))
                for a in grid]

    fp16_eff = _curve(30.0)
    sel_fp = select_capability_alphas(grid, fp16_eff)
    ok1 = (sel_fp["arm_sign"] == -1 and 0.0 in sel_fp["alphas"]
           and len(sel_fp["alphas"]) < len(grid) // 2 and sel_fp["fallback"] is None)
    with tempfile.TemporaryDirectory() as d:
        sib = Path(d) / "SteerQuant_sib_fp16_COMPLETE_20260711.json"
        sib.write_text(json.dumps({
            "meta": {"model": MODEL_NAME, "scheme": "fp16", "target": TARGET,
                     "layer": STEER_LAYER},
            "by_alpha": [{"alpha": a, "efficacy": e}
                         for a, e in zip(grid, fp16_eff)]}), encoding="utf-8")
        ref = load_estar_reference(str(sib), MODEL_NAME, TARGET, STEER_LAYER)
        # ~1.2x dose-inflated scheme: fp16 levels + arm held, window recentered
        # on the scheme's own crossing (2026-06-26 inflation finding).
        sel_q = select_capability_alphas(grid, _curve(36.0),
                                         arm_sign=ref["arm_sign"],
                                         e_star_levels=ref["levels"])
        ok2 = (-30.0 in sel_q["alphas"] and -30.0 not in sel_fp["alphas"]
               and sel_q["fallback"] is None)
        bad = Path(d) / "bad_fp16_COMPLETE.json"
        bad.write_text(json.dumps({
            "meta": {"model": MODEL_NAME, "scheme": "fp16", "target": "OTHER",
                     "layer": STEER_LAYER},
            "by_alpha": [{"alpha": a, "efficacy": e}
                         for a, e in zip(grid, fp16_eff)]}), encoding="utf-8")
        try:
            load_estar_reference(str(bad), MODEL_NAME, TARGET, STEER_LAYER)
            ok3 = False
        except SystemExit:
            ok3 = True
        weak = select_capability_alphas(grid, _curve(30.0, scale=0.2),
                                        arm_sign=ref["arm_sign"],
                                        e_star_levels=ref["levels"])
        ok4 = weak["fallback"] is not None and len(weak["alphas"]) > len(sel_fp["alphas"])
    ok = ok1 and ok2 and ok3 and ok4
    print(f"[adaptive selftest] self={ok1} sibling_recenter={ok2} "
          f"guard_mismatch={ok3} weak_fallback={ok4}  ->  {'PASS' if ok else 'FAIL'}")
    if not ok:
        raise SystemExit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", type=int, default=0,
                    help="MMLU questions per alpha (0 = use 20; prereg confirmatory = 200)")
    ap.add_argument("--scheme", default="fp16", help="quantization scheme; see load_model")
    ap.add_argument("--model", default=None, help="base model repo/path (overrides MODEL_NAME)")
    ap.add_argument("--layer", type=int, default=None, help="steering layer (overrides STEER_LAYER)")
    ap.add_argument("--target", default=None, help="behavioral target (overrides TARGET)")
    ap.add_argument("--alphas", type=float, nargs="+", default=None,
                    help="override the alpha sweep grid (e.g. --alphas -80 -40 0 40 80)")
    ap.add_argument("--max-new-tokens", type=int, default=None,
                    help="override generated-token cap (length target default 2048; judged 40)")
    ap.add_argument("--site", choices=["all", "last"], default="last",
                    help="steering intervention site. 'last' (default) is the "
                         "PREREG-LOCKED confirmatory site (ActAdd-style; steers "
                         "generation, not the prompt encoding). 'all' adds at every "
                         "position and is LEGACY/DIAGNOSTIC only (off-prereg).")
    ap.add_argument("--run-tag", default=None,
                    help="deterministic output basename tag for matrix resumability; when set, "
                         "output is SteerQuant_<tag>_COMPLETE_<date>.json (no auto-increment). "
                         "The orchestrator skips a cell whose COMPLETE file already exists.")
    ap.add_argument("--trajectory", action="store_true",
                    help="S9 EXPLORATORY (prereg sec.3): capture norm-robust hidden-state "
                         "trajectory summaries per generation from the SAME generate call "
                         "(no extra GPU passes; TRAJECTORY_CAPTURE_SPEC.md). Default OFF; "
                         "must be UNIFORM across the matrix, so decide before the run.")
    ap.add_argument("--save-raw-trajectory", action="store_true",
                    help="Deep-anchor cells ONLY (sec.9): also dump raw per-layer states + "
                         "logits as compressed .npz per generation (float16). Implies "
                         "--trajectory. Large; leave OFF except the two anchor cells.")
    ap.add_argument("--resample-run", type=int, default=None,
                    help="S10 (prereg sec.4): data-level resample index r>=1. Resamples "
                         "contrast pairs (=> steering-vector variability) + eval prompts + "
                         "MMLU items WITH REPLACEMENT, deterministically seeded from "
                         "(SEED, r). Greedy decoding is unchanged. Omit for the canonical "
                         "un-resampled run (output byte-identical to legacy).")
    ap.add_argument("--resample-selftest", action="store_true",
                    help="S10: offline determinism/shape check of resample_plan (no model "
                         "load, no GPU). Prints PASS/FAIL and exits.")
    ap.add_argument("--capability", choices=["mmlu", "gsm8k"], default="gsm8k",
                    help="capability probe. 'gsm8k' (DEFAULT, 2026-07-11) = the prereg-deviation "
                         "2026-07-10 generation-consistent CoT probe (dose-matched to efficacy "
                         "under site=last); the confirmatory PRIMARY. 'mmlu' = legacy single-pass "
                         "log-prob MC, DOSE-BLIND under site=last (07-07 adjudicator) -- pass it "
                         "EXPLICITLY for legacy reproduction/diagnostics only, never confirmatory.")
    ap.add_argument("--secondary-mmlu", action="store_true",
                    help="with --capability gsm8k, ALSO run the single-pass MMLU probe and "
                         "store it separately (capability_secondary_mmlu*). Keeps single-pass "
                         "MMLU as the SECONDARY probe (Saurav 2026-07-10; the "
                         "generative-vs-single-pass dissociation figure).")
    ap.add_argument("--capability-max-new-tokens", type=int, default=None,
                    help="token cap for the GSM8K capability generation (default 512). "
                         "MMLU is single-pass and ignores this.")
    ap.add_argument("--gsm8k-selftest", action="store_true",
                    help="offline parser/scoring check for the GSM8K probe (no model, no "
                         "GPU). Prints PASS/FAIL and exits.")
    ap.add_argument("--capability-alpha-mode", choices=["full", "adaptive"], default="full",
                    help="which alphas get the (expensive) capability probe. 'full' "
                         "(default) = every grid alpha, output-equivalent to the legacy "
                         "single-loop harness. 'adaptive' (2026-07-11 design): efficacy "
                         "is still collected at EVERY alpha, then the prereg sec.7 E* "
                         "rule (steerquant_estar.py) picks baseline + a window around "
                         "each ladder fraction's efficacy crossing, and ONLY those "
                         "alphas get the capability probe. v1 scope: inline-efficacy "
                         "targets (sentiment, length) only.")
    ap.add_argument("--e-star-from", default=None,
                    help="sibling fp16 COMPLETE json (same model/target/layer). REQUIRED "
                         "with --capability-alpha-mode adaptive when scheme != fp16: E* "
                         "levels + arm are derived from the fp16 curve (S11) while the "
                         "crossing is found on THIS scheme's own efficacy curve.")
    ap.add_argument("--capability-alpha-neighbors", type=int, default=2,
                    help="adaptive mode: grid neighbors kept on EACH side of the "
                         "nearest-to-crossing alpha (default 2; interpolation margin).")
    ap.add_argument("--capability-batch-size", type=int, default=1,
                    help="GSM8K probe generation batch size (2026-07-11 economy lever). "
                         "1 (default) = original per-item path, byte-identical. >1 uses "
                         "LEFT-padded batched generation (Steerer site=last compatible); "
                         "run the GPU equivalence smoke before confirmatory use.")
    ap.add_argument("--adaptive-selftest", action="store_true",
                    help="offline check of the adaptive capability-alpha plumbing (no "
                         "model, no GPU): self path, sibling path, fail-closed guards. "
                         "Prints PASS/FAIL and exits.")
    args = ap.parse_args()
    if args.resample_selftest:
        _resample_selftest()
        return
    if args.gsm8k_selftest:
        _gsm8k_selftest()
        return
    if args.adaptive_selftest:
        _adaptive_selftest()
        return
    cap_max_new = args.capability_max_new_tokens or MAX_NEW_TOKENS_GSM8K
    n_q = args.subset if args.subset else 20

    # CLI overrides of the module defaults. Kept as globals so the rest of main() and
    # the results metadata continue to read them unchanged.
    global MODEL_NAME, STEER_LAYER, TARGET, ALPHAS
    if args.model is not None:
        MODEL_NAME = args.model
    if args.layer is not None:
        STEER_LAYER = args.layer
    if args.target is not None:
        TARGET = args.target
    if args.alphas is not None:
        ALPHAS = sorted(args.alphas)

    # Resolve the behavioral target: contrast pairs, eval prompts, judged-vs-length,
    # and the generated-token cap (length needs a generous cap so legitimate long
    # traces terminate; anything still going at the cap is a failure, prereg sec.8).
    tcfg = get_target_config(TARGET)
    contrast_pairs, eval_prompts = tcfg["pairs"], tcfg["evals"]
    target_kind = tcfg["kind"]
    max_new = args.max_new_tokens if args.max_new_tokens is not None else tcfg["max_new"]
    merge_cfg = tcfg.get("merge")            # S6 sycophancy: STRONG/MILD merge rule (None otherwise)
    neutral_evals = tcfg.get("neutral_evals")  # S6: pre-sweep alpha=0 saturation diagnostic (None otherwise)
    defer_scoring = bool(tcfg.get("defer_scoring"))  # S6: score in the separate LLM-judge pass, not inline

    # Adaptive capability-alpha guards (2026-07-11): FAIL BEFORE loading the
    # model. v1 scope = inline-efficacy targets (sentiment, length); deferred-
    # scoring targets don't know their own efficacy curve at selection time
    # (SteerQuant_capability_alphas_adaptive_design_2026-07-11.md, sec.3).
    estar_ref = None
    if args.capability_alpha_mode == "adaptive":
        if defer_scoring:
            raise SystemExit(
                "--capability-alpha-mode adaptive requires INLINE efficacy; "
                f"target {TARGET!r} defers scoring to a separate judge pass. "
                "Use --capability-alpha-mode full (or a static --alphas band).")
        if args.scheme != "fp16" and not args.e_star_from:
            raise SystemExit(
                "--capability-alpha-mode adaptive with scheme != fp16 requires "
                "--e-star-from <sibling fp16 COMPLETE json> (E* level + arm are "
                "fp16-derived, prereg sec.7 / S11).")
        if args.e_star_from:
            estar_ref = load_estar_reference(args.e_star_from, MODEL_NAME, TARGET,
                                             STEER_LAYER)
            print(f"  [adaptive] E* reference {estar_ref['source']}: "
                  f"arm={estar_ref['arm_sign']:+d} levels="
                  f"{[round(x, 3) for x in estar_ref['levels']]}")

    # S10: data-level resampling (prereg sec.4). resample is None => canonical run,
    # output byte-identical to legacy. When --resample-run r is set we resample the
    # DATA (pairs -> vector variability; eval prompts; MMLU items) with replacement,
    # deterministically from (SEED, r). Pair/MMLU resamples are applied at their use
    # sites; the eval-prompt resample is applied here.
    resample = None
    _resid_probe_prompt = eval_prompts[0]  # stable S13 residual-norm probe (pre-resample)
    pairs_used = contrast_pairs
    merge_pairs_strong = merge_cfg["pairs_strong"] if merge_cfg else None
    merge_pairs_mild = merge_cfg["pairs_mild"] if merge_cfg else None
    if args.resample_run is not None:
        if args.resample_run < 1:
            raise SystemExit("--resample-run must be >= 1 (or omit for the canonical run)")
        n_mild = len(merge_cfg["pairs_mild"]) if merge_cfg else 0
        n_main = len(merge_cfg["pairs_strong"]) if merge_cfg else len(contrast_pairs)
        resample = resample_plan(SEED, args.resample_run, n_main,
                                 len(eval_prompts), n_q, n_pairs_mild=n_mild)
        eval_prompts = [eval_prompts[int(i)] for i in resample["prompt_indices"]]
        if merge_cfg:
            merge_pairs_strong = [merge_cfg["pairs_strong"][int(i)] for i in resample["pair_indices"]]
            merge_pairs_mild = [merge_cfg["pairs_mild"][int(i)] for i in resample["pair_mild_indices"]]
        else:
            pairs_used = [contrast_pairs[int(i)] for i in resample["pair_indices"]]
        _npairs = len(merge_pairs_strong) if merge_cfg else len(pairs_used)
        print(f"  [resample] run r={args.resample_run}: {_npairs} pairs, "
              f"{len(eval_prompts)} prompts, {n_q} mmlu items (with replacement, "
              f"seed_key=[{SEED},{args.resample_run}])")

    set_seeds(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d")
    if args.run_tag:
        # Deterministic name for matrix resumability (orchestrator globs SteerQuant_<tag>_COMPLETE_*.json).
        base = f"SteerQuant_{args.run_tag}"; n = ""
    else:
        base = f"SteerQuant_phase0_{args.scheme}"
        # Guard against overwrite: auto-increment suffix.
        n = ""; k = 2
        while (OUTPUT_DIR / f"{base}{n}_COMPLETE_{stamp}.json").exists():
            n = f"_{k}"; k += 1
    partial = OUTPUT_DIR / f"{base}{n}_PARTIAL_{stamp}.json"

    print("=" * 64)
    print(f"  SteerQuant Phase 0  |  {args.scheme}  |  {MODEL_NAME}")
    print(f"  Device: {DEVICE}  Layer: {STEER_LAYER}  Site: {args.site}  Alphas: {ALPHAS}")
    print("=" * 64)

    t0 = time.time()
    model, tok = load_model(args.scheme, MODEL_NAME)
    print(f"  Model loaded ({time.time()-t0:.1f}s). Building steering vector...")
    sycophancy_merge = None
    if merge_cfg:
        # S6 merge rule (pre-registered, Saurav): build STRONG and MILD vectors;
        # if they point the same way (cosine > threshold) pool the pairs into ONE
        # vector, else keep STRONG as primary and record MILD as robustness-only.
        # (S10: merge_pairs_* are the per-run resampled pair sets, or the full sets
        # for the canonical run.)
        v_strong = build_steering_vector(model, tok, merge_pairs_strong, STEER_LAYER)
        v_mild = build_steering_vector(model, tok, merge_pairs_mild, STEER_LAYER)
        # both are unit vectors, so dot == cosine.
        cos = float(torch.dot(v_strong, v_mild).item())
        thr = float(merge_cfg["threshold"])
        if cos > thr:
            vector = build_steering_vector(
                model, tok, list(merge_pairs_strong) + list(merge_pairs_mild),
                STEER_LAYER)
            decision = "merged_pooled"
        else:
            vector = v_strong
            decision = "strong_primary_mild_robustness"
        sycophancy_merge = {"cosine": round(cos, 4), "threshold": thr, "decision": decision,
                            "n_pairs_strong": len(merge_pairs_strong),
                            "n_pairs_mild": len(merge_pairs_mild)}
        print(f"  [merge rule] cosine(strong,mild)={cos:.4f}  thr={thr}  -> {decision}")
    else:
        vector = build_steering_vector(model, tok, pairs_used, STEER_LAYER)
    # S4: persist the unit steering vector for offline geometric-fidelity work
    # (cosine fp16-vs-quant vector, H5 regime contrast, vector transfer) -- no GPU reruns.
    vector_file = f"{base}{n}_vector_{stamp}.npy"
    np.save(OUTPUT_DIR / vector_file, vector.numpy())
    # Diagnostic: residual-stream scale at the steer layer. alpha must be
    # comparable to this norm for a unit steering vector to have any effect.
    _ref = residual_at_layer(model, tok, _resid_probe_prompt, STEER_LAYER)
    print(f"  Residual norm @ layer {STEER_LAYER}: {_ref.norm().item():.1f} "
          f"(alpha should be on this order to bite)")
    steerer = Steerer(model, STEER_LAYER, vector, site=args.site)
    # Length is judge-free (token count); sycophancy defers scoring to the separate
    # LLM-judge pass (no inline judge loaded); other judged targets load one now.
    judge = None if (target_kind == "length" or defer_scoring) else get_judge(TARGET)
    if target_kind == "length":
        judge_manifest = "judge-free:generated-token-count"
    elif defer_scoring:
        judge_manifest = "deferred:LLM-judge separate pass (steerquant_score_llm.py)"
    else:
        judge_manifest = judge.manifest()
    print(f"  Target: {TARGET} ({target_kind}, max_new={max_new})  Judge: {judge_manifest}")

    results = {"meta": {"model": MODEL_NAME, "scheme": args.scheme,
                        "layer": STEER_LAYER, "seed": SEED, "date": stamp,
                        "gsm8k_parser": "v2.3.1",  # method note 2026-07-13 (+v2.1/v2.2); v2.3 2026-07-15 terminal-line; v2.3.1 2026-07-17 non-finite guard (crash fix, no scoring change)

                        "device": str(DEVICE), "target": TARGET,
                        "target_kind": target_kind, "max_new": max_new,
                        "site": args.site, "run_tag": args.run_tag, "judge": judge_manifest,
                        "residual_norm_at_layer": round(float(_ref.norm()), 2),  # S4 (S13 uses)
                        "vector_file": vector_file},
               "by_alpha": []}
    if sycophancy_merge is not None:
        results["meta"]["sycophancy_merge"] = sycophancy_merge  # S6: cosine + merge decision
    # Capability-probe provenance -- ALWAYS stamped (2026-07-11; supersedes the
    # only-when-non-default rule). Every output file must be self-identifying so
    # analysis/pooling can hard-filter on probe identity: a wrong-probe run must
    # never look like a valid confirmatory file (the 07-07 lesson). NOTE: the
    # primary capability values live in capability_mmlu* for analysis compat;
    # this key records what probe produced them.
    results["meta"]["capability_probe"] = (
        "single_pass_mmlu" if args.capability == "mmlu" else args.capability)
    results["meta"]["capability_max_new"] = (
        cap_max_new if args.capability == "gsm8k" else None)
    results["meta"]["capability_secondary"] = (
        "single_pass_mmlu" if (args.capability == "gsm8k" and args.secondary_mmlu)
        else None)
    # 2026-07-11 economy levers -- ALWAYS stamped (same self-identification
    # discipline as capability_probe): how the capability alphas were chosen
    # and how the probe generations were batched. The adaptive SELECTION
    # detail (levels/crossings/chosen alphas) is added after the efficacy pass.
    results["meta"]["capability_alpha_mode"] = args.capability_alpha_mode
    results["meta"]["capability_batch_size"] = int(args.capability_batch_size)
    # Environment provenance (2026-07-11 flag): GPU + library versions in meta.
    try:
        import transformers as _transformers
        _tf_ver = getattr(_transformers, "__version__", None)
    except Exception:
        _tf_ver = None
    results["meta"]["env"] = {
        "torch": getattr(torch, "__version__", None),
        "transformers": _tf_ver,
        "gpu": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)}
    if resample is not None:
        # S10: record the data-level resample so the run is fully reproducible.
        rmeta = {"run": args.resample_run, "seed_key": [SEED, args.resample_run],
                 "draw_order": RESAMPLE_DRAW_ORDER,
                 "policy": "all_runs_resampled; r=1 not reserved as identity",
                 "with_replacement": True,
                 "pair_indices": [int(i) for i in resample["pair_indices"]],
                 "prompt_indices": [int(i) for i in resample["prompt_indices"]],
                 "mmlu_item_indices": [int(i) for i in resample["mmlu_item_indices"]]}
        if resample["pair_mild_indices"] is not None:
            rmeta["pair_mild_indices"] = [int(i) for i in resample["pair_mild_indices"]]
        results["meta"]["resample"] = rmeta
    if args.trajectory or args.save_raw_trajectory:
        results["meta"]["trajectory"] = True  # S9: exploratory trajectory capture on
    try:
        with steerer:
            if neutral_evals is not None:
                # S6 neutral diagnostic (pre-sweep, alpha=0 ONLY): baseline
                # saturation check. Texts only -- scoring happens in the separate
                # LLM-judge pass. The abort criterion is a RECORDED FLAG, not an
                # automatic abort (needs Ben+Saurav ratification before enforcement).
                n_nd = len(neutral_evals)
                print(f"\r  [neutral diagnostic] gen 0/{n_nd} ...", end="", flush=True)
                ndiag = []
                for gi, p in enumerate(neutral_evals, 1):
                    g = generate(model, tok, p, steerer, 0.0, max_new=max_new)
                    g.pop("token_ids", None)  # keep files lean (as in the alpha loop)
                    ndiag.append({"prompt": p, **g})
                    print(f"\r  [neutral diagnostic] gen {gi}/{n_nd}", end="", flush=True)
                results["neutral_diagnostic"] = {
                    "alpha": 0.0,
                    "generations": ndiag,
                    "proposed_abort_threshold": tcfg.get("abort_threshold"),
                    "note": ("Texts only; scored in the separate LLM-judge pass. "
                             "PROPOSED abort criterion (NEEDS Ben+Saurav ratification, "
                             "NOT auto-enforced): if mean LEADING-set judge score at "
                             "alpha=0 exceeds the proposed threshold, the positive sweep "
                             "is saturated/compressed and the design should be revisited "
                             "BEFORE burning compute.")}
                print(f"\r  [neutral diagnostic] {n_nd} prompts saved (alpha=0)      ")
            # ── PASS 1: EFFICACY at every grid alpha (adaptive design
            # 2026-07-11: the efficacy curve keeps FULL grid resolution for
            # every scheme -- only the capability probe is alpha-reduced). ──
            for a in ALPHAS:
                tag = f"  alpha={a:5.1f}"
                n_p = len(eval_prompts)
                print(f"\r{tag}  gen 0/{n_p} ...", end="", flush=True)
                gens = []
                token_lists = []  # S3: model tokens per gen; used for rep-4, NOT persisted
                for gi, p in enumerate(eval_prompts, 1):
                    g = generate(model, tok, p, steerer, a, max_new=max_new,
                                 trajectory=args.trajectory,
                                 return_raw=args.save_raw_trajectory)
                    token_lists.append(g.pop("token_ids", None))  # strip before persisting -> lean files
                    if args.save_raw_trajectory:
                        rp, rl = g.pop("raw_paths", None), g.pop("raw_logits", None)
                        if rp is not None:
                            np.savez_compressed(
                                OUTPUT_DIR / f"{base}{n}_rawtraj_{stamp}_a{a:+.0f}_p{gi}.npz",
                                paths=rp, logits=rl, prompt=p, alpha=a)
                    gens.append({"prompt": p, **g})
                    print(f"\r{tag}  gen {gi}/{n_p}", end="", flush=True)
                record = {"alpha": a, "generations": gens}
                # S9: per-alpha aggregate of the per-generation trajectory summaries
                # (exploratory). Absent entirely when --trajectory/--save-raw are off,
                # so the default JSON structure is byte-identical.
                _tr = [gg["trajectory"] for gg in gens if gg.get("trajectory")]
                if _tr:
                    record["trajectory_mean"] = {k: float(np.mean([t[k] for t in _tr]))
                                                 for k in _tr[0]}
                if target_kind == "length":
                    # Judge-free efficacy: median generated-token count over
                    # NON-FAILURE traces; failures (no EOS / structural loop /
                    # rep-4>0.5) are excluded from length and counted in the
                    # co-primary failure rate (prereg sec.8).
                    flags, lengths = [], []
                    for g, toks in zip(gens, token_lists):
                        res = detect_termination_failure(
                            g["text"], eos_emitted=g["eos_emitted"],
                            tokens=toks,  # S3: rep-4 on model tokens (prereg sec.8); None -> word fallback
                            num_generated_tokens=g["n_tokens"], max_new_tokens=max_new)
                        flags.append(1 if res.failed else 0)
                        lengths.append(g["n_tokens"])
                    survivors = [L for L, f in zip(lengths, flags) if not f]
                    eff = float(np.median(survivors)) if survivors else 0.0
                    record["efficacy"] = round(eff, 4)                 # median survivor length
                    record["efficacy_per_prompt"] = [int(L) for L in survivors]
                    record["length_tokens_per_prompt"] = [int(L) for L in lengths]
                    record["failure_flags"] = flags
                    record["failure_rate"] = round(float(np.mean(flags)), 4)
                    eff_disp = f"len(med)={eff:.0f}  fail={record['failure_rate']:.2f}"
                elif defer_scoring:
                    # S6: sycophancy scoring runs in the separate LLM-judge pass
                    # (steerquant_score_llm.py) to avoid VRAM contention with the
                    # on-GPU model; the harness only persists the generations here.
                    record["efficacy"] = None
                    record["scoring_deferred"] = "steerquant_score_llm.py"
                    eff_disp = "efficacy=deferred(LLM-judge pass)"
                else:
                    eff_scores = score_behavior([g["text"] for g in gens], TARGET, judge=judge)
                    eff = float(np.mean(eff_scores))
                    record["efficacy"] = round(eff, 4)
                    record["efficacy_per_prompt"] = [round(float(s), 4) for s in eff_scores]
                    eff_disp = f"efficacy={eff:+.3f}"
                results["by_alpha"].append(record)
                print(f"\r{tag}  {eff_disp}      ")
                partial.write_text(json.dumps(results, indent=2))  # checkpoint (efficacy pass)

            # ── Capability-alpha selection (adaptive design 2026-07-11) ──
            # mode=full probes every alpha (output-equivalent to the legacy
            # single-loop harness). mode=adaptive runs the prereg sec.7 rule
            # ONLINE: baseline + a window around each ladder fraction's
            # efficacy crossing -- fp16 self-referential, or fp16-derived
            # levels/arm from --e-star-from with the crossing found on THIS
            # scheme's own just-collected curve (S11 discipline).
            if args.capability_alpha_mode == "adaptive":
                eff_curve = [r["efficacy"] for r in results["by_alpha"]]
                if estar_ref is not None:
                    sel = select_capability_alphas(
                        ALPHAS, eff_curve,
                        neighbors=args.capability_alpha_neighbors,
                        arm_sign=estar_ref["arm_sign"],
                        e_star_levels=estar_ref["levels"])
                    sel["source"] = estar_ref["source"]
                else:
                    sel = select_capability_alphas(
                        ALPHAS, eff_curve,
                        neighbors=args.capability_alpha_neighbors)
                    sel["source"] = "self(fp16 efficacy curve; ladder union)"
                cap_alphas = set(float(x) for x in sel["alphas"])
                results["meta"]["capability_alpha_selection"] = sel
                print(f"  [adaptive] capability alphas: {sorted(cap_alphas)} "
                      f"({len(cap_alphas)}/{len(ALPHAS)}; source={sel['source']})"
                      + (f"  FALLBACK: {sel['fallback']}" if sel["fallback"] else ""))
            else:
                cap_alphas = set(float(a) for a in ALPHAS)

            # ── PASS 2: CAPABILITY at the selected alphas only ──
            for record in results["by_alpha"]:
                a = record["alpha"]
                if float(a) not in cap_alphas:
                    continue   # adaptive: no capability read at this alpha
                tag = f"  alpha={a:5.1f}"
                cap_idx = resample["mmlu_item_indices"] if resample else None
                if args.capability == "gsm8k":
                    # Generation-consistent PRIMARY probe (prereg deviation 2026-07-10).
                    # The primary capability goes into the existing capability_mmlu* keys
                    # so steerquant_analysis.py consumes it UNCHANGED; meta.capability_probe
                    # records that these values are GSM8K (not MMLU). Failure rate is stored
                    # separately (Saurav 2026-07-10). A later cosmetic analysis pass can
                    # rename to a probe-agnostic key.
                    gsm_correct, gsm_failed, gsm_texts, gsm_eos = gsm8k_probe(
                        model, tok, steerer, a, n_q, cap_max_new,
                        progress_prefix=tag, item_indices=cap_idx,
                        batch_size=args.capability_batch_size)
                    cap = float(np.mean(gsm_correct)) if gsm_correct else 0.0
                    fail_rate = float(np.mean(gsm_failed)) if gsm_failed else 0.0
                    record["capability_mmlu"] = round(cap, 4)
                    record["capability_mmlu_items"] = [int(x) for x in gsm_correct]
                    record["capability_failure_flags"] = [int(x) for x in gsm_failed]
                    record["capability_failure_rate"] = round(fail_rate, 4)
                    # Audit trail (method note 2026-07-13, follow-up 1): persist the RAW
                    # capability generations + per-item EOS flags so any future parser/
                    # format issue is an OFFLINE re-score (correct = eos AND parse(text)
                    # == gold; failed = (not eos) OR unparseable) -- never a GPU re-run.
                    record["capability_texts"] = list(gsm_texts)
                    record["capability_eos_flags"] = [int(x) for x in gsm_eos]
                    if args.secondary_mmlu:
                        sec_items = mmlu_probe(model, tok, steerer, a, n_q,
                                               progress_prefix=tag, item_indices=cap_idx)
                        record["capability_secondary_mmlu"] = round(
                            float(np.mean(sec_items)) if sec_items else 0.0, 4)
                        record["capability_secondary_mmlu_items"] = [int(x) for x in sec_items]
                    cap_disp = f"gsm8k={cap:.3f} fail={fail_rate:.2f}"
                else:
                    cap_items = mmlu_probe(model, tok, steerer, a, n_q, progress_prefix=tag,
                                           item_indices=cap_idx)
                    cap = float(np.mean(cap_items)) if cap_items else 0.0
                    record["capability_mmlu"] = round(cap, 4)
                    record["capability_mmlu_items"] = [int(x) for x in cap_items]
                    cap_disp = f"mmlu={cap:.3f}"
                print(f"\r{tag}  {cap_disp}      ")
                partial.write_text(json.dumps(results, indent=2))  # checkpoint (capability pass)
    except KeyboardInterrupt:
        print(f"\n  Interrupted — PARTIAL kept at {partial.name}")
        sys.exit(1)

    complete = OUTPUT_DIR / f"{base}{n}_COMPLETE_{stamp}.json"
    partial.rename(complete)
    # Companion manifest (per conventions).
    (OUTPUT_DIR / f"{base}{n}_COMPLETE_{stamp}.txt").write_text(
        f"File:      {complete.name}\n"
        f"Created:   {datetime.now().isoformat()}\n"
        f"Script:    steerquant_phase0_harness.py\n"
        f"Source:    local inference ({MODEL_NAME}, {args.scheme})\n"
        f"Records:   {len(results['by_alpha'])} alpha conditions\n"
        f"Judge:     {judge_manifest}\n"
        f"Notes:     Efficacy from the behavioral judge above (NOT the old lexical scorer).\n"
        f"           Baseline = alpha 0.0. Iso-effect (IECC) analysis: steerquant_analysis.py.\n"
        f"Next step: sweep weight-only schemes (w*_bnb / _gptq / _awq) via --scheme; aggregate IECC.\n")
    print(f"\n  DONE in {time.time()-t0:.1f}s -> {complete.name}")
    print("  Sanity check: efficacy magnitude should grow with |alpha|; the capability probe (" + args.capability + ") should degrade only where the steering dose bites -- plateau near baseline, decline on the steered arm.")


if __name__ == "__main__":
    main()
