#!/usr/bin/env python3
"""
Logit-lens the LENGTH steering vector.
======================================
Question: when we add +vector (= mean(verbose) - mean(terse)), does it literally
push the next-token distribution toward EOS / answer-y / closing tokens? If so,
that mechanistically explains "+alpha shortens" -- the vector is (partly) a
"stop / wrap-up" direction, not a pure length-propensity direction.

Method (cheap: one model load, NO generation): rebuild the length vector with the
harness's corrected assistant-role extraction, then project it through the output
embedding (unembedding) and rank the tokens it most promotes / suppresses. Also
report cosine + logit-lens rank of each EOS id.

CAVEAT: this is approximate. The vector lives at the steer layer; logit-lens
skips the remaining layers and the final norm, so treat the token list as a
heuristic for the direction's "flavour," not an exact generation prediction.

Usage:
    python logit_lens_length_vector.py                 # layer from harness default
    python logit_lens_length_vector.py --layer 16      # try another layer
    python logit_lens_length_vector.py --target length # (length is the default)
"""
import argparse
import torch

from steerquant_phase0_harness import (
    load_model, build_steering_vector, get_target_config, _eos_ids,
    STEER_LAYER, MODEL_NAME)


def show(tok, idxs, vals, title):
    print(f"\n  {title}")
    for i, val in zip(idxs.tolist(), vals.tolist()):
        t = tok.decode([i]).replace("\n", "\\n")
        print(f"    {val:+8.3f}  id={i:>6}  {t!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=STEER_LAYER)
    ap.add_argument("--target", default="length")
    ap.add_argument("--model", default=MODEL_NAME)
    ap.add_argument("--topk", type=int, default=30)
    args = ap.parse_args()

    model, tok = load_model("fp16", args.model)
    tcfg = get_target_config(args.target)
    # +v = mean(pos) - mean(neg); for length, pos=verbose => +v is the "verbose"
    # direction. Unit-normalised by build_steering_vector.
    v = build_steering_vector(model, tok, tcfg["pairs"], args.layer).float().cpu()

    W = model.get_output_embeddings().weight.detach().float().cpu()  # [V, d]
    logits = W @ v                                                    # [V]
    k = args.topk
    top = torch.topk(logits, k)
    bot = torch.topk(-logits, k)

    print(f"  Logit-lens of the {args.target} vector @ layer {args.layer} "
          f"(+v = pos - neg; for length, verbose - terse). unit norm={v.norm():.3f}")
    print(f"  model={args.model}")
    show(tok, top.indices, top.values, f"TOP {k} PROMOTED by +v:")
    show(tok, bot.indices, bot.values, f"TOP {k} SUPPRESSED by +v:")

    eos = sorted(_eos_ids(model, tok))
    print(f"\n  EOS ids: {eos}")
    V = W.shape[0]
    for e in eos:
        row = W[e]
        cos = float(torch.dot(row, v) / (row.norm() * v.norm() + 1e-8))
        rank = int((logits > logits[e]).sum().item())  # 0 = most promoted
        ts = tok.decode([e]).replace("\n", "\\n")
        print(f"    eos id={e} {ts!r}: cos(+v, W[eos])={cos:+.4f}  "
              f"logit-lens rank={rank}/{V}  (low rank => +v favours stopping)")

    print("\n  Read: if +v PROMOTES EOS / closing-punctuation / 'answer'/'so' tokens")
    print("  (high cos, low rank), then '+alpha shortens' is the vector pointing at")
    print("  STOP/wrap-up register. If +v instead promotes connective reasoning words")
    print("  ('step', 'first', 'because', 'let'), the direction is more length-like and")
    print("  the inverted sign is just convention -> flip it.")


if __name__ == "__main__":
    main()
