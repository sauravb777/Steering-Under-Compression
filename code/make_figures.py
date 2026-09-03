#!/usr/bin/env python3
"""Figures F1-F4 for Steering Under Compression. Light-mode print palette
(validated): fp16 #2a78d6, int8 #eb6834, nf4 #1baf7a."""
import csv, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

plt.rcParams.update({
    "font.family": "serif", "font.size": 7.5, "axes.titlesize": 8,
    "axes.labelsize": 7.5, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5, "mathtext.fontset": "stix",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "axes.edgecolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#52514e",
    "axes.labelcolor": "#0b0b0b", "text.color": "#0b0b0b",
    "figure.dpi": 200, "savefig.dpi": 300})
C = {"fp16": "#2a78d6", "int8": "#eb6834", "nf4": "#1baf7a"}
GRID = dict(color="#e4e3df", lw=0.5)

rows = list(csv.DictReader(open(
    "/mnt/user-data/uploads/Claude/projects/steering-quantization/length_matrix_long.csv")))
MODELS = [("Qwen2.5-7B-Instruct", "Qwen2.5-7B"),
          ("Meta-Llama-3.1-8B-Instruct", "Llama-3.1-8B"),
          ("Mistral-7B-Instruct-v0.3", "Mistral-7B-v0.3"),
          ("gemma-2-9b-it", "Gemma-2-9B")]
SCHEMES = ["fp16", "int8", "nf4"]

def curves(model, scheme):
    """Mean over resamples of survivor-median length (NaN where all resamples
    all-fail) and mean failure rate, per alpha."""
    d = {}
    for r in rows:
        if r["model"] != model or r["scheme"] != scheme: continue
        a = float(r["alpha"]); e = float(r["length_median"]); f = float(r["length_fail"])
        d.setdefault(a, []).append((e if f < 1.0 else np.nan, f))
    alphas = np.array(sorted(d))
    eff = np.array([np.nanmean([x[0] for x in d[a]]) if any(np.isfinite(x[0]) for x in d[a])
                    else np.nan for a in alphas])
    fail = np.array([np.mean([x[1] for x in d[a]]) for a in alphas])
    return alphas, eff, fail

# ---------- F1: dose-response + failure panel ----------
fig, axes = plt.subplots(2, 4, figsize=(5.5, 3.1),
                         gridspec_kw=dict(height_ratios=[2.1, 1], hspace=0.14, wspace=0.34))
for j, (model, label) in enumerate(MODELS):
    top, bot = axes[0, j], axes[1, j]
    for scheme in SCHEMES:
        a, e, f = curves(model, scheme)
        top.plot(a, e, color=C[scheme], lw=1.1, marker="o", ms=1.6,
                 mew=0, label=scheme, zorder=3)
        bot.plot(a, f, color=C[scheme], lw=0.9, marker="o", ms=1.4, mew=0, zorder=3)
    top.set_title(label, pad=3)
    top.grid(axis="y", **GRID); bot.grid(axis="y", **GRID)
    top.set_xticklabels([])
    bot.axhline(0.10, color="#52514e", lw=0.6, ls=(0, (3, 2)), zorder=2)
    bot.set_ylim(-0.05, 1.08); bot.set_yticks([0, 0.5, 1.0])
    bot.set_xlabel(r"steering coefficient $\alpha$", labelpad=1.5)
    for ax in (top, bot):
        ax.margins(x=0.03)
    if j == 0:
        top.set_ylabel("median tokens\n(survivors)", labelpad=2)
        bot.set_ylabel("failure\nrate", labelpad=2)
        bot.annotate("censor threshold (0.10)", xy=(0.03, 0.16), xycoords="axes fraction",
                     fontsize=5.6, color="#52514e")
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=3, frameon=False,
           bbox_to_anchor=(0.5, 1.035), handlelength=1.4, columnspacing=1.2)
fig.savefig("fig/F1_length_dose_response.pdf", bbox_inches="tight")
fig.savefig("fig/F1_length_dose_response.png", bbox_inches="tight")
plt.close(fig)

# ---------- F2: E* anatomy, Llama fp16 r1 positive arm ----------
r1 = {float(r["alpha"]): (float(r["length_median"]), float(r["length_fail"]))
      for r in rows if r["model"] == "Meta-Llama-3.1-8B-Instruct"
      and r["scheme"] == "fp16" and r["resample"] == "1" and float(r["alpha"]) >= 0}
al = np.array(sorted(r1)); ev = np.array([r1[a][0] for a in al]); fv = np.array([r1[a][1] for a in al])
base = ev[al == 0][0]
fig, ax = plt.subplots(figsize=(4.6, 2.5))
ax.grid(axis="y", **GRID)
ax.plot(al, ev, color="#b4b2a9", lw=1.0, ls="--", zorder=2)          # raw curve
keep = fv <= 0.10
ax.plot(al[keep], ev[keep], color=C["fp16"], lw=1.6, marker="o", ms=3.4, mew=0,
        zorder=4, label="censored curve (fail $\\leq$ 10%)")
bad = ~keep
ax.plot(al[bad], ev[bad], ls="none", marker="x", ms=4.2, mew=1.1,
        color="#eb6834", zorder=4, label="excluded (fail > 10%)")
ax.axhline(base, color="#52514e", lw=0.7, ls=(0, (4, 2)))
ax.annotate("baseline (180)", xy=(29.5, base + 6), ha="right", fontsize=6.2, color="#52514e")
# as-coded ladder level f=0.5 anchored on collapse floor: 90
ax.axhline(90, color="#e87ba4", lw=1.0)
x1 = 12 + 3 * (338 - 90) / 338.0
ax.plot([x1], [90], marker="v", ms=5, color="#e87ba4", zorder=5)
ax.annotate("as-coded $E^{*}(0.5)=90$:\nanchored on the collapse floor,\ncrossing lands on the cliff",
            xy=(x1, 90), xytext=(17.5, 150), fontsize=6.2, color="#993556",
            arrowprops=dict(arrowstyle="-", color="#e87ba4", lw=0.6))
# censored level: extreme = max surviving effect on fail<=10% subset
ext = np.nanmax(np.where(keep, ev, np.nan)); lvl = base + 0.5 * (ext - base)
ax.axhline(lvl, color="#4a3aa7", lw=1.0)
# crossing of censored curve at lvl
ka, ke = al[keep], ev[keep]
xc = None
for i in range(len(ka) - 1):
    d0, d1 = ke[i] - lvl, ke[i + 1] - lvl
    if d0 == 0 or d0 * d1 < 0:
        t = d0 / (d0 - d1); xc = ka[i] + t * (ka[i + 1] - ka[i]); break
if xc is not None:
    ax.plot([xc], [lvl], marker="^", ms=5, color="#4a3aa7", zorder=5)
    ax.annotate("censored $E^{*}(0.5)$:\nanchored on the surviving peak,\ncrossing in the clean region",
                xy=(xc, lvl), xytext=(9.5, 268), fontsize=6.2, color="#4a3aa7",
                arrowprops=dict(arrowstyle="-", color="#4a3aa7", lw=0.6))
ax.annotate("collapse\nfloor", xy=(22, 8), fontsize=6.2, color="#52514e", ha="center")
ax.set_xlabel(r"steering coefficient $\alpha$ (positive arm)")
ax.set_ylabel("median generated tokens")
ax.set_xlim(-0.8, 31); ax.set_ylim(-14, 360)
ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.0, 1.02), handlelength=1.6)
fig.savefig("fig/F2_estar_anatomy.pdf", bbox_inches="tight")
fig.savefig("fig/F2_estar_anatomy.png", bbox_inches="tight")
plt.close(fig)

# ---------- F3: sentiment forest plot (Option C, 90% CI) ----------
F3 = [  # (label, scheme, mean, lo90, hi90, pooled?)
    ("Qwen2.5-7B",      "int8", -0.0113, -0.0387, +0.0162, False),
    ("Llama-3.1-8B",    "int8", +0.0164, -0.0840, +0.1167, False),
    ("Mistral-7B-v0.3", "int8", -0.0237, -0.0696, +0.0222, False),
    ("Pooled (3 models)","int8", -0.0126, -0.0302, +0.0050, True),
    ("Qwen2.5-7B",      "nf4",  +0.0137, -0.0159, +0.0433, False),
    ("Llama-3.1-8B",    "nf4",  -0.0402, -0.1231, +0.0426, False),
    ("Mistral-7B-v0.3", "nf4",  -0.0682, -0.1228, -0.0136, False),
    ("Pooled (3 models)","nf4",  -0.0278, -0.0757, +0.0202, True),
]
fig, ax = plt.subplots(figsize=(4.6, 2.3))
ys = [7.6, 6.6, 5.6, 4.6, 3.0, 2.0, 1.0, 0.0]
ax.add_patch(Rectangle((-0.03, -0.7), 0.06, 9.1, facecolor="#f1efe8",
                       edgecolor="none", zorder=1))
ax.axvline(0, color="#52514e", lw=0.7, zorder=2)
for (lab, sch, m, lo, hi, pooled), y in zip(F3, ys):
    col = C[sch]
    ax.plot([lo, hi], [y, y], color=col, lw=1.3, zorder=3,
            solid_capstyle="round")
    ax.plot([m], [y], marker="D" if pooled else "o",
            ms=4.6 if pooled else 3.6, color=col, mew=0, zorder=4)
    ax.annotate(lab, xy=(-0.165, y), va="center", ha="left", fontsize=6.6,
                fontweight="bold" if pooled else "normal", annotation_clip=False)
ax.annotate("int8", xy=(-0.165, 8.45), fontsize=7, fontweight="bold",
            color=C["int8"], annotation_clip=False)
ax.annotate("nf4", xy=(-0.165, 3.85), fontsize=7, fontweight="bold",
            color=C["nf4"], annotation_clip=False)
ax.annotate(r"equivalence margin $\pm\delta=0.03$", xy=(0.0, 8.7), ha="center",
            fontsize=6.0, color="#52514e")
ax.set_xlim(-0.17, 0.17); ax.set_ylim(-0.7, 8.6)
ax.set_yticks([])
ax.spines["left"].set_visible(False)
ax.set_xlabel(r"IECC contrast: scheme $-$ fp16  (90% CI)   $\leftarrow$ cheaper | costlier $\rightarrow$")
fig.savefig("fig/F3_sentiment_forest.pdf", bbox_inches="tight")
fig.savefig("fig/F3_sentiment_forest.png", bbox_inches="tight")
plt.close(fig)

# ---------- F4: baseline shift ----------
base_v = {}
for r in rows:
    if float(r["alpha"]) == 0.0 and r["gsm8k"]:
        base_v.setdefault((r["model"], r["scheme"]), []).append(float(r["gsm8k"]))
fig, ax = plt.subplots(figsize=(4.3, 1.9))
ax.grid(axis="y", **GRID)
w = 0.26
for k, scheme in enumerate(SCHEMES):
    xs = np.arange(4) + (k - 1) * (w + 0.015)
    vs = [np.mean(base_v[(m, scheme)]) for m, _ in MODELS]
    b = ax.bar(xs, vs, width=w, color=C[scheme], label=scheme, zorder=3)
    for x, v in zip(xs, vs):
        ax.annotate(f"{v:.2f}".lstrip("0"), xy=(x, v + 0.015), ha="center",
                    fontsize=5.6, color="#52514e")
ax.set_xticks(np.arange(4)); ax.set_xticklabels([l for _, l in MODELS], fontsize=6.6)
ax.set_ylim(0, 1.02); ax.set_ylabel("GSM8K accuracy\nat $\\alpha=0$")
ax.legend(frameon=False, ncol=3, loc="upper right", bbox_to_anchor=(1.0, 1.12),
          handlelength=1.0, columnspacing=1.0)
fig.savefig("fig/F4_baseline_shift.pdf", bbox_inches="tight")
fig.savefig("fig/F4_baseline_shift.png", bbox_inches="tight")
plt.close(fig)
print("done:", *sorted(__import__("os").listdir("fig")), sep="\n  ")
