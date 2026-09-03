#!/usr/bin/env python3
"""Saurav's requested analyses 2+3: E* sensitivity to failure-censoring
threshold, and failure-rate vs IECC correlation. Extends estar_crossings.py."""
import csv, json, sys
import numpy as np
sys.path.insert(0, "/mnt/user-data/uploads/Claude/projects/steering-quantization")
from steerquant_estar import _arm, _first_crossing, E_STAR_LADDER, DELTA_MIN

rows = list(csv.DictReader(open(
    "/mnt/user-data/uploads/Claude/projects/steering-quantization/length_matrix_long.csv")))
cells = {}
for r in rows:
    cells.setdefault((r["model"], r["scheme"], int(r["resample"])), []).append(r)
for k in cells: cells[k].sort(key=lambda r: float(r["alpha"]))
MODELS = ["Qwen2.5-7B-Instruct", "Meta-Llama-3.1-8B-Instruct",
          "Mistral-7B-Instruct-v0.3", "gemma-2-9b-it"]
SCHEMES = ["fp16", "int8", "nf4"]

def arrays(key, thr):
    rs = cells[key]
    alphas = np.array([float(r["alpha"]) for r in rs])
    eff = np.array([float(r["length_median"]) for r in rs])
    fail = np.array([float(r["length_fail"]) for r in rs])
    cap = np.array([float(r["gsm8k"]) if r["gsm8k"] else np.nan for r in rs])
    if thr is not None:
        eff = eff.copy(); eff[fail > thr + 1e-9] = np.nan
    return alphas, eff, fail, cap

def analyze(key, sign, levels, thr):
    alphas, eff, fail, cap = arrays(key, thr)
    m = np.isfinite(eff)
    if m.sum() < 3: return []
    absa, sa, e, c, base, s2 = _arm(alphas[m], eff[m], cap[m], fixed_sign=sign)
    arm_full = sorted((abs(a), f) for a, f in zip(alphas, fail) if a * sign >= 0)
    fa = np.array([x[0] for x in arm_full]); fr = np.array([x[1] for x in arm_full])
    out = []
    for f_, level in zip(E_STAR_LADDER, levels):
        a_star = _first_crossing(absa, e, level)
        rec = {"frac": f_, "a_star": a_star}
        if a_star is not None:
            mc = np.isfinite(c)
            if mc.any():
                cap0 = float(np.interp(0.0, absa[mc], c[mc]))
                cap_at = float(np.interp(a_star, absa[mc], c[mc]))
                lo = fa[fa <= a_star + 1e-9].max(); hi = fa[fa >= a_star - 1e-9].min()
                rec.update({"iecc": cap0 - cap_at,
                            "maxbf": float(max(fr[np.argmin(np.abs(fa-lo))],
                                               fr[np.argmin(np.abs(fa-hi))]))})
        out.append(rec)
    return out

def fp16_ref(model, r, thr):
    alphas, eff, fail, cap = arrays((model, "fp16", r), thr)
    m = np.isfinite(eff)
    absa, sa, e, c, base, sign = _arm(alphas[m], eff[m], cap[m])
    extreme = float(e[int(np.argmax(np.abs(e - base)))])
    return sign, [base + f * (extreme - base) for f in E_STAR_LADDER]

print("CENSORING-THRESHOLD SWEEP (drop alphas with length_fail > thr)")
for thr, label in [(None, "as-coded (no censor)"), (0.999, "censor all-fail only"),
                   (0.25, "thr 25%"), (0.10, "thr 10%"), (0.05, "thr 5%")]:
    print(f"\n--- {label} ---")
    for model in MODELS:
        accs, astars, ieccs = 0, [], []
        for r in range(1, 6):
            try: sign, levels = fp16_ref(model, r, thr)
            except Exception: continue
            recs = analyze((model, "fp16", r), sign, levels, thr)
            if any(rec["a_star"] is not None and rec.get("iecc") is not None
                   and rec["iecc"] < DELTA_MIN for rec in recs): accs += 1
            f05 = recs[0] if recs else None
            if f05 and f05["a_star"] is not None:
                astars.append(f05["a_star"])
                if f05.get("iecc") is not None: ieccs.append(f05["iecc"])
        a_s = f"a*={np.mean(astars):7.2f} sd {np.std(astars):5.2f} n={len(astars)}" if astars else "no crossings"
        i_s = f"IECC={np.mean(ieccs):+.3f}" if ieccs else ""
        print(f"  {model[:26]:26s} accept {accs}/5   fp16 f=0.5 {a_s}  {i_s}")

print("\nFAILURE-vs-IECC CORRELATION (censor-all-fail convention, f=0.5, all 60 cells)")
xs, ys = [], []
for model in MODELS:
    for scheme in SCHEMES:
        for r in range(1, 6):
            sign, levels = fp16_ref(model, r, 0.999)
            recs = analyze((model, scheme, r), sign, levels, 0.999)
            f05 = recs[0] if recs else None
            if f05 and f05.get("iecc") is not None and f05.get("maxbf") is not None:
                xs.append(f05["maxbf"]); ys.append(f05["iecc"])
def spearman(x, y):
    rx = np.argsort(np.argsort(x)).astype(float); ry = np.argsort(np.argsort(y)).astype(float)
    return float(np.corrcoef(rx, ry)[0, 1])
xs, ys = np.array(xs), np.array(ys)
print(f"  n={len(xs)} cells with crossings; Spearman rho = {spearman(xs, ys):.3f}; "
      f"Pearson r = {float(np.corrcoef(xs, ys)[0,1]):.3f}")
clean = ys[xs < 0.05]; dirty = ys[xs >= 0.30]
print(f"  IECC when bracket fail <5%:  mean {np.mean(clean):+.3f} (n={len(clean)})")
print(f"  IECC when bracket fail >=30%: mean {np.mean(dirty):+.3f} (n={len(dirty)})")
