#!/usr/bin/env python3
"""E* crossing + contamination analysis over the 60-cell length matrix.

Two conventions for the survivor-median efficacy at ALL-FAIL alphas:
  A (as-coded)  : efficacy = 0.0  (harness JSON convention; extreme may be the
                  collapse floor -> ladder levels can sit BELOW baseline)
  B (censored)  : efficacy = NaN (all-fail alphas dropped from the curve;
                  extreme = peak surviving effect)

S11 discipline: arm + E* levels fixed from the SAME-resample fp16 curve,
crossing found on each scheme's own curve. Ladder acceptance (prereg sec.7):
first frac in (0.5, 0.4, 0.3) whose fp16 IECC < 0.10.
"""
import csv, json, sys
import numpy as np
sys.path.insert(0, "/mnt/user-data/uploads/Claude/projects/steering-quantization")
from steerquant_estar import _arm, _first_crossing, E_STAR_LADDER, DELTA_MIN

rows = list(csv.DictReader(open(
    "/mnt/user-data/uploads/Claude/projects/steering-quantization/length_matrix_long.csv")))
cells = {}
for r in rows:
    key = (r["model"], r["scheme"], int(r["resample"]))
    cells.setdefault(key, []).append(r)
for k in cells:
    cells[k].sort(key=lambda r: float(r["alpha"]))

MODELS = ["Qwen2.5-7B-Instruct", "Meta-Llama-3.1-8B-Instruct",
          "Mistral-7B-Instruct-v0.3", "gemma-2-9b-it"]
SCHEMES = ["fp16", "int8", "nf4"]

def arrays(key, conv):
    rs = cells[key]
    alphas = np.array([float(r["alpha"]) for r in rs])
    eff = np.array([float(r["length_median"]) for r in rs])
    fail = np.array([float(r["length_fail"]) for r in rs])
    cap = np.array([float(r["gsm8k"]) if r["gsm8k"] else np.nan for r in rs])
    if conv == "B":
        eff = eff.copy(); eff[fail >= 1.0] = np.nan
    return alphas, eff, fail, cap

def masked_arm(alphas, eff, cap, fixed_sign=None):
    """_arm, but on the finite-eff subset (convention B drops all-fail points)."""
    m = np.isfinite(eff)
    return _arm(alphas[m], eff[m], cap[m], fixed_sign=fixed_sign), alphas[m], eff[m]

def levels_from_fp16(model, r, conv):
    alphas, eff, fail, cap = arrays((model, "fp16", r), conv)
    (absa, sa, e, c, base, sign), _, _ = masked_arm(alphas, eff, cap)
    extreme = float(e[int(np.argmax(np.abs(e - base)))])
    levels = [base + f * (extreme - base) for f in E_STAR_LADDER]
    return sign, levels, base, extreme

def cell_iecc(key, sign, levels, conv):
    alphas, eff, fail, cap = arrays(key, conv)
    (absa, sa, e, c, base, s2), _, _ = masked_arm(alphas, eff, cap, fixed_sign=sign)
    out = []
    # failure-rate lookup on the SAME signed arm (full grid incl. all-fail pts)
    arm_full = [(abs(a), f) for a, f in zip(alphas, fail) if a * sign >= 0]
    arm_full.sort()
    fa = np.array([x[0] for x in arm_full]); fr = np.array([x[1] for x in arm_full])
    for f_, level in zip(E_STAR_LADDER, levels):
        a_star = _first_crossing(absa, e, level)
        rec = {"frac": f_, "level": round(level, 1), "a_star": None}
        if a_star is not None:
            m = np.isfinite(c)
            cap0 = float(np.interp(0.0, absa[m], c[m])) if m.any() else np.nan
            cap_at = float(np.interp(a_star, absa[m], c[m])) if m.any() else np.nan
            # bracketing grid alphas on the full arm + their length-failure rates
            lo = fa[fa <= a_star + 1e-9].max() if (fa <= a_star + 1e-9).any() else None
            hi = fa[fa >= a_star - 1e-9].min() if (fa >= a_star - 1e-9).any() else None
            fl = float(fr[np.argmin(np.abs(fa - lo))]) if lo is not None else np.nan
            fh = float(fr[np.argmin(np.abs(fa - hi))]) if hi is not None else np.nan
            rec.update({"a_star": round(float(a_star), 2), "cap0": round(cap0, 3),
                        "cap_at": round(cap_at, 3), "iecc": round(cap0 - cap_at, 3),
                        "bracket": (lo, hi), "bracket_fail": (fl, fh),
                        "contaminated": bool(max(fl, fh) >= 0.30)})
        out.append(rec)
    return out

report = {}
for conv in ("A", "B"):
    print(f"\n{'='*76}\nCONVENTION {conv} "
          f"({'as-coded: all-fail eff=0' if conv=='A' else 'censored: all-fail eff=NaN'})\n{'='*76}")
    report[conv] = {}
    for model in MODELS:
        print(f"\n-- {model}")
        report[conv][model] = {}
        # acceptance per resample on fp16
        acc = []
        for r in range(1, 6):
            sign, levels, base, ext = levels_from_fp16(model, r, conv)
            recs = cell_iecc((model, "fp16", r), sign, levels, conv)
            a = next(({"frac": rec["frac"], "iecc": rec["iecc"]}
                      for rec in recs if rec["a_star"] is not None
                      and rec.get("iecc") is not None and rec["iecc"] < DELTA_MIN), None)
            acc.append(a)
        n_acc = sum(1 for a in acc if a)
        fr_acc = [a["frac"] for a in acc if a]
        print(f"   fp16 ladder acceptance: {n_acc}/5 resamples accept"
              + (f" (fracs {fr_acc})" if n_acc else "  -> NO confirmatory IECC"))
        report[conv][model]["acceptance"] = acc
        for scheme in SCHEMES:
            per_r = []
            for r in range(1, 6):
                sign, levels, base, ext = levels_from_fp16(model, r, conv)
                recs = cell_iecc((model, scheme, r), sign, levels, conv)
                per_r.append(recs)
            # summarize at frac=0.5 across resamples
            f05 = [rr[0] for rr in per_r]
            a_stars = [x["a_star"] for x in f05 if x["a_star"] is not None]
            ieccs = [x["iecc"] for x in f05 if x.get("iecc") is not None]
            cont = [x.get("contaminated") for x in f05 if x["a_star"] is not None]
            if a_stars:
                bf = [x["bracket_fail"] for x in f05 if x["a_star"] is not None]
                mbf = max(max(b) for b in bf)
                print(f"   {scheme:5s} f=0.5: a* {np.mean(a_stars):7.2f}"
                      f" (sd {np.std(a_stars):5.2f}, n={len(a_stars)}/5)"
                      f"  IECC {np.mean(ieccs):+.3f} (sd {np.std(ieccs):.3f})"
                      f"  contaminated {sum(bool(x) for x in cont)}/{len(cont)}"
                      f"  worst-bracket fail {mbf:.2f}")
            else:
                print(f"   {scheme:5s} f=0.5: no crossing on any resample")
            report[conv][model][scheme] = per_r

json.dump(report, open("estar_crossings_report.json", "w"), indent=1, default=str)
print("\nsaved estar_crossings_report.json")
