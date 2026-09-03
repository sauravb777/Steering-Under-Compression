#!/usr/bin/env python3
"""Per-scheme E* table at the ratified 10% censoring rule, with probe-gap
(distance from a* to nearest capability-probed alpha, in grid steps)."""
import csv, sys, json
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
THR = 0.10

def arrays(key):
    rs = cells[key]
    a = np.array([float(r["alpha"]) for r in rs])
    e = np.array([float(r["length_median"]) for r in rs])
    f = np.array([float(r["length_fail"]) for r in rs])
    c = np.array([float(r["gsm8k"]) if r["gsm8k"] else np.nan for r in rs])
    e = e.copy(); e[f > THR + 1e-9] = np.nan
    return a, e, f, c

out = {}
for model in MODELS:
    for scheme in ["fp16", "int8", "nf4"]:
        recs = []
        for r in range(1, 6):
            a0, e0, f0, c0 = arrays((model, "fp16", r))
            m0 = np.isfinite(e0)
            absa, sa, e_, c_, base, sign = _arm(a0[m0], e0[m0], c0[m0])
            extreme = float(e_[int(np.argmax(np.abs(e_ - base)))])
            levels = [base + fr * (extreme - base) for fr in E_STAR_LADDER]
            a1, e1, f1, c1 = arrays((model, scheme, r))
            m1 = np.isfinite(e1)
            A, S, E, C, B, _ = _arm(a1[m1], e1[m1], c1[m1], fixed_sign=sign)
            # accepted frac = first ladder frac with fp16 iecc < DELTA_MIN (on fp16 cell)
            acc_frac = None
            Af, Sf, Ef, Cf, Bf, _ = _arm(a0[m0], e0[m0], c0[m0], fixed_sign=sign)
            for fr, lv in zip(E_STAR_LADDER, levels):
                ast = _first_crossing(Af, Ef, lv)
                if ast is None: continue
                mc = np.isfinite(Cf)
                if not mc.any(): continue
                ie = float(np.interp(0.0, Af[mc], Cf[mc])) - float(np.interp(ast, Af[mc], Cf[mc]))
                if ie < DELTA_MIN: acc_frac = fr; break
            if acc_frac is None: recs.append(None); continue
            lv = levels[E_STAR_LADDER.index(acc_frac)]
            ast = _first_crossing(A, E, lv)
            if ast is None: recs.append(None); continue
            mc = np.isfinite(C)
            cap0 = float(np.interp(0.0, A[mc], C[mc]))
            capat = float(np.interp(ast, A[mc], C[mc]))
            probed = A[mc]
            gap = float(np.min(np.abs(probed - ast)))
            # grid step near a*: spacing of full arm grid around ast
            allarm = np.array(sorted(abs(x) for x in a1 if x * sign >= 0))
            j = int(np.searchsorted(allarm, ast))
            step = float(allarm[min(j, len(allarm)-1)] - allarm[max(j-1, 0)]) or 1.0
            recs.append({"frac": acc_frac, "a_star": ast, "iecc": cap0 - capat,
                         "gap_alpha": gap, "gap_steps": gap / step})
        ok = [x for x in recs if x]
        if ok:
            out[(model, scheme)] = ok
            fr_used = sorted(set(x["frac"] for x in ok))
            print(f"{model[:26]:26s} {scheme:5s} n={len(ok)}/5 frac={fr_used} "
                  f"a*={np.mean([x['a_star'] for x in ok]):7.2f} (sd {np.std([x['a_star'] for x in ok]):5.2f})  "
                  f"IECC={np.mean([x['iecc'] for x in ok]):+.3f} (sd {np.std([x['iecc'] for x in ok]):.3f})  "
                  f"probe-gap={np.mean([x['gap_alpha'] for x in ok]):6.2f} alpha "
                  f"({np.mean([x['gap_steps'] for x in ok]):.1f} steps)")
        else:
            print(f"{model[:26]:26s} {scheme:5s} no accepted resamples")
json.dump({f"{k[0]}|{k[1]}": v for k, v in out.items()},
          open("final_table_thr10.json", "w"), indent=1)
