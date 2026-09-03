#!/usr/bin/env python3
"""SteerQuant -- IECC + bootstrap CIs + H1 power tools. Pure numpy, no GPU.
Self-test: --selftest. Power table: --power. IECC report: --results <glob>.
2026-07-07: length-target bootstrap fixed (survivor-median over resampled
prompts; censoring propagated); prereg sec.7 E* acceptance ladder auto-applied;
report() groups by (model, target); H1 three-label + H2 Rule A encoded.
2026-07-11: E*/arm/crossing rule EXTRACTED to steerquant_estar.py (shared with
the harness's online adaptive capability-alpha selection; no behavior change).
Adaptive-capability files (capability probed only at baseline + a window around
the E* crossings) are now supported: missing capability at an alpha enters as
NaN and IECC interpolates over the probed alphas only.
2026-07-12: POOLING GUARD -- report() refuses files whose meta.capability_probe
is missing or != the prereg primary (gsm8k); --expect-probe overrides for
labeled diagnostics only (selftest [17]).
2026-07-14: Option C run-level combiner (Saurav ruling 2026-07-12;
METHOD_NOTE_optionC-REML-HK_2026-07-14.md): the 5 resample runs per (model,
scheme) combine as a random-effects meta-analysis -- REML tau2 + MODIFIED
Knapp-Hartung CI (t, k-1 df); --combine-runs (selftest [18])."""
from __future__ import annotations

import argparse
import glob as _glob
import json
import re
from math import erf as _erf
from pathlib import Path

import numpy as np

from steerquant_estar import (E_STAR_LADDER, DELTA_MIN,  # noqa: F401 (re-export;
                              _arm, _first_crossing)      # single source of truth)


def efficacy_mean(cell) -> float:
    for k in ("efficacy", "efficacy_sentiment", "efficacy_llm"):
        if k in cell and cell[k] is not None:
            return float(cell[k])
    raise KeyError(f"no efficacy field in cell (alpha={cell.get('alpha')})")


def capability_mean(cell) -> float:
    if cell.get("capability_mmlu") is not None:
        return float(cell["capability_mmlu"])
    raise KeyError(f"no capability_mmlu in cell (alpha={cell.get('alpha')})")


def efficacy_items(cell):
    return cell.get("efficacy_per_prompt")


def capability_items(cell):
    return cell.get("capability_mmlu_items")


def is_length_cells(cells) -> bool:
    """True iff every cell carries the length target's per-PROMPT arrays
    (length_tokens_per_prompt + failure_flags, harness prereg sec.8 output)."""
    return bool(cells) and all(
        c.get("length_tokens_per_prompt") is not None
        and c.get("failure_flags") is not None for c in cells)


def _length_eff_reducer(cells):
    """Per-draw efficacy reducer for the LENGTH target (2026-07-07 fix).

    The harness's `efficacy_per_prompt` for length holds SURVIVORS ONLY, whose
    composition differs per alpha, so the old bootstrap (a) paired unrelated
    prompts across alphas (breaking the S12 pairing premise), (b) silently
    truncated survivor tails to the min count, and (c) bootstrapped a MEAN
    around a MEDIAN point estimate. Fix: resample PROMPT indices from the full
    per-prompt arrays, drop that draw's failures, and recompute the survivor
    MEDIAN -- the same statistic as the point estimate, with censoring
    propagated into every draw."""
    toks = [np.asarray(c["length_tokens_per_prompt"], float) for c in cells]
    flags = [np.asarray(c["failure_flags"], bool) for c in cells]
    n = min(len(t) for t in toks)

    def reduce(j, idx):
        t, f = toks[j][:n][idx], flags[j][:n][idx]
        s = t[~f]
        return float(np.median(s)) if s.size else None
    return n, reduce


def _judged_eff_reducer(cells):
    """Per-draw efficacy reducer for judged targets: mean of resampled
    per-prompt judge scores (unchanged legacy behavior)."""
    eff_it = [efficacy_items(c) for c in cells]
    if any(x is None for x in eff_it):
        return 0, None
    eff_it = [np.asarray(x, float) for x in eff_it]
    n = min(len(x) for x in eff_it) if eff_it else 0

    def reduce(j, idx):
        return float(eff_it[j][:n][idx].mean())
    return n, reduce


def _eff_reducer(cells):
    """Return (n_prompts, reduce(alpha_idx, prompt_idx)->float|None, kind)."""
    if is_length_cells(cells):
        n, red = _length_eff_reducer(cells)
        return n, red, "length"
    n, red = _judged_eff_reducer(cells)
    return n, red, "judged"


# _arm() and _first_crossing() moved VERBATIM to steerquant_estar.py
# (2026-07-11 extraction) and are imported above -- the harness's online
# capability-alpha selection uses the SAME functions.


def iecc_from_arrays(alphas, eff, cap, frac, min_shift=0.05,
                     arm_sign=None, e_star_level=None, rel_min_shift=None):
    """IECC on one cell's curve. S11: `arm_sign` fixes the arm and
    `e_star_level` fixes the absolute reference efficacy E* (both derived once
    from the fp16 point estimate); when None, the legacy free-arm + per-curve E*
    behavior is used. `rel_min_shift`: weak-signal guard as a FRACTION of the
    baseline, for the token-scale length target where the absolute judge-scale
    min_shift=0.05 is meaningless (2026-07-07)."""
    absa, sa, e, c, base, sign = _arm(alphas, eff, cap, fixed_sign=arm_sign)
    extreme = float(e[int(np.argmax(np.abs(e - base)))])
    level = e_star_level if e_star_level is not None else base + frac * (extreme - base)
    a_star = _first_crossing(absa, e, level)
    if rel_min_shift is not None:
        weak = bool(abs(extreme - base) < rel_min_shift * abs(base))
    else:
        weak = bool(abs(extreme - base) < min_shift)
    out = {"E_star": level, "baseline_eff": base, "sign": sign,
           "max_shift": extreme - base,
           "weak_signal": weak}
    if a_star is None:
        out["reached"] = False
        return out
    # 2026-07-11 adaptive-capability files: capability may be missing (NaN) at
    # far alphas; interpolate over the alphas that HAVE a capability read. For
    # legacy full-grid files the mask is all-True and this is byte-identical.
    m = np.isfinite(np.asarray(c, float))
    if not m.any():
        out["reached"] = False
        out["no_capability"] = True
        return out
    cap0 = float(np.interp(0.0, absa[m], c[m]))
    cap_at = float(np.interp(a_star, absa[m], c[m]))
    out.update({"reached": True, "abs_alpha_star": a_star, "alpha_star": sign * a_star,
                "cap_baseline": cap0, "cap_at_star": cap_at, "iecc": cap0 - cap_at,
                "n_cap_alphas": int(m.sum())})
    return out


def point_iecc(cells, frac, min_shift=0.05, arm_sign=None, e_star_level=None):
    alphas = [c["alpha"] for c in cells]
    eff = np.array([efficacy_mean(c) for c in cells], float)
    # 2026-07-11: adaptive-capability files carry capability only at the probed
    # alphas -- missing entries become NaN (interpolated over downstream).
    cap = np.array([float(c["capability_mmlu"]) if c.get("capability_mmlu") is not None
                    else np.nan for c in cells], float)
    rel = 0.05 if is_length_cells(cells) else None  # 5%-of-baseline guard on token scale
    return iecc_from_arrays(alphas, eff, cap, frac, min_shift,
                            arm_sign=arm_sign, e_star_level=e_star_level,
                            rel_min_shift=rel)


def bootstrap_iecc(cells, frac, n_boot, rng, arm_sign=None, e_star_level=None):
    # S12: the SAME eval prompts and MMLU items are used at every alpha, so pair
    # the resample across alphas -- ONE prompt-index vector and ONE capability-
    # index vector per draw, reused at every alpha. 2026-07-07: efficacy now goes
    # through _eff_reducer -- judged targets keep the per-prompt MEAN; the length
    # target resamples PROMPTS from the full per-prompt arrays and recomputes the
    # survivor MEDIAN per draw (efficacy_per_prompt is survivors-only for length
    # and must not be index-paired across alphas).
    n_eff, eff_red, eff_kind = _eff_reducer(cells)
    cap_it = [capability_items(c) for c in cells]
    # 2026-07-11: adaptive-capability files have per-item arrays only at the
    # probed alphas; missing alphas enter each draw as NaN and iecc_from_arrays
    # interpolates over the probed ones. ALL-missing -> no bootstrap (legacy).
    if eff_red is None or n_eff == 0 or all(x is None for x in cap_it):
        return None
    alphas = [c["alpha"] for c in cells]
    cap_it = [np.asarray(x, float) if x is not None else None for x in cap_it]
    n_cap = min((len(x) for x in cap_it if x is not None), default=0)
    rel = 0.05 if eff_kind == "length" else None
    draws, reached, no_survivor = [], 0, 0
    for _ in range(n_boot):
        pi = rng.integers(0, n_eff, n_eff)
        ci = rng.integers(0, n_cap, n_cap) if n_cap else None
        eff_vals = [eff_red(j, pi) for j in range(len(cells))]
        if any(v is None for v in eff_vals):
            no_survivor += 1   # a resample left some alpha with zero survivors
            continue
        eff = np.array(eff_vals, float)
        cap = np.array([((x[:n_cap][ci].mean() if n_cap else 0.0)
                         if x is not None else np.nan) for x in cap_it], float)
        r = iecc_from_arrays(alphas, eff, cap, frac,
                             arm_sign=arm_sign, e_star_level=e_star_level,
                             rel_min_shift=rel)
        if r["reached"]:
            draws.append(r["iecc"]); reached += 1
    if not draws:
        return {"reached_frac": 0.0, "n_boot": n_boot}
    arr = np.array(draws, float)
    out = {"mean": float(arr.mean()), "lo": float(np.percentile(arr, 2.5)),
           "hi": float(np.percentile(arr, 97.5)), "reached_frac": reached / n_boot,
           "n_boot": n_boot,
           "var": float(arr.var(ddof=1)) if arr.size > 1 else 0.0,
           "eff_stat": "survivor_median" if eff_kind == "length" else "mean"}
    if no_survivor:
        out["no_survivor_frac"] = no_survivor / n_boot
    return out


# --- Prereg sec.7 E* acceptance rule (auto-evaluated; no researcher discretion)
# E_STAR_LADDER / DELTA_MIN moved to steerquant_estar.py (2026-07-11) and are
# imported at the top -- ONE encoding of the rule, shared with the harness's
# online capability-alpha selection. accept_e_star_frac itself stays here: the
# ACCEPTANCE step needs capability data, which only exists post-hoc.


def accept_e_star_frac(fp16_cells, ladder=E_STAR_LADDER, delta_min=DELTA_MIN):
    """Prereg sec.7: E* = the FIRST fraction in 0.5 -> 0.4 -> 0.3 whose FP16
    (clean-model) IECC at E* is < delta_min. Auto-evaluated on the fp16 curve so
    no discretion remains (2026-07-07; previously this rule existed only in the
    prereg text and the fraction was a manual CLI flag). Returns {frac,
    accepted, trail}; `trail` records each fraction tried with its fp16 IECC
    (the audit line for the report/paper). accepted=False -> even 0.3 costs
    >= delta_min on fp16; E* is OUT of the reliable band and confirmatory IECC
    for this cell must not be claimed."""
    trail = []
    for f in ladder:
        pt = point_iecc(fp16_cells, f)
        trail.append({"frac": f, "reached": bool(pt.get("reached")),
                      "fp16_iecc": pt.get("iecc")})
        if pt.get("reached") and pt["iecc"] < delta_min:
            return {"frac": f, "accepted": True, "trail": trail}
    return {"frac": ladder[-1], "accepted": False, "trail": trail}


# H1 contrast (real data) + a priori power analysis (prereg sec.7)
Z_ONE_SIDED_95 = 1.6448536269514722   # z for a one-sided 95% bound; TOST uses a 90% CI
Z_TWO_SIDED_95 = 1.959963984540054    # z for a two-sided 95% CI (H2 Rule A)


def h1_three_label(lo90, hi90, delta=0.03):
    """Prereg sec.H1 three-label outcome (Saurav 2026-07-02) on the pooled 90% CI
    of the contrast (IECC_scheme - IECC_fp16; positive = the scheme costs more).
      Equivalent         : CI entirely within [-delta, +delta]
      Meaningfully Worse : CI entirely ABOVE +delta (degradation side)
      Inconclusive       : CI straddles a +/-delta boundary
    A CI entirely BELOW -delta (scheme better than fp16 beyond the margin) is
    not one of the prereg's three labels; returned explicitly as descriptive so
    it cannot be silently mislabeled Equivalent."""
    if lo90 >= -delta and hi90 <= delta:
        return "Equivalent"
    if lo90 > delta:
        return "Meaningfully Worse"
    if hi90 < -delta:
        return "OutsideMargin_Better (descriptive, not an H1 label)"
    return "Inconclusive"


def h2_rule_a(lo95, delta_min=DELTA_MIN):
    """Prereg sec.H2 Rule A (Saurav 2026-07-02): support iff the pooled 95% CI
    LOWER bound exceeds delta_min -- the whole CI sits above the 10% line."""
    return bool(lo95 > delta_min)


def _normal_cdf(x):
    return 0.5 * (1.0 + _erf(x / np.sqrt(2.0)))


def iecc_se(n_items, p=0.74, paired=True, rho=0.5):
    var_one = p * (1.0 - p) / n_items
    var_diff = 2.0 * var_one * (1.0 - rho) if paired else 2.0 * var_one
    return float(np.sqrt(var_diff))


def contrast_se(n_items, p=0.74, paired=True, rho=0.5):
    return float(np.sqrt(2.0) * iecc_se(n_items, p, paired, rho))


def pooled_se(within_se, tau, k):
    return float(np.sqrt((within_se ** 2 + tau ** 2) / max(k, 1)))


def tost_power(delta, se, true_gap=0.0, z=Z_ONE_SIDED_95):
    if se <= 0:
        return 1.0 if abs(true_gap) < delta else 0.0
    hi = (delta - z * se - true_gap) / se
    lo = (-delta + z * se - true_gap) / se
    return float(max(0.0, _normal_cdf(hi) - _normal_cdf(lo)))


def breakeven_tau(delta, within_se, k, z=Z_ONE_SIDED_95):
    rhs = (delta / z) ** 2 * max(k, 1) - within_se ** 2
    return float(np.sqrt(rhs)) if rhs > 0 else None


def power_report(p=0.74, delta=0.03, k=16, paired=True, rho=0.5,
                 n_grid=(100, 200, 400, 800, 1600), taus=(0.0, 0.01, 0.02, 0.03)):
    z = Z_ONE_SIDED_95
    print("=" * 78)
    print(f"H1 equivalence power   p(acc)={p}  delta(+/-)={delta}  K_cells={k}  rho={rho}")
    print("  TOST uses a 90% CI; half-width = 1.645 * SE. 'Feasible' needs half-width < delta.")
    print("=" * 78)
    print(f"{'N/cell':>7} {'contrastSE':>11} {'1cell_HW':>9} {'1cell_pow':>10} {'breakeven_tau':>14}")
    for n in n_grid:
        s_c = contrast_se(n, p, paired, rho)
        bt = breakeven_tau(delta, s_c, k)
        bts = f"{bt:.4f}" if bt is not None else "none (need N)"
        print(f"{n:>7} {s_c:>11.4f} {z*s_c:>9.4f} {tost_power(delta, s_c):>10.2f} {bts:>14}")
    print("\nPooled over K cells -- 90% CI half-width by (N, tau):")
    print("   N \\ tau " + "".join(f"{t:>9.3f}" for t in taus))
    for n in n_grid:
        s_c = contrast_se(n, p, paired, rho)
        print(f"  {n:>6} " + "".join(f"{z*pooled_se(s_c, t, k):>9.4f}" for t in taus))
    print(f"\nRead: a pooled half-width below delta={delta} means equivalence is in reach at")
    print("that (N, tau). tau is the BETWEEN-model SD of the contrast, unknown until >=3-4")
    print("models are run; estimate a prior from per-model tables in the quant/steering lit.")
    print("=" * 78)


def bootstrap_contrast(fp16_cells, wo_cells, frac, n_boot, rng, ci=0.90, min_shift=0.05,
                       arm_sign=None, e_star_level=None, legacy_arm=False):
    fp = {c["alpha"]: c for c in fp16_cells}
    wo = {c["alpha"]: c for c in wo_cells}
    alphas = sorted(set(fp) & set(wo))
    if not alphas:
        return None
    # S11: fix the arm + E* ONCE from the fp16 POINT estimate, then apply the SAME
    # arm/E* to BOTH fp16 and the scheme in every draw -- otherwise the contrast can
    # compare opposite arms (prereg sec.7). legacy_arm=True keeps the old free-arm rule.
    if not legacy_arm and (arm_sign is None or e_star_level is None):
        ref = point_iecc(fp16_cells, frac, min_shift)
        if ref.get("reached"):
            if arm_sign is None:
                arm_sign = ref["sign"]
            if e_star_level is None:
                e_star_level = ref["E_star"]

    def cols(cbya, key):
        # 2026-07-11: per-alpha None allowed (adaptive-capability files); the
        # missing alphas enter each draw as NaN and iecc_from_arrays
        # interpolates over the probed ones. A scheme with NO capability at
        # all still -> None (no contrast possible).
        out = []
        for a in alphas:
            v = cbya[a].get(key)
            out.append(np.asarray(v, float) if v is not None else None)
        return None if all(x is None for x in out) else out

    fcap, wcap = cols(fp, "capability_mmlu_items"), cols(wo, "capability_mmlu_items")
    if any(x is None for x in (fcap, wcap)):
        return None
    # 2026-07-07: efficacy via _eff_reducer -- judged targets keep the per-prompt
    # mean; the length target resamples PROMPTS and recomputes the survivor
    # MEDIAN (efficacy_per_prompt is survivors-only for length; index-pairing it
    # across alphas/schemes was invalid).
    fp_sh = [fp[a] for a in alphas]
    wo_sh = [wo[a] for a in alphas]
    n_f, red_f, kind_f = _eff_reducer(fp_sh)
    n_w, red_w, kind_w = _eff_reducer(wo_sh)
    if red_f is None or red_w is None:
        return None
    rel = 0.05 if kind_f == "length" else None
    # S12: one capability-index vector and one prompt-index vector PER DRAW, reused
    # across every alpha AND across fp16/scheme (the same items/prompts are evaluated
    # at every alpha and shared between schemes). Common sizes taken as the min across
    # alphas and both schemes so a single index vector is valid everywhere.
    n_cap = min(len(x) for x in list(fcap) + list(wcap) if x is not None)
    n_eff = min(n_f, n_w)
    if n_cap == 0 or n_eff == 0:
        return None
    draws, skipped = [], 0
    for _ in range(n_boot):
        ci_idx = rng.integers(0, n_cap, n_cap)
        pi_idx = rng.integers(0, n_eff, n_eff)
        fe = [red_f(j, pi_idx) for j in range(len(alphas))]
        we = [red_w(j, pi_idx) for j in range(len(alphas))]
        if any(v is None for v in fe) or any(v is None for v in we):
            skipped += 1   # a resample left some alpha with zero survivors (length)
            continue
        fc = [float(fcap[j][:n_cap][ci_idx].mean()) if fcap[j] is not None
              else np.nan for j in range(len(alphas))]
        wc = [float(wcap[j][:n_cap][ci_idx].mean()) if wcap[j] is not None
              else np.nan for j in range(len(alphas))]
        rf = iecc_from_arrays(alphas, np.array(fe), np.array(fc), frac, min_shift,
                              arm_sign=arm_sign, e_star_level=e_star_level,
                              rel_min_shift=rel)
        rw = iecc_from_arrays(alphas, np.array(we), np.array(wc), frac, min_shift,
                              arm_sign=arm_sign, e_star_level=e_star_level,
                              rel_min_shift=rel)
        if rf["reached"] and rw["reached"]:
            draws.append(rw["iecc"] - rf["iecc"])
    if not draws:
        return None
    arr = np.array(draws, float)
    lo_q, hi_q = (1 - ci) / 2 * 100, (1 + ci) / 2 * 100
    out = {"contrast_mean": float(arr.mean()), "lo": float(np.percentile(arr, lo_q)),
           "hi": float(np.percentile(arr, hi_q)), "ci_level": ci, "n_used": len(draws),
           "var": float(arr.var(ddof=1)) if arr.size > 1 else 0.0,
           "eff_stat": "survivor_median" if kind_f == "length" else "mean"}
    if skipped:
        out["no_survivor_frac"] = skipped / n_boot
    return out


# ============================================================================
# Option C run-level combiner (Saurav ruling 2026-07-12; instantiated
# 2026-07-14 -- METHOD_NOTE_optionC-REML-HK_2026-07-14.md). The 5 physical
# resample runs per (model, scheme) form a small random-effects meta-analysis:
# y_r = per-run point contrast (or IECC), v_r = its within-run bootstrap
# variance, between-run (vector) variance tau2 by REML, CI by MODIFIED
# Knapp-Hartung with t at k-1 df. Under the literal-sec.4 files (runs
# resampled pairs AND prompts AND items; Saurav: the files stand), each v_r
# already carries the prompt/item noise, so REML's excess-variance tau2 is the
# vector component -- every noise source counted exactly once.
# ============================================================================

# two-sided t quantiles: T_TABLE[df] = (t_{df,0.95}, t_{df,0.975});
# .95 column -> 90% two-sided CI (H1/TOST), .975 -> 95% CI (H2 Rule A).
# df > 10 uses the df=10 row (larger t = conservative; k=5 runs -> df=4).
T_TABLE = {1: (6.313752, 12.706205), 2: (2.919986, 4.302653),
           3: (2.353363, 3.182446), 4: (2.131847, 2.776445),
           5: (2.015048, 2.570582), 6: (1.943180, 2.446912),
           7: (1.894579, 2.364624), 8: (1.859548, 2.306004),
           9: (1.833113, 2.262157), 10: (1.812461, 2.228139)}


def reml_tau2(est, var, max_iter=200, tol=1e-12):
    """REML estimate of the between-run variance tau2 in y_r ~ N(mu, v_r + tau2).

    Iterative fixed-point (Viechtbauer 2005): with w_r = 1/(v_r + tau2),
      tau2 <- sum(w^2 [(y - mu_hat)^2 - v]) / sum(w^2) + 1 / sum(w),
    truncated at 0 each step. Balanced case (all v equal) has the closed form
    tau2 = max(0, S^2 - mean(v)) with S^2 the sample variance of y (exact;
    selftest [18]). Verified against a brute-force grid maximization of the
    REML log-likelihood (2026-07-14, sandbox). Returns (tau2, converged)."""
    y = np.asarray(est, float)
    v = np.asarray(var, float)
    if y.size < 2:
        return 0.0, True
    t2 = max(0.0, float(np.var(y, ddof=1) - v.mean()))   # moment start
    for _ in range(max_iter):
        w = 1.0 / (v + t2)
        mu = float(np.sum(w * y) / np.sum(w))
        new = float(np.sum(w ** 2 * ((y - mu) ** 2 - v)) / np.sum(w ** 2)
                    + 1.0 / np.sum(w))
        new = max(0.0, new)
        if abs(new - t2) < tol:
            return new, True
        t2 = new
    return t2, False


def hk_ci(est, var):
    """MODIFIED Knapp-Hartung combination of per-run estimates.

    w_r = 1/(v_r + tau2_REML); mu_hat = sum(w y)/sum(w);
    q_hat = sum(w (y - mu_hat)^2) / (k-1);
    SE^2(mu_hat) = max(q_hat, 1) / sum(w)   [the max() is the MODIFIED part:
    it guards the q_hat < 1 case where plain KH is NARROWER than the
    conventional normal CI -- never anti-conservative];
    CI = mu_hat +/- t_{k-1} * SE with t from T_TABLE."""
    y = np.asarray(est, float)
    v = np.asarray(var, float)
    k = int(y.size)
    if k < 2:
        raise ValueError("hk_ci needs >= 2 runs")
    tau2, conv = reml_tau2(y, v)
    w = 1.0 / (v + tau2)
    mu = float(np.sum(w * y) / np.sum(w))
    df = k - 1
    q = float(np.sum(w * (y - mu) ** 2) / df)
    se = float(np.sqrt(max(q, 1.0) / np.sum(w)))
    t90, t95 = T_TABLE[min(df, 10)]
    return {"mu": mu, "se": se, "tau2": tau2, "tau": float(np.sqrt(tau2)),
            "k": k, "df": df, "q": q,
            "hk_modified_applied": bool(q < 1.0), "reml_converged": bool(conv),
            "ci90_lo": mu - t90 * se, "ci90_hi": mu + t90 * se,
            "ci95_lo": mu - t95 * se, "ci95_hi": mu + t95 * se}


def _run_key(path, meta):
    """Run id for the Option C combiner: meta.resample.run (S10 record),
    else the _r<k>_ filename convention; None = unassignable."""
    r = (meta.get("resample") or {}).get("run")
    if r is not None:
        return int(r)
    m = re.search(r"_r(\d+)_", Path(str(path)).name)
    return int(m.group(1)) if m else None


def combine_runs_group(metas, n_boot, seed=20260714, delta=0.03):
    """Option C run-level combiner for ONE (model, target) group.

    Files are split by run r (meta.resample.run, filename fallback). PER RUN:
    the prereg sec.7 E* ladder is evaluated on THAT run's fp16 curve, the S11
    arm/E* reference is fixed from THAT run's fp16 point estimate, and each
    scheme's run-paired contrast y_r = IECC_scheme - IECC_fp16 (point) is
    computed with its within-run bootstrap variance v_r (S12 paired draws).
    ACROSS RUNS, per scheme: REML tau2 + modified Knapp-Hartung CI (t, k-1 df)
    -> the cell's confirmatory contrast; H1 three-label on the KH 90% CI.
    fp16's own IECC is combined the same way (descriptive). Runs whose fp16
    fails the sec.7 ladder or never reaches E* are EXCLUDED and named, as are
    duplicate (scheme, run) files (masquerade guard, 07-12 lesson)."""
    runs, unassigned = {}, []
    for t in metas:
        r = _run_key(t[0], t[1])
        (runs.setdefault(r, []) if r is not None else unassigned).append(t)
    notes = [f"file without a run id EXCLUDED from the combine: {p}"
             for (p, _m, _c) in unassigned]
    per_scheme, fp16_runs = {}, []
    for r in sorted(runs):
        group = runs[r]
        fps = [(p, m, c) for (p, m, c) in group if m.get("scheme") == "fp16"]
        if len(fps) > 1:
            notes.append(f"run r={r}: {len(fps)} fp16 files; using {fps[0][0]}, "
                         f"IGNORING the rest (duplicate guard)")
        fp = fps[0] if fps else None
        if fp is None:
            notes.append(f"run r={r}: no fp16 file -> run EXCLUDED")
            continue
        acc = accept_e_star_frac(fp[2])
        if not acc["accepted"]:
            notes.append(f"run r={r}: sec.7 E* ladder NOT accepted on fp16 -> run EXCLUDED")
            continue
        frac = acc["frac"]
        ref = point_iecc(fp[2], frac)
        if not ref.get("reached"):
            notes.append(f"run r={r}: fp16 E* not reached -> run EXCLUDED")
            continue
        sign, level = ref["sign"], ref["E_star"]
        bs_fp = bootstrap_iecc(fp[2], frac, n_boot,
                               np.random.default_rng([seed, r, 0]),
                               arm_sign=sign, e_star_level=level)
        if bs_fp is not None and bs_fp.get("var") is not None:
            fp16_runs.append({"run": r, "frac": frac, "est": float(ref["iecc"]),
                              "var": float(bs_fp["var"]),
                              "alpha_star": float(ref["alpha_star"])})
        seen = set()
        for (p, m, c) in group:
            sch = m.get("scheme")
            if sch == "fp16":
                continue
            if sch in seen:
                notes.append(f"run r={r} scheme={sch}: DUPLICATE file {p} IGNORED")
                continue
            seen.add(sch)
            pt = point_iecc(c, frac, arm_sign=sign, e_star_level=level)
            bc = bootstrap_contrast(fp[2], c, frac, n_boot,
                                    np.random.default_rng([seed, r, 1]),
                                    arm_sign=sign, e_star_level=level)
            if not pt.get("reached") or bc is None or bc.get("var") is None:
                notes.append(f"run r={r} scheme={sch}: contrast unavailable "
                             f"-> this run dropped for {sch}")
                continue
            per_scheme.setdefault(sch, []).append(
                {"run": r, "frac": frac,
                 "est": float(pt["iecc"] - ref["iecc"]),
                 "var": float(bc["var"]),
                 "boot_mean": float(bc["contrast_mean"])})
    out = {"fp16_iecc": None, "schemes": {}, "notes": notes}
    if len(fp16_runs) >= 2:
        hk = hk_ci([d["est"] for d in fp16_runs], [d["var"] for d in fp16_runs])
        hk["per_run"] = fp16_runs
        out["fp16_iecc"] = hk
    for sch in sorted(per_scheme):
        d = per_scheme[sch]
        if len(d) < 2:
            notes.append(f"scheme={sch}: <2 usable runs -> no combined estimate")
            continue
        hk = hk_ci([x["est"] for x in d], [x["var"] for x in d])
        hk["per_run"] = d
        hk["h1_label"] = h1_three_label(hk["ci90_lo"], hk["ci90_hi"], delta)
        out["schemes"][sch] = hk
    return out


def _print_combined(res):
    print("Option C combine: REML tau2 + modified Knapp-Hartung over runs "
          "(METHOD_NOTE_optionC-REML-HK_2026-07-14.md)")
    for n in res["notes"]:
        print(f"  NOTE: {n}")
    fp = res.get("fp16_iecc")
    if fp is not None:
        runs = ",".join(str(d["run"]) for d in fp["per_run"])
        fracs = sorted({d["frac"] for d in fp["per_run"]})
        print(f"  fp16 IECC (descriptive): {fp['mu']:+.4f}  "
              f"95% CI [{fp['ci95_lo']:+.4f},{fp['ci95_hi']:+.4f}]  "
              f"tau={fp['tau']:.4f}  runs=[{runs}]  E* frac(s)={fracs}")
    for sch, hk in sorted(res["schemes"].items()):
        per = "  ".join(f"r{d['run']}:{d['est']:+.4f}" for d in hk["per_run"])
        print(f"\n  {sch}  contrast IECC(scheme) - IECC(fp16), k={hk['k']} runs")
        print(f"    per-run: {per}")
        print(f"    tau2(REML)={hk['tau2']:.6f}  q={hk['q']:.3f}"
              f"{'  [modified KH applied]' if hk['hk_modified_applied'] else ''}"
              f"{'' if hk['reml_converged'] else '  [REML NOT CONVERGED]'}")
        print(f"    combined {hk['mu']:+.4f}  90% CI [{hk['ci90_lo']:+.4f},"
              f"{hk['ci90_hi']:+.4f}]  95% CI [{hk['ci95_lo']:+.4f},"
              f"{hk['ci95_hi']:+.4f}]  (t, df={hk['df']})")
        print(f"    H1 three-label: {hk['h1_label']}")


def _combine_report(groups, n_boot):
    """--combine-runs driver: Option C per (model, target) group, then the
    prereg sec.2A cross-model pool per (target, scheme) when >= 2 models are
    present. Rows returned carry (model, target, scheme, est, se, hk) -- the
    inputs the full K=12 pool consumes."""
    pool_rows = []
    for key in sorted(groups, key=str):
        print(f"\n{'#' * 78}\nGROUP  model={key[0]}  target={key[1]}  "
              f"[Option C run-level combine]\n{'#' * 78}")
        res = combine_runs_group(groups[key], n_boot)
        _print_combined(res)
        for sch, hk in sorted(res["schemes"].items()):
            pool_rows.append({"model": key[0], "target": key[1], "scheme": sch,
                              "est": hk["mu"], "se": hk["se"], "hk": hk})
    bysch = {}
    for row in pool_rows:
        bysch.setdefault((row["target"], row["scheme"]), []).append(row)
    for (tgt, sch), rws in sorted(bysch.items(), key=str):
        if len(rws) < 2:
            continue
        hp = heterogeneity_pool([{"model": w["model"], "target": tgt,
                                  "est": w["est"], "se": w["se"]} for w in rws])
        print(f"\nCROSS-MODEL POOL  target={tgt}  scheme={sch}  "
              f"models={[w['model'] for w in rws]}")
        print(f"  pooled={hp['pooled']:+.4f}  90% CI [{hp['ci90_lo']:+.4f},"
              f"{hp['ci90_hi']:+.4f}]  95% CI [{hp['ci95_lo']:+.4f},"
              f"{hp['ci95_hi']:+.4f}]  tau_model={hp['tau_model']:.4f}  "
              f"I2={hp['I2']:.2f}  branch={hp['branch']}")
        print(f"  H1 (sec.2A gate): {hp['h1_label']}"
              + ("" if hp["confirmatory"] else "  [gate FAILED -> descriptive only]")
              + ("  [k<12: subset of the K=12 pool -- confirmatory H1 is the "
                 "full pool]" if hp["k"] < 12 else ""))
    return pool_rows


# ============================================================================
# Reasoning-length target (prereg sec.8) + K=16 pooling/heterogeneity (sec.2A)
# ============================================================================
TAU_THRESHOLD = 0.058   # H1 breakeven; OPERATIONAL, not literature-derived
I2_THRESHOLD = 0.50


# --- (a) Failure-rate endpoint (co-primary, sec.8) --------------------------
def failure_rate(flags) -> float:
    """Fraction of traces flagged as termination failures
    (flags from termination_failure_detector.detect_termination_failure)."""
    f = np.asarray(flags, float)
    return float(f.mean()) if f.size else float("nan")


def bootstrap_failure_rate(flags, n_boot, rng, ci=0.95):
    """Point failure rate + bootstrap CI over traces."""
    f = np.asarray(flags, float)
    if f.size == 0:
        return None
    draws = np.array([f[rng.integers(0, f.size, f.size)].mean() for _ in range(n_boot)])
    lo_q, hi_q = (1 - ci) / 2 * 100, (1 + ci) / 2 * 100
    return {"rate": float(f.mean()), "n": int(f.size),
            "lo": float(np.percentile(draws, lo_q)),
            "hi": float(np.percentile(draws, hi_q)), "ci_level": ci}


def failure_rate_contrast(flags_scheme, flags_fp16, n_boot, rng, ci=0.95):
    """rate(scheme) - rate(fp16) with a bootstrap CI. A material, CI-excludes-0
    gap is itself a degradation finding (sec.8 failure-rate rule)."""
    a, b = np.asarray(flags_scheme, float), np.asarray(flags_fp16, float)
    if a.size == 0 or b.size == 0:
        return None
    draws = np.array([a[rng.integers(0, a.size, a.size)].mean()
                      - b[rng.integers(0, b.size, b.size)].mean() for _ in range(n_boot)])
    lo_q, hi_q = (1 - ci) / 2 * 100, (1 + ci) / 2 * 100
    lo, hi = float(np.percentile(draws, lo_q)), float(np.percentile(draws, hi_q))
    return {"diff": float(a.mean() - b.mean()), "lo": lo, "hi": hi,
            "ci_level": ci, "excludes_0": bool(lo > 0 or hi < 0)}


def failure_rates_comparable(flags_a, flags_b, n_boot, rng, ci=0.95):
    """True iff the failure-rate contrast CI includes 0 -- the precondition for
    INTERPRETING a length comparison between the two cells (sec.8)."""
    c = failure_rate_contrast(flags_a, flags_b, n_boot, rng, ci)
    return None if c is None else (not c["excludes_0"])


# --- (b) Length endpoint + difference-in-differences (sec.8) ----------------
def cell_length_efficacy(token_counts, failure_flags) -> float:
    """Median generated-token count over NON-FAILURE traces only. This is the
    judge-free efficacy value; it plugs into iecc_from_arrays as `eff`."""
    t = np.asarray(token_counts, float)
    f = np.asarray(failure_flags, bool)
    survivors = t[~f]
    return float(np.median(survivors)) if survivors.size else float("nan")


def length_delta(cells_by_alpha):
    """Delta_len(alpha) = median_tokens(alpha) - median_tokens(alpha=0), over
    survivors. `cells_by_alpha`: list of {alpha, token_counts, failure_flags}."""
    alphas = np.array([c["alpha"] for c in cells_by_alpha], float)
    med = np.array([cell_length_efficacy(c["token_counts"], c["failure_flags"])
                    for c in cells_by_alpha], float)
    base = float(med[int(np.argmin(np.abs(alphas)))])
    return alphas, med, med - base


def length_diff_in_differences(scheme_cells, fp16_cells):
    """Iso-effect length DiD: at each shared alpha, the SECOND difference
    [Delta_len(alpha)]_scheme - [Delta_len(alpha)]_fp16. Baseline-verbosity
    shifts cancel via the within-scheme first difference, so the DiD isolates
    the scheme's effect on the length response (sec.8)."""
    a_s, _, d_s = length_delta(scheme_cells)
    a_f, _, d_f = length_delta(fp16_cells)
    shared = sorted(set(a_s.tolist()) & set(a_f.tolist()))
    ms = {a: d for a, d in zip(a_s.tolist(), d_s.tolist())}
    mf = {a: d for a, d in zip(a_f.tolist(), d_f.tolist())}
    return {"alphas": shared, "did": [ms[a] - mf[a] for a in shared]}


# --- (c) K=16 random-effects pooling + heterogeneity gate (sec.2A) ----------
def _dl_pool(est, se):
    """DerSimonian-Laird random-effects pool -> (pooled, se_pooled, tau2, I2, Q)."""
    est = np.asarray(est, float)
    se = np.asarray(se, float)
    w = 1.0 / se ** 2
    fixed = float(np.sum(w * est) / np.sum(w))
    Q = float(np.sum(w * (est - fixed) ** 2))
    df = len(est) - 1
    C = float(np.sum(w) - np.sum(w ** 2) / np.sum(w))
    tau2 = max(0.0, (Q - df) / C) if C > 0 and df > 0 else 0.0
    w_star = 1.0 / (se ** 2 + tau2)
    pooled = float(np.sum(w_star * est) / np.sum(w_star))
    se_pooled = float(np.sqrt(1.0 / np.sum(w_star)))
    I2 = max(0.0, (Q - df) / Q) if Q > 0 and df > 0 else 0.0
    return pooled, se_pooled, tau2, I2, Q


def _group_tau(estimates, key):
    """Between-GROUP SD (tau_model or tau_target): DL-pool each group to a
    mean+SE, then DL-pool the group means; the resulting tau is the
    between-group SD. Operationalizes sec.2A's separate tau_model/tau_target."""
    groups = {}
    for e in estimates:
        groups.setdefault(e[key], []).append(e)
    g_est, g_se = [], []
    for members in groups.values():
        p, sep, _, _, _ = _dl_pool([m["est"] for m in members],
                                   [m["se"] for m in members])
        g_est.append(p)
        g_se.append(sep)
    if len(g_est) < 2:
        return 0.0
    _, _, tau2, _, _ = _dl_pool(g_est, g_se)
    return float(np.sqrt(tau2))


def heterogeneity_pool(estimates, delta=0.03, tau_thr=TAU_THRESHOLD,
                       i2_thr=I2_THRESHOLD):
    """Apply the prereg sec.2A pooling gate.

    `estimates`: list of {model, target, est, se} for the K cells (est = the
    per-cell IECC contrast vs FP16; se = its within-cell SE). Returns the pooled
    K=16 estimate + 90% CI, tau_model, tau_target, I^2, and the branch decision.
    Fallback branches are DESCRIPTIVE, not confirmatory.
    """
    pooled, se_pooled, tau2, I2, Q = _dl_pool([e["est"] for e in estimates],
                                              [e["se"] for e in estimates])
    tau_model = _group_tau(estimates, "model")
    tau_target = _group_tau(estimates, "target")
    model_ok, target_ok, i2_ok = tau_model <= tau_thr, tau_target <= tau_thr, I2 < i2_thr
    if i2_ok and model_ok and target_ok:
        branch, confirmatory = "pooled_K16", True
    elif (not model_ok) and target_ok:
        branch, confirmatory = "per_model", False
    elif model_ok and (not target_ok):
        branch, confirmatory = "per_target", False
    else:
        branch, confirmatory = "per_cell", False
    ci90_lo = pooled - Z_ONE_SIDED_95 * se_pooled
    ci90_hi = pooled + Z_ONE_SIDED_95 * se_pooled
    ci95_lo = pooled - Z_TWO_SIDED_95 * se_pooled
    ci95_hi = pooled + Z_TWO_SIDED_95 * se_pooled
    label = h1_three_label(ci90_lo, ci90_hi, delta)
    return {"branch": branch, "confirmatory": confirmatory,
            "k": len(estimates),  # 16 pre-deviation; 12 after the 07-06 K-pool deviation
            "pooled": pooled, "se_pooled": se_pooled,
            "ci90_lo": ci90_lo, "ci90_hi": ci90_hi,
            "ci95_lo": ci95_lo, "ci95_hi": ci95_hi,
            "tau2": tau2, "I2": I2, "Q": Q,
            "tau_model": tau_model, "tau_target": tau_target,
            # Prereg decision rules, ENCODED (2026-07-07). h1_label / h2 are
            # confirmatory ONLY when the gate passes (branch == pooled_K16);
            # on fallback branches they are descriptive.
            "h1_label": label,
            "h2_rule_a_supported": h2_rule_a(ci95_lo),
            "equivalence_supported": bool(label == "Equivalent")}


def load_result(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data.get("meta", {}), data.get("by_alpha", [])


def analyze_file(path, frac, n_boot, seed=12345, arm_sign=None, e_star_level=None):
    meta, cells = load_result(path)
    rng = np.random.default_rng(seed)
    return {"path": str(path), "meta": meta, "cells": cells,
            "point": point_iecc(cells, frac, arm_sign=arm_sign, e_star_level=e_star_level),
            "boot": bootstrap_iecc(cells, frac, n_boot, rng,
                                   arm_sign=arm_sign, e_star_level=e_star_level)}


def _fmt_ci(boot):
    if boot is None:
        return "(no per-item arrays -> point only)"
    if boot.get("reached_frac", 0) == 0:
        return "E* never reached in bootstrap"
    s = f"95% CI [{boot['lo']:+.4f}, {boot['hi']:+.4f}]"
    if boot["reached_frac"] < 0.99:
        s += f"  (E* reached in {boot['reached_frac']*100:.0f}% of draws)"
    return s


PRIMARY_PROBE = "gsm8k"   # prereg deviation 2026-07-10: generation-consistent primary


def report(paths, frac, n_boot, legacy_arm=False, expected_probe=PRIMARY_PROBE,
           combine_runs=False):
    """Group files by (model, target) and analyze each group SEPARATELY.

    2026-07-07 guard: E*/arm are per-MODEL quantities (prereg sec.7) and the H1
    contrast is within-(model, target); a mixed glob previously took the FIRST
    fp16 file as the reference for every row, silently crossing models/targets.
    frac=None -> the prereg sec.7 acceptance ladder is auto-evaluated on each
    group's fp16 cell; a numeric frac is an explicit off-prereg override.

    2026-07-12 pooling guard: every file's meta.capability_probe must equal
    `expected_probe` (default = the prereg primary, gsm8k). Files with a
    different or MISSING stamp (pre-2026-07-11 files were never stamped; the
    07-07 dose-blind single-pass-MMLU adjudicator is the motivating case)
    ABORT the report with a list of offenders -- the belt-and-suspenders twin
    of the harness's always-stamp discipline and run_matrix's is_done guard.
    Diagnostics: expected_probe='single_pass_mmlu' analyzes legacy-probe files;
    'any' disables the guard (label such output diagnostic, never
    confirmatory)."""
    metas = [(p, *load_result(p)) for p in paths]
    if expected_probe and expected_probe != "any":
        bad = [(p, m.get("capability_probe")) for (p, m, _c) in metas
               if m.get("capability_probe") != expected_probe]
        if bad:
            lines = [f"  {p}  capability_probe=" +
                     ("MISSING (unstamped pre-2026-07-11 file)" if s is None else repr(s))
                     for (p, s) in bad]
            raise SystemExit(
                f"POOLING GUARD: {len(bad)} of {len(metas)} file(s) do not carry "
                f"meta.capability_probe == {expected_probe!r} (the prereg primary):\n"
                + "\n".join(lines)
                + "\nMixed/legacy-probe files must never pool into confirmatory numbers "
                  "(2026-07-07 dose-blind MMLU lesson). Re-run with --expect-probe "
                  "<stamp> or --expect-probe any ONLY for labeled diagnostics.")
    groups = {}
    for t in metas:
        key = (t[1].get("model"), t[1].get("target"))
        groups.setdefault(key, []).append(t)
    if combine_runs:
        # 2026-07-14 Option C: run-level combine + sec.2A cross-model pool.
        return _combine_report(groups, n_boot)
    if len(groups) > 1:
        print(f"NOTE: {len(groups)} (model, target) groups in this file set -- each is "
              "analyzed separately (per-model E*/arm; contrasts never cross groups).")
    rows_all = []
    for key in sorted(groups, key=str):
        if len(groups) > 1:
            print(f"\n{'#' * 78}\nGROUP  model={key[0]}  target={key[1]}\n{'#' * 78}")
        rows_all.extend(_report_group(groups[key], frac, n_boot, legacy_arm))
    return rows_all


def _report_group(metas, frac, n_boot, legacy_arm=False):
    # S11: fix the arm sign + E* level ONCE from the fp16 POINT estimate (prereg
    # sec.7), then hold both fixed across every scheme + every bootstrap draw.
    # --legacy-arm restores the old free-arm rule (re-pick per cell/per draw).
    fp = next(((p, m, c) for (p, m, c) in metas if m.get("scheme") == "fp16"), None)
    # Prereg sec.7 E* acceptance ladder (auto) unless an explicit frac override.
    if frac is None:
        if fp is not None:
            acc = accept_e_star_frac(fp[2])
            frac = acc["frac"]
            trail = ", ".join(
                f"{t['frac']}->" + ("unreached" if not t["reached"]
                                    else f"fp16 IECC {t['fp16_iecc']:+.4f}")
                for t in acc["trail"])
            verdict = ("ACCEPTED" if acc["accepted"] else
                       f"NOT ACCEPTED -- even 0.3 costs >= {DELTA_MIN} on fp16; "
                       "IECC here is NOT confirmatory")
            e_star_note = (f"E* acceptance (prereg sec.7, auto): frac={frac} "
                           f"{verdict}   [trail: {trail}]")
        else:
            frac = 0.5
            e_star_note = ("E* acceptance: no fp16 cell in group -> default frac=0.5 "
                           "(acceptance rule NOT evaluated)")
    else:
        e_star_note = (f"E* frac={frac} set MANUALLY (off-prereg override; the sec.7 "
                       "acceptance ladder was not applied)")
    ref_sign = ref_level = None
    if not legacy_arm and fp is not None:
        rp = point_iecc(fp[2], frac)
        if rp.get("reached"):
            ref_sign, ref_level = rp["sign"], rp["E_star"]
    rows = []
    for (p, m, c) in metas:
        rng = np.random.default_rng(12345)
        rows.append({"path": str(p), "meta": m, "cells": c,
                     "point": point_iecc(c, frac, arm_sign=ref_sign, e_star_level=ref_level),
                     "boot": bootstrap_iecc(c, frac, n_boot, rng,
                                            arm_sign=ref_sign, e_star_level=ref_level)})
    print("=" * 78)
    print(f"SteerQuant IECC report   frac(E*)={frac}   n_boot={n_boot}")
    print(e_star_note)
    if ref_sign is not None:
        print(f"S11 reference FIXED from fp16 point: arm {'+' if ref_sign > 0 else '-'}, "
              f"E*={ref_level:+.4f} (held across all schemes + draws)")
    elif legacy_arm:
        print("S11: LEGACY free-arm mode (arm + E* re-picked per cell and per draw)")
    else:
        print("S11: no fp16 cell reached E* -> per-cell free-arm fallback")
    print("=" * 78)
    fp16_astar = None
    for r in rows:
        m, pt, bs = r["meta"], r["point"], r["boot"]
        print(f"\n{m.get('model','?')}  [{m.get('scheme','?')}]")
        if not pt.get("reached"):
            print(f"  E*={pt['E_star']:+.4f}  NOT REACHED (max shift {pt['max_shift']:+.4f})")
            continue
        if pt.get("weak_signal"):
            print(f"  WEAK SIGNAL: max shift {pt['max_shift']:+.4f} -- IECC not meaningful here.")
        print(f"  E*={pt['E_star']:+.4f} (baseline {pt['baseline_eff']:+.4f}, arm {'+' if pt['sign']>0 else '-'})  alpha*={pt['alpha_star']:+.2f}")
        print(f"  capability {pt['cap_baseline']:.4f} -> {pt['cap_at_star']:.4f}   IECC = {pt['iecc']:+.4f}   {_fmt_ci(bs)}")
        if m.get("scheme") == "fp16":
            fp16_astar = pt["abs_alpha_star"]
    if fp16_astar:
        # S13: alpha is applied to a UNIT vector, but the residual-stream scale can
        # differ per scheme under quantization; the raw alpha* ratio then conflates a
        # residual-scale change with a genuine steering-sensitivity change. Report a
        # normalized ratio alongside: (alpha*/residual_norm) vs the fp16 same, where
        # meta carries residual_norm_at_layer (S4). Legacy files lacking it print n/a.
        fp16_rn = next((r["meta"].get("residual_norm_at_layer") for r in rows
                        if r["meta"].get("scheme") == "fp16"), None)
        fp16_astar_norm = (fp16_astar / fp16_rn) if fp16_rn else None
        print("\n" + "-" * 78)
        print("Coefficient inflation   raw(alpha*/alpha*_FP16)   |   norm(alpha*/resid_norm ratio) [S13]")
        for r in rows:
            pt = r["point"]
            if not pt.get("reached"):
                continue
            raw = pt["abs_alpha_star"] / fp16_astar
            rn = r["meta"].get("residual_norm_at_layer")
            if rn and fp16_astar_norm:
                norm_s = f"{(pt['abs_alpha_star']/rn)/fp16_astar_norm:5.2f}x"
            else:
                norm_s = "  n/a"
            print(f"  {r['meta'].get('scheme','?'):<16} raw {raw:5.2f}x    norm {norm_s}")
    # H1 contrast: IECC(weight-only) - IECC(fp16) with a 90% bootstrap CI. This is
    # the equivalence quantity (prereg sec.7); a CI inside +/-delta supports H1.
    fp16_row = next((r for r in rows if r["meta"].get("scheme") == "fp16"), None)
    others = [r for r in rows if r["meta"].get("scheme") != "fp16"] if fp16_row else []
    if fp16_row is not None and others:
        print("\n" + "-" * 78)
        print("H1 contrast  IECC(scheme) - IECC(fp16)   [90% CI; <0 = cheaper than fp16]")
        crng = np.random.default_rng(20260626)
        for r in others:
            bc = bootstrap_contrast(fp16_row["cells"], r["cells"], frac, n_boot, crng,
                                    arm_sign=ref_sign, e_star_level=ref_level,
                                    legacy_arm=legacy_arm)
            scheme = r["meta"].get("scheme", "?")
            if bc is None:
                print(f"  {scheme:<16} (E* not jointly reached / no per-item arrays)")
            else:
                print(f"  {scheme:<16} {bc['contrast_mean']:+.4f}   "
                      f"90% CI [{bc['lo']:+.4f}, {bc['hi']:+.4f}]   (n={bc['n_used']})")
    print("=" * 78)
    return rows


def _synth_cells(rng, n_items=200, n_prompts=20, work=True, collapse=True, noise=0.08,
                 cap_mid=50.0):
    alphas = np.array([-80, -60, -40, -20, 0, 20, 40, 60, 80], float)
    cells = []
    for a in alphas:
        eff = float(np.tanh(a / 40.0)) if work else 0.0
        cap = 0.9 - 0.6 / (1.0 + np.exp(-(abs(a) - cap_mid) / 8.0)) if collapse else 0.9
        cap = float(np.clip(cap, 0.0, 1.0))
        cap_items = (rng.random(n_items) < cap).astype(int).tolist()
        eff_items = np.clip(eff + rng.normal(0, noise, n_prompts), -1, 1).tolist()
        cells.append({"alpha": a, "efficacy": float(np.mean(eff_items)),
                      "efficacy_per_prompt": [round(x, 4) for x in eff_items],
                      "capability_mmlu": float(np.mean(cap_items)),
                      "capability_mmlu_items": cap_items})
    return cells


def selftest():
    import tempfile
    ok = True
    rng = np.random.default_rng(0)
    cells = _synth_cells(rng)
    pt = point_iecc(cells, 0.5)
    bs = bootstrap_iecc(cells, 0.5, 1000, np.random.default_rng(1))
    print(f"[1] weak={pt['weak_signal']} alpha*={pt.get('alpha_star'):+.2f} IECC={pt.get('iecc'):+.4f} CI=[{bs['lo']:+.4f},{bs['hi']:+.4f}] reached={bs['reached_frac']:.2f}")
    ok &= pt["reached"] and (not pt["weak_signal"]) and pt["iecc"] > 0
    ok &= bs["reached_frac"] > 0.95 and bs["lo"] <= pt["iecc"] <= bs["hi"]
    ok &= 15 < pt["abs_alpha_star"] < 30
    ptf = point_iecc(_synth_cells(rng, work=False, noise=0.0), 0.5)
    print(f"[2] no-steering weak_signal={ptf['weak_signal']} (expect True)")
    ok &= (ptf["weak_signal"] is True)
    xc = _first_crossing(np.array([0, 20, 40, 60, 80.]), np.array([0.0, 0.3, 0.6, 0.4, 0.2]), 0.5)
    print(f"[3] first crossing of 0.5 = {xc:.2f} (expect ~33.3)")
    ok &= (abs(xc - 33.333) < 0.5)
    legacy = {"meta": {"model": "demo", "scheme": "fp16"},
              "by_alpha": [{"alpha": a, "efficacy_sentiment": float(np.tanh(a / 40)),
                            "capability_mmlu": float(np.clip(0.9 - 0.6 / (1 + np.exp(-(abs(a) - 50) / 8)), 0, 1))}
                           for a in (-80, -40, 0, 40, 80)]}
    with tempfile.TemporaryDirectory() as d:
        pth = Path(d) / "legacy_COMPLETE.json"
        pth.write_text(json.dumps(legacy), encoding="utf-8")
        r = analyze_file(str(pth), 0.5, 200)
    print(f"[4] legacy: reached={r['point']['reached']} boot={r['boot']} (None expected)")
    ok &= r["point"]["reached"] and (r["boot"] is None)
    se200, se800 = contrast_se(200, 0.74), contrast_se(800, 0.74)
    pw1 = tost_power(0.03, se200)
    pw_pooled = tost_power(0.03, pooled_se(se200, 0.01, 16))
    bt = breakeven_tau(0.03, se200, 16)
    print(f"[5] contrastSE 200->{se200:.4f} 800->{se800:.4f} 1cell_pow={pw1:.2f} pooled_pow={pw_pooled:.2f} breakeven_tau={bt}")
    ok &= se800 < se200 and pw_pooled > pw1 and ((bt is None) or (bt > 0))
    rng6 = np.random.default_rng(7)
    bc = bootstrap_contrast(_synth_cells(rng6), _synth_cells(rng6), 0.5, 300, np.random.default_rng(8))
    print(f"[6] contrast CI: [{bc['lo']:+.4f}, {bc['hi']:+.4f}] (n={bc['n_used']})")
    ok &= bc is not None and bc["lo"] <= bc["hi"]
    # [7] failure-rate endpoint + contrast
    rfr = np.random.default_rng(11)
    hi_fail = [1] * 30 + [0] * 70   # 0.30
    lo_fail = [1] * 5 + [0] * 95    # 0.05
    fr = bootstrap_failure_rate(hi_fail, 2000, rfr)
    fc = failure_rate_contrast(hi_fail, lo_fail, 3000, rfr)
    comp = failure_rates_comparable(lo_fail, lo_fail[:], 3000, rfr)
    print(f"[7] failure rate={fr['rate']:.2f} CI=[{fr['lo']:.2f},{fr['hi']:.2f}]  "
          f"contrast={fc['diff']:+.2f} excl0={fc['excludes_0']} comparable(self)={comp}")
    ok &= abs(fr["rate"] - 0.30) < 1e-9 and fc["excludes_0"] and comp is True

    # [8] length efficacy excludes failures + iso-effect DiD
    med = cell_length_efficacy([100, 120, 110, 9999, 130], [0, 0, 0, 1, 0])
    rln = np.random.default_rng(12)

    def _mk(alpha, slope):
        n = 50
        return {"alpha": alpha,
                "token_counts": (100 + slope * alpha + rln.normal(0, 2, n)).tolist(),
                "failure_flags": [0] * n}
    did = length_diff_in_differences([_mk(a, 1.0) for a in (-20, 0, 20)],
                                     [_mk(a, 0.5) for a in (-20, 0, 20)])
    j = did["alphas"].index(20)
    print(f"[8] survivor median={med} (9999 excluded)  DiD@+20={did['did'][j]:+.1f} (expect >5)")
    ok &= (med == 115.0) and (did["did"][j] > 5)

    # [9] K=16 pooling + heterogeneity gate branches
    rh = np.random.default_rng(13)
    homo = [{"model": f"m{m}", "target": f"t{t}", "est": 0.005 + rh.normal(0, 0.002),
             "se": 0.02} for m in range(4) for t in range(4)]
    hetm = [{"model": f"m{m}", "target": f"t{t}", "est": 0.12 * m + rh.normal(0, 0.002),
             "se": 0.02} for m in range(4) for t in range(4)]
    hett = [{"model": f"m{m}", "target": f"t{t}", "est": 0.12 * t + rh.normal(0, 0.002),
             "se": 0.02} for m in range(4) for t in range(4)]
    hp, hpm, hpt = heterogeneity_pool(homo), heterogeneity_pool(hetm), heterogeneity_pool(hett)
    print(f"[9] homo->{hp['branch']} (I2={hp['I2']:.2f}); model-spread->{hpm['branch']} "
          f"(tauM={hpm['tau_model']:.3f}); target-spread->{hpt['branch']} (tauT={hpt['tau_target']:.3f})")
    ok &= hp["branch"] == "pooled_K16" and hp["confirmatory"]
    ok &= hpm["branch"] in ("per_model", "per_cell") and hpm["tau_model"] > TAU_THRESHOLD
    ok &= hpt["branch"] in ("per_target", "per_cell") and hpt["tau_target"] > TAU_THRESHOLD

    # [10] S11: fp16 and scheme that would pick OPPOSITE arms under the free rule
    # must, once fixed, both use the fp16 arm + the fp16 E* level.
    def _asym_cells(pos_stronger, rng_, n_items=200, n_prompts=20):
        alphas = np.array([-80, -40, 0, 40, 80], float)
        cells_ = []
        for a in alphas:
            t = float(np.tanh(a / 40.0))
            strong, weak = (0.9, 0.15)
            if a >= 0:
                eff = (strong if pos_stronger else weak) * t
            else:
                eff = (weak if pos_stronger else strong) * t
            cap = float(np.clip(0.9 - 0.6 / (1.0 + np.exp(-(abs(a) - 50) / 8.0)), 0, 1))
            cap_items = (rng_.random(n_items) < cap).astype(int).tolist()
            eff_items = np.clip(eff + rng_.normal(0, 0.02, n_prompts), -1, 1).tolist()
            cells_.append({"alpha": a, "efficacy": float(np.mean(eff_items)),
                           "efficacy_per_prompt": [round(x, 4) for x in eff_items],
                           "capability_mmlu": float(np.mean(cap_items)),
                           "capability_mmlu_items": cap_items})
        return cells_
    ra = np.random.default_rng(21)
    fp16c = _asym_cells(True, ra)      # strongest shift on the + arm
    schc = _asym_cells(False, ra)      # strongest shift on the - arm
    free_fp = point_iecc(fp16c, 0.5)
    free_sc = point_iecc(schc, 0.5)
    fixed_sc = point_iecc(schc, 0.5, arm_sign=free_fp["sign"], e_star_level=free_fp["E_star"])
    print(f"[10] free arms: fp16={'+' if free_fp['sign']>0 else '-'} scheme={'+' if free_sc['sign']>0 else '-'} "
          f"(differ={free_fp['sign']!=free_sc['sign']}); fixed scheme arm={'+' if fixed_sc['sign']>0 else '-'} "
          f"E*match={abs(fixed_sc['E_star']-free_fp['E_star'])<1e-12}")
    ok &= (free_fp["sign"] == +1 and free_sc["sign"] == -1)          # opposite under old rule
    ok &= (fixed_sc["sign"] == free_fp["sign"])                      # fix forces fp16 arm
    ok &= (abs(fixed_sc["E_star"] - free_fp["E_star"]) < 1e-12)      # fp16 E* level held

    # [11] S12 paired bootstrap + S13 normalized column: report() runs with the S4
    # residual_norm meta (norm ratio numeric) and without it (prints n/a), no crash.
    import io as _io, contextlib as _ctx, tempfile as _tf
    rpt = np.random.default_rng(31)
    fp_cells, sc_cells = _synth_cells(rpt), _synth_cells(rpt)

    def _wf(d, name, cells, meta):
        p = Path(d) / name
        p.write_text(json.dumps({"meta": meta, "by_alpha": cells}), encoding="utf-8")
        return str(p)
    with _tf.TemporaryDirectory() as d:
        f1 = _wf(d, "fp16_COMPLETE.json", fp_cells,
                 {"model": "m", "scheme": "fp16", "residual_norm_at_layer": 300.0,
                  "capability_probe": "gsm8k"})
        f2 = _wf(d, "nf4_COMPLETE.json", sc_cells,
                 {"model": "m", "scheme": "w4a16_bnb_nf4", "residual_norm_at_layer": 260.0,
                  "capability_probe": "gsm8k"})
        f3 = _wf(d, "legacy_COMPLETE.json", fp_cells,
                 {"model": "m2", "scheme": "fp16",
                  "capability_probe": "gsm8k"})   # no residual_norm key
        buf = _io.StringIO()
        with _ctx.redirect_stdout(buf):
            rows_k = report([f1, f2], 0.5, 200)   # both carry S4 meta -> numeric norm col
            rows_l = report([f3, f2], 0.5, 200)   # legacy fp16 -> norm col n/a, must not crash
        out = buf.getvalue()
    print(f"[11] report S12/S13: rows_k={len(rows_k)} rows_l={len(rows_l)} "
          f"S13_col={'[S13]' in out} n/a_seen={'n/a' in out}")
    ok &= (rows_k is not None and rows_l is not None
           and "[S13]" in out and "n/a" in out)

    # [12] LENGTH bootstrap fix: survivor-median statistic, prompt-level pairing,
    # censoring propagated. efficacy_per_prompt is deliberately written as
    # SURVIVORS ONLY (exactly as the harness does) to prove the length path
    # never touches it.
    rlb = np.random.default_rng(41)

    def _len_cell(alpha, fail_top=False):
        n = 40
        t = 100 + 2.0 * float(alpha) + rlb.normal(0, 3, n)
        fl = np.zeros(n, dtype=int)
        if fail_top:
            fl[np.argsort(t)[-10:]] = 1   # censor the LONGEST quarter
        cap = (rlb.random(200) < 0.85).astype(int).tolist()
        surv = t[fl == 0]
        return {"alpha": float(alpha),
                "efficacy": float(np.median(surv)),
                "efficacy_per_prompt": [round(float(x), 2) for x in surv],
                "length_tokens_per_prompt": [round(float(x), 2) for x in t],
                "failure_flags": fl.tolist(),
                "capability_mmlu": float(np.mean(cap)),
                "capability_mmlu_items": cap}

    lcells = [_len_cell(a, fail_top=(a >= 60)) for a in (0, 10, 20, 30, 40, 60, 80)]
    pt_len = point_iecc(lcells, 0.5)
    bs_len = bootstrap_iecc(lcells, 0.5, 500, np.random.default_rng(42))
    flat_cells = []
    for a in (0, 10, 20):
        c0 = _len_cell(0)
        c0["alpha"] = float(a)
        flat_cells.append(c0)
    pt_flat = point_iecc(flat_cells, 0.5)
    print(f"[12] length: stat={bs_len.get('eff_stat')} reached={bs_len.get('reached_frac', 0):.2f} "
          f"CI=[{bs_len.get('lo', float('nan')):+.4f},{bs_len.get('hi', float('nan')):+.4f}] "
          f"point={pt_len.get('iecc'):+.4f} weak={pt_len['weak_signal']} flat_weak={pt_flat['weak_signal']}")
    ok &= bs_len.get("eff_stat") == "survivor_median"
    ok &= bs_len.get("reached_frac", 0) > 0.9
    ok &= bs_len["lo"] <= pt_len["iecc"] <= bs_len["hi"]
    ok &= pt_len["weak_signal"] is False
    ok &= pt_flat["weak_signal"] is True   # rel guard fires on the token scale

    # [13] Prereg sec.7 E* acceptance ladder: 0.5 costs >= 10% on fp16 -> step to
    # 0.4 (passes); a cliff-at-origin curve rejects all three fractions.
    ecells = [{"alpha": float(a), "efficacy": float(np.tanh(a / 40.0)),
               "capability_mmlu": float(0.9 - 0.30 / (1 + np.exp(-(a - 19) / 1.5)))}
              for a in (0, 5, 10, 15, 20, 25, 30, 40, 60, 80)]
    acc = accept_e_star_frac(ecells)
    hard = [{"alpha": float(a), "efficacy": float(np.tanh(a / 40.0)),
             "capability_mmlu": float(0.9 - 0.5 / (1 + np.exp(-(a - 8) / 1.0)))}
            for a in (0, 5, 10, 15, 20, 25, 30, 40, 60, 80)]
    acc2 = accept_e_star_frac(hard)
    tr = [(t["frac"], None if t["fp16_iecc"] is None else round(t["fp16_iecc"], 3))
          for t in acc["trail"]]
    print(f"[13] E* ladder: frac={acc['frac']} accepted={acc['accepted']} trail={tr}; "
          f"cliff-curve accepted={acc2['accepted']} (expect False)")
    ok &= acc["accepted"] and acc["frac"] == 0.4 and len(acc["trail"]) == 2
    ok &= acc["trail"][0]["fp16_iecc"] >= DELTA_MIN
    ok &= (acc2["accepted"] is False) and len(acc2["trail"]) == 3

    # [14] H1 three-label + H2 Rule A, incl. wiring into heterogeneity_pool.
    l_eq = h1_three_label(-0.010, 0.020)
    l_mw = h1_three_label(0.035, 0.090)
    l_in = h1_three_label(-0.020, 0.050)
    l_ob = h1_three_label(-0.090, -0.050)
    print(f"[14] H1 labels: {l_eq} / {l_mw} / {l_in} / {l_ob};  "
          f"H2 ruleA lo95=0.12->{h2_rule_a(0.12)} lo95=0.08->{h2_rule_a(0.08)};  "
          f"pool h1_label={hp['h1_label']}")
    ok &= (l_eq == "Equivalent" and l_mw == "Meaningfully Worse"
           and l_in == "Inconclusive" and l_ob.startswith("OutsideMargin"))
    ok &= h2_rule_a(0.12) is True and h2_rule_a(0.08) is False
    ok &= hp["h1_label"] == "Equivalent" and "ci95_lo" in hp and hp["k"] == 16

    # [15] report() group guard: a mixed-model file set analyzes each
    # (model, target) group separately; fp16 reference + H1 contrast never
    # cross groups; frac=None triggers the sec.7 auto ladder per group.
    rg = np.random.default_rng(51)
    ga_fp, ga_nf4, gb_fp = _synth_cells(rg), _synth_cells(rg), _synth_cells(rg)
    with _tf.TemporaryDirectory() as d:
        pa1 = _wf(d, "A_fp16.json", ga_fp,
                  {"model": "modelA", "target": "sentiment", "scheme": "fp16",
                   "capability_probe": "gsm8k"})
        pa2 = _wf(d, "A_nf4.json", ga_nf4,
                  {"model": "modelA", "target": "sentiment", "scheme": "w4a16_bnb_nf4",
                   "capability_probe": "gsm8k"})
        pb1 = _wf(d, "B_fp16.json", gb_fp,
                  {"model": "modelB", "target": "sentiment", "scheme": "fp16",
                   "capability_probe": "gsm8k"})
        buf = _io.StringIO()
        with _ctx.redirect_stdout(buf):
            rows_g = report([pa1, pa2, pb1], None, 200)
        out_g = buf.getvalue()
    n_groups = out_g.count("GROUP  model=")
    print(f"[15] multi-group: rows={len(rows_g)} groups={n_groups} "
          f"auto_e_star={'E* acceptance (prereg sec.7' in out_g} "
          f"contrast_blocks={out_g.count('H1 contrast')}")
    ok &= len(rows_g) == 3 and n_groups == 2
    ok &= "E* acceptance (prereg sec.7" in out_g
    ok &= out_g.count("H1 contrast") == 1   # only group A has an fp16+scheme pair

    # [16] 2026-07-11 adaptive-capability files: capability present ONLY at
    # baseline + a window around the crossing. point/bootstrap/contrast must
    # all still work, and the point IECC must EQUAL the full-grid value (same
    # curve; interpolation knots around alpha* are inside the kept window).
    r16 = np.random.default_rng(41)
    full16 = _synth_cells(r16)
    pt_full16 = point_iecc(full16, 0.5)
    # The synth curve is arm-symmetric, so per-alpha noise decides the free
    # arm -- keep the window on WHICHEVER arm the point estimate picked
    # (baseline + the grid points around alpha*~21 on that arm).
    sgn16 = pt_full16["sign"]
    keep16 = {0.0, sgn16 * 20.0, sgn16 * 40.0, sgn16 * 60.0}
    part16 = []
    for c in full16:
        c = dict(c)
        if c["alpha"] not in keep16:
            c["capability_mmlu"] = None
            c["capability_mmlu_items"] = None
        part16.append(c)
    pt_part16 = point_iecc(part16, 0.5)
    bs_part16 = bootstrap_iecc(part16, 0.5, 400, np.random.default_rng(42))
    bc_part16 = bootstrap_contrast(full16, part16, 0.5, 300, np.random.default_rng(43))
    same_pt = (pt_part16.get("reached") and pt_full16.get("reached")
               and abs(pt_part16["iecc"] - pt_full16["iecc"]) < 1e-9)
    print(f"[16] adaptive files: point match={same_pt} "
          f"(full {pt_full16.get('iecc'):+.4f} vs part {pt_part16.get('iecc'):+.4f}, "
          f"n_cap_alphas={pt_part16.get('n_cap_alphas')}) "
          f"boot_reached={None if bs_part16 is None else bs_part16.get('reached_frac')} "
          f"contrast={'ok' if bc_part16 is not None else 'None'}")
    ok &= bool(same_pt) and pt_part16.get("n_cap_alphas") == 4
    ok &= bs_part16 is not None and bs_part16.get("reached_frac", 0) > 0.9
    ok &= bc_part16 is not None and bc_part16["lo"] <= bc_part16["hi"]

    # [17] 2026-07-12 pooling guard: report() refuses files whose
    # meta.capability_probe is missing or != the expected primary; an explicit
    # expected_probe override ('single_pass_mmlu' / 'any') admits diagnostics.
    r17 = np.random.default_rng(61)
    c17 = _synth_cells(r17)
    with _tf.TemporaryDirectory() as d:
        g1 = _wf(d, "g_fp16.json", c17,
                 {"model": "mg", "target": "sentiment", "scheme": "fp16",
                  "capability_probe": "gsm8k"})
        m1 = _wf(d, "m_fp16.json", c17,
                 {"model": "mg", "target": "sentiment", "scheme": "fp16",
                  "capability_probe": "single_pass_mmlu"})
        u1 = _wf(d, "u_fp16.json", c17,
                 {"model": "mg", "target": "sentiment", "scheme": "fp16"})  # unstamped
        guard_mixed = guard_unstamped = False
        buf = _io.StringIO()
        with _ctx.redirect_stdout(buf):
            try:
                report([g1, m1], 0.5, 50)
            except SystemExit as e:
                guard_mixed = "POOLING GUARD" in str(e)
            try:
                report([u1], 0.5, 50)
            except SystemExit as e:
                guard_unstamped = "POOLING GUARD" in str(e)
            rows_ok = report([g1], 0.5, 50)                        # clean gsm8k set passes
            rows_diag = report([m1], 0.5, 50,
                               expected_probe="single_pass_mmlu")  # explicit diagnostic
            rows_any = report([u1], 0.5, 50, expected_probe="any")
    print(f"[17] pooling guard: mixed_blocked={guard_mixed} "
          f"unstamped_blocked={guard_unstamped} clean_rows={len(rows_ok)} "
          f"diag_rows={len(rows_diag)} any_rows={len(rows_any)}")
    ok &= guard_mixed and guard_unstamped
    ok &= len(rows_ok) == 1 and len(rows_diag) == 1 and len(rows_any) == 1

    # [18] Option C run-level combiner (2026-07-14): REML tau2 + modified
    # Knapp-Hartung. (a) balanced-case REML identity tau2 == max(0, S^2-vbar);
    # (b) zero-heterogeneity -> q<1 -> modified KH falls back to the
    # conventional SE (never narrower); (c) t-table df=4 values; (d) end-to-end
    # report(combine_runs=True): 5 runs x (fp16 + equivalent nf4 + int8 with a
    # run-varying capability penalty) -> k=5 both schemes, int8 costlier with
    # tau2(int8) >= tau2(nf4); single model -> no cross-model pool printed.
    y18 = np.array([0.010, -0.020, 0.035, 0.005, -0.015])
    v18 = np.full(5, 0.0004)
    t2_18, conv18 = reml_tau2(y18, v18)
    closed18 = max(0.0, float(np.var(y18, ddof=1) - v18.mean()))
    hk0 = hk_ci([0.01] * 5, [0.0004] * 5)
    conv_se0 = float(np.sqrt((0.0004 + hk0["tau2"]) / 5))
    mids18 = [10.0, 18.0, 26.0, 14.0, 22.0]
    paths18 = []
    with _tf.TemporaryDirectory() as d:
        for rr in range(1, 6):
            for si, (sch, cm) in enumerate((("fp16", 50.0),
                                            ("w4a16_bnb_nf4", 50.0),
                                            ("w8a16_bnb_int8", 50.0 - mids18[rr - 1]))):
                cells18 = _synth_cells(np.random.default_rng([90 + rr, si]),
                                       cap_mid=cm)
                paths18.append(_wf(d, f"m18_{sch}_r{rr}.json", cells18,
                                   {"model": "m18", "target": "sentiment",
                                    "scheme": sch, "capability_probe": "gsm8k",
                                    "resample": {"run": rr}}))
        buf = _io.StringIO()
        with _ctx.redirect_stdout(buf):
            rows18 = report(paths18, None, 200, combine_runs=True)
        out18 = buf.getvalue()
    hk_by18 = {r["scheme"]: r["hk"] for r in rows18}
    nf4_18, int8_18 = hk_by18["w4a16_bnb_nf4"], hk_by18["w8a16_bnb_int8"]
    print(f"[18] REML balanced {t2_18:.6f} == {closed18:.6f} conv={conv18}; "
          f"mKH0 se={hk0['se']:.4f} (conv {conv_se0:.4f}) mod={hk0['hk_modified_applied']}; "
          f"combine k=({nf4_18['k']},{int8_18['k']}) nf4={nf4_18['mu']:+.4f} "
          f"int8={int8_18['mu']:+.4f} tau2=({nf4_18['tau2']:.5f},{int8_18['tau2']:.5f}) "
          f"labels=({nf4_18['h1_label']} / {int8_18['h1_label']})")
    ok &= conv18 and abs(t2_18 - closed18) < 1e-10
    ok &= hk0["hk_modified_applied"] and abs(hk0["se"] - conv_se0) < 1e-12
    ok &= abs(T_TABLE[4][0] - 2.131847) < 1e-9 and abs(T_TABLE[4][1] - 2.776445) < 1e-9
    ok &= nf4_18["k"] == 5 and int8_18["k"] == 5
    ok &= int8_18["mu"] > nf4_18["mu"] and int8_18["tau2"] >= nf4_18["tau2"]
    ok &= "Option C combine" in out18 and "CROSS-MODEL POOL" not in out18

    print("\nSELFTEST:", "PASS" if ok else "FAIL")
    return ok


def main():
    ap = argparse.ArgumentParser(description="SteerQuant IECC + H1 power analysis.")
    ap.add_argument("--results", nargs="*", default=[], help="result JSON paths or globs")
    ap.add_argument("--e-star-frac", type=float, default=None,
                    help="E* fraction of max shift. DEFAULT None = AUTO per the "
                         "prereg sec.7 acceptance ladder (0.5->0.4->0.3, evaluated "
                         "on each group's fp16 cell). Passing a number is an "
                         "off-prereg override and is printed as such.")
    ap.add_argument("--n-boot", type=int, default=2000, help="bootstrap resamples")
    ap.add_argument("--selftest", action="store_true", help="run unit self-test and exit")
    ap.add_argument("--power", action="store_true", help="print the H1 equivalence power/tau table")
    ap.add_argument("--delta", type=float, default=0.03, help="equivalence margin (MMLU acc)")
    ap.add_argument("--p", type=float, default=0.74, help="assumed baseline accuracy for power")
    ap.add_argument("--k", type=int, default=16, help="pooled cells (models x targets)")
    ap.add_argument("--legacy-arm", action="store_true",
                    help="S11: use the OLD free-arm rule (re-pick best sign arm + E* per "
                         "cell and per bootstrap draw). Default fixes arm + E* once from "
                         "the fp16 point estimate (prereg sec.7).")
    ap.add_argument("--expect-probe", default=PRIMARY_PROBE,
                    help="2026-07-12 pooling guard: every --results file must carry "
                         "meta.capability_probe == this value (default gsm8k = the "
                         "prereg primary). 'single_pass_mmlu' admits legacy-probe "
                         "diagnostics; 'any' disables the guard. Non-default output "
                         "is DIAGNOSTIC, never confirmatory.")
    ap.add_argument("--combine-runs", action="store_true",
                    help="Option C run-level combiner (Saurav 2026-07-12; method "
                         "note 2026-07-14): per (model, target) group, split by "
                         "resample run, per-run triad contrasts, REML tau2 + "
                         "modified Knapp-Hartung CI across runs; then the sec.2A "
                         "cross-model pool. The confirmatory H1/H2 path.")
    args = ap.parse_args()
    if args.selftest:
        raise SystemExit(0 if selftest() else 1)
    if args.power:
        power_report(p=args.p, delta=args.delta, k=args.k)
        return
    paths = []
    for pat in args.results:
        paths.extend(sorted(_glob.glob(pat)) or [pat])
    if not paths:
        raise SystemExit("no --results files given (or run --selftest / --power)")
    report(paths, args.e_star_frac, args.n_boot, legacy_arm=args.legacy_arm,
           expected_probe=args.expect_probe, combine_runs=args.combine_runs)


if __name__ == "__main__":
    main()
