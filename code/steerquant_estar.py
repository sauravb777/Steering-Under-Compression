#!/usr/bin/env python3
"""SteerQuant -- shared E*/arm/crossing rule (prereg sec.7). Pure numpy, torch-free.

2026-07-11: EXTRACTED from steerquant_analysis.py so the post-hoc analysis and the
harness's ONLINE adaptive capability-alpha selection import ONE encoding of the
rule (single source of truth; prevents the "prereg rule encoded twice" drift the
insights doc names as a recurring root cause). `_arm` and `_first_crossing` moved
VERBATIM -- no behavior change; steerquant_analysis.py re-imports them.

`accept_e_star_frac` (the ladder ACCEPTANCE step) stays in the analysis because
acceptance needs CAPABILITY data, which the harness does not have at selection
time. Instead, the online selection covers the crossings of ALL ladder fractions
(select_capability_alphas), so whichever fraction the ladder later accepts has
capability measured around its crossing.

Selftest (offline, no GPU):  python steerquant_estar.py
"""
from __future__ import annotations

import numpy as np

# --- Prereg sec.7 constants (moved from steerquant_analysis.py 2026-07-11) ---
E_STAR_LADDER = (0.5, 0.4, 0.3)   # acceptance ladder, first-accepted wins
DELTA_MIN = 0.10                  # meaningful-degradation line (H2 / E* acceptance)


def _arm(alphas, eff, cap, fixed_sign=None):
    """Select the steering arm (sign of alpha) and return its ordered curve.

    S11: when `fixed_sign` is given, USE that arm rather than re-picking the
    max-shift arm. Passing the fp16 arm into every scheme + every bootstrap draw
    stops the H1 contrast from silently comparing opposite arms (prereg sec.7).
    fixed_sign=None reproduces the legacy free-arm behavior (max |shift| wins).
    """
    alphas = np.asarray(alphas, float)
    b_i = int(np.argmin(np.abs(alphas)))
    base = float(eff[b_i])
    if fixed_sign is not None:
        sign = int(fixed_sign)
        sel = sorted((i for i in range(len(alphas)) if alphas[i] * sign >= 0),
                     key=lambda i: abs(alphas[i]))
    else:
        best = None
        for sign_ in (+1, -1):
            sel_ = sorted((i for i in range(len(alphas)) if alphas[i] * sign_ >= 0),
                          key=lambda i: abs(alphas[i]))
            e_ = np.array([eff[i] for i in sel_], float)
            shift = float(np.max(np.abs(e_ - base))) if len(e_) else 0.0
            if best is None or shift > best[0]:
                best = (shift, sign_, sel_)
        _, sign, sel = best
    absa = np.array([abs(alphas[i]) for i in sel], float)
    sa = np.array([alphas[i] for i in sel], float)
    e = np.array([eff[i] for i in sel], float)
    c = np.array([cap[i] for i in sel], float)
    return absa, sa, e, c, base, sign


def _first_crossing(x, y, level):
    d = np.asarray(y, float) - level
    if abs(d[0]) < 1e-12:
        return float(x[0])
    for i in range(len(x) - 1):
        if d[i] == 0.0:
            return float(x[i])
        if d[i] * d[i + 1] < 0:
            t = d[i] / (d[i] - d[i + 1])
            return float(x[i] + t * (x[i + 1] - x[i]))
    if abs(d[-1]) < 1e-12:
        return float(x[-1])
    return None


def e_star_levels_from_curve(alphas, eff, ladder=E_STAR_LADDER, fixed_sign=None):
    """Arm + reference efficacy LEVELS for every ladder fraction, from ONE
    efficacy curve. No capability data needed. For fp16 this is
    self-referential; for a quantized scheme, call it on the SIBLING fp16 curve
    and pass the result into select_capability_alphas -- level and arm stay
    fp16-derived (S11) while the crossing is found on the scheme's OWN curve.
    Same math as accept_e_star_frac/iecc_from_arrays use post-hoc:
    level = base + frac * (extreme - base)."""
    alphas = np.asarray(alphas, float)
    eff = np.asarray(eff, float)
    cap = np.full(len(alphas), np.nan)   # _arm slices it; values unused here
    absa, sa, e, c, base, sign = _arm(alphas, eff, cap, fixed_sign=fixed_sign)
    extreme = float(e[int(np.argmax(np.abs(e - base)))])
    levels = [float(base + f * (extreme - base)) for f in ladder]
    return {"arm_sign": int(sign), "baseline_eff": float(base),
            "extreme_eff": extreme, "ladder": [float(f) for f in ladder],
            "levels": levels}


def select_capability_alphas(alphas, eff, neighbors=2, ladder=E_STAR_LADDER,
                             arm_sign=None, e_star_levels=None):
    """ONLINE capability-alpha selection (adaptive design 2026-07-11).

    Given the FULL-grid efficacy curve of the running scheme, pick the small set
    of alphas that actually need the expensive capability probe: the baseline
    (alpha=0) plus, for EACH ladder fraction's E* level, the grid alpha nearest
    the efficacy crossing and `neighbors` grid points on each side (interpolation
    margin). Covering the whole ladder means whichever fraction
    accept_e_star_frac later accepts post-hoc has capability measured around its
    crossing -- acceptance itself needs capability data and is NOT decided here.

    arm_sign/e_star_levels: pass BOTH (derived from the sibling fp16 curve via
    e_star_levels_from_curve) for a non-fp16 scheme (S11 discipline); leave None
    for the fp16 self-referential path. The crossing is always found on THIS
    curve, so a dose-inflated quantized scheme gets its window recentered on its
    own crossing (the 2026-06-26 ~1.1-1.2x inflation finding).

    Fallback: if NO ladder level crosses this curve (e.g. a weak scheme never
    reaches the fp16-derived levels), the whole selected arm (+ baseline) is
    returned and `fallback` says why -- fail-open to full-arm coverage, never to
    a silently mis-centered window.

    Returns {"alphas": sorted signed grid alphas, "arm_sign", "levels",
    "neighbors", "crossings": [{frac, level, abs_alpha_star|None}], "fallback"}.
    """
    alphas_arr = np.asarray(alphas, float)
    eff_arr = np.asarray(eff, float)
    if arm_sign is None or e_star_levels is None:
        ref = e_star_levels_from_curve(alphas_arr, eff_arr, ladder=ladder)
        arm_sign, e_star_levels = ref["arm_sign"], ref["levels"]
    cap = np.full(len(alphas_arr), np.nan)
    absa, sa, e, c, base, sign = _arm(alphas_arr, eff_arr, cap, fixed_sign=arm_sign)
    baseline_alpha = float(alphas_arr[int(np.argmin(np.abs(alphas_arr)))])
    picked = {baseline_alpha}
    crossings, any_crossed = [], False
    for f, level in zip(ladder, e_star_levels):
        a_star = _first_crossing(absa, e, level)
        crossings.append({"frac": float(f), "level": float(level),
                          "abs_alpha_star": None if a_star is None else float(a_star)})
        if a_star is None:
            continue
        any_crossed = True
        j = int(np.argmin(np.abs(absa - a_star)))
        for k in range(max(0, j - neighbors), min(len(absa), j + neighbors + 1)):
            picked.add(float(sa[k]))
    fallback = None
    if not any_crossed:
        fallback = ("no ladder level crossed on this efficacy curve -> probing "
                    "the FULL selected arm (+ baseline) instead of an adaptive "
                    "window")
        for a in sa:
            picked.add(float(a))
    return {"alphas": sorted(picked), "arm_sign": int(sign),
            "levels": [float(x) for x in e_star_levels],
            "neighbors": int(neighbors), "crossings": crossings,
            "fallback": fallback}


# ── Selftest (torch-free; registered in run_offline_checks.py) ──────────────
def _selftest():
    ok = True
    grid = [-80.0, -60.0, -40.0, -35.0, -30.0, -25.0, -20.0, -15.0, -10.0, -5.0,
            0.0, 5.0, 10.0, 15.0, 20.0, 40.0, 60.0, 80.0]

    def curve(dose, neg_scale=1.0, pos_scale=0.2):
        return [float(3.0 * np.tanh(abs(a) / dose)
                      * (neg_scale if a < 0 else pos_scale)) for a in grid]

    # [1] fp16 self-referential path: arm, determinism, window contents/size.
    fp16_eff = curve(30.0)
    s1 = select_capability_alphas(grid, fp16_eff)
    s1b = select_capability_alphas(grid, fp16_eff)
    det = (s1 == s1b)
    arm_ok = (s1["arm_sign"] == -1 and all(a <= 0 for a in s1["alphas"]))
    # crossings for f=0.5/0.4/0.3 land at |a|~16.4/12.6/9.2 -> union {0..-25}
    want = {0.0, -5.0, -10.0, -15.0, -20.0, -25.0}
    win_ok = (set(s1["alphas"]) == want and s1["fallback"] is None)
    print(f"[1] self path: det={det} arm={s1['arm_sign']:+d} "
          f"alphas={s1['alphas']} win_ok={win_ok}")
    ok &= det and arm_ok and win_ok

    # [2] sibling path with coefficient inflation (~1.2x dose): fp16 levels +
    # arm held (S11), crossing recentered on the SCHEME's own curve.
    ref = e_star_levels_from_curve(grid, fp16_eff)
    infl_eff = curve(36.0)
    s2 = select_capability_alphas(grid, infl_eff, arm_sign=ref["arm_sign"],
                                  e_star_levels=ref["levels"])
    # f=0.5 crossing moves ~16.4 -> ~19.6: window must now reach -30.
    recentered = (-30.0 in s2["alphas"]) and (-30.0 not in s1["alphas"])
    print(f"[2] sibling path: alphas={s2['alphas']} recentered={recentered}")
    ok &= recentered and s2["arm_sign"] == -1 and s2["fallback"] is None

    # [3] crossing values match the closed form (tanh inverse), via crossings[].
    x_expect = 30.0 * float(np.arctanh(ref["levels"][0] / 3.0))
    x_got = s1["crossings"][0]["abs_alpha_star"]
    close = abs(x_got - x_expect) < 1.0   # grid-linear interp vs analytic
    print(f"[3] f=0.5 crossing: got {x_got:.2f} expect ~{x_expect:.2f} -> {close}")
    ok &= close

    # [4] fallback: a weak scheme that never reaches the fp16 levels must get
    # the FULL arm (+ baseline), never a silently mis-centered window.
    weak_eff = curve(30.0, neg_scale=0.20, pos_scale=0.04)  # extreme ~0.59 < level(0.3) ~0.89
    s4 = select_capability_alphas(grid, weak_eff, arm_sign=ref["arm_sign"],
                                  e_star_levels=ref["levels"])
    n_arm = sum(1 for a in grid if a <= 0)   # negative arm incl. 0
    fb_ok = (s4["fallback"] is not None and len(s4["alphas"]) == n_arm)
    print(f"[4] fallback: fired={s4['fallback'] is not None} "
          f"n={len(s4['alphas'])}/{n_arm}")
    ok &= fb_ok

    # [5] neighbors knob: +-1 window is strictly smaller than +-2.
    s5 = select_capability_alphas(grid, fp16_eff, neighbors=1)
    print(f"[5] neighbors=1: {len(s5['alphas'])} alphas < {len(s1['alphas'])}")
    ok &= len(s5["alphas"]) < len(s1["alphas"])

    print(f"[estar selftest] {'PASS' if ok else 'FAIL'}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    _selftest()
