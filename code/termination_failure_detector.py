"""
Termination-Failure Detector (SteerQuant, prereg §8)
====================================================
Deterministic, content-blind classifier for whether a single generated
reasoning trace is a TERMINATION FAILURE. Used by the judge-free
reasoning-length steering target.

A generation is a FAILURE if ANY of:
    (a) it does not emit EOS before `max_new_tokens` (ran to the cap /
        non-terminating);
    (b) STRUCTURAL LOOP -- a consecutive n-gram repetition at/above the
        length-graduated thresholds of degenerate_repeat_detector
        (n<=2: >=5, n=3-5: >=3, n>=6: >=2, plus a sub-word char check).
        We use ONLY its loop flag; we do NOT use its truncate-and-recover
        output. No loop collapse, no length recovery.
    (c) rep-4 (seq-rep-4: fraction of 4-grams that are non-unique,
        Welleck et al. 2019 / Holtzman et al. 2019) exceeds the
        operational cutoff (default 0.5; pre-registered OPERATIONAL value,
        NOT literature-derived -- report at {0.4, 0.5, 0.6} robustness).

This module reports DETECTION ONLY. There is no recovery, no cleaning, no
content inspection -- same input -> same output, so it does not reintroduce
a judge. A failure is a failure even if it contains genuine reasoning.

Failed traces are EXCLUDED from the length measurement and COUNTED in the
co-primary failure-rate endpoint.

Cost: CPU-only post-processing on already-generated traces; no model call.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

# --- Import the existing, tested structural-loop detector ----------------
# In the project tree it lives at D:\Claude\Tools\degenerate_repeat_detector.py.
# Try a few import paths so this works whether Tools is on sys.path, is a
# package, or sits beside this file.
try:
    from degenerate_repeat_detector import check_for_loops  # type: ignore
except ImportError:  # pragma: no cover - import-path shim
    try:
        from Tools.degenerate_repeat_detector import check_for_loops  # type: ignore
    except ImportError:
        import os
        import sys

        _here = os.path.dirname(os.path.abspath(__file__))
        for _cand in (
            _here,
            os.path.join(_here, "Tools"),
            os.path.join(_here, "..", "Tools"),
            os.path.join(_here, "..", "..", "Tools"),  # D:\Claude\Tools
        ):
            if os.path.isdir(_cand) and _cand not in sys.path:
                sys.path.insert(0, _cand)
        from degenerate_repeat_detector import check_for_loops  # type: ignore


# Default operational cutoff for rep-4 (criterion c). Pre-registered as an
# OPERATIONAL value; report the length analysis at {0.4, 0.5, 0.6}.
DEFAULT_REP4_CUTOFF = 0.5
DEFAULT_REP_N = 4


@dataclass
class FailureResult:
    """Outcome of the termination-failure check for one trace."""
    failed: bool
    reasons: List[str] = field(default_factory=list)  # subset of {"a","b","c"}
    rep_n: float = 0.0          # the computed seq-rep-n value (criterion c)
    rep_n_cutoff: float = DEFAULT_REP4_CUTOFF
    loop_pattern: Optional[str] = None    # diagnostic only (criterion b)
    loop_repeats: int = 0                 # diagnostic only (criterion b)


def _tokenize(text: str, tokens: Optional[Sequence] = None) -> List:
    """
    Return the unit sequence for the seq-rep-n metric.

    Prefer caller-supplied model tokens (matches the efficacy metric, which
    uses the model tokenizer). Otherwise fall back to whitespace words so the
    detector stays tokenizer-independent like degenerate_repeat_detector.
    """
    if tokens is not None:
        return list(tokens)
    if not text:
        return []
    return text.split()


def seq_rep_n(text: str = "", n: int = DEFAULT_REP_N,
              tokens: Optional[Sequence] = None) -> float:
    """
    seq-rep-n = 1 - |unique n-grams| / |total n-grams|  (Welleck et al. 2019).

    Fraction of a sequence's n-grams that are NOT unique -- i.e. how much of
    the trace is repeated material. Returns 0.0 when there are fewer than n
    units (no n-gram can repeat). Range [0, 1].
    """
    units = _tokenize(text, tokens)
    if len(units) < n:
        return 0.0
    ngrams = [tuple(units[i:i + n]) for i in range(len(units) - n + 1)]
    total = len(ngrams)
    if total == 0:
        return 0.0
    unique = len(set(ngrams))
    return 1.0 - (unique / total)


def detect_termination_failure(
    text: str = "",
    *,
    eos_emitted: bool,
    tokens: Optional[Sequence] = None,
    rep_n: int = DEFAULT_REP_N,
    rep_n_cutoff: float = DEFAULT_REP4_CUTOFF,
    num_generated_tokens: Optional[int] = None,
    max_new_tokens: Optional[int] = None,
) -> FailureResult:
    """
    Classify one generated trace as failure / non-failure per prereg §8.

    Args:
        text: decoded generation (used for criteria b and, if `tokens` not
            given, c).
        eos_emitted: True iff the model emitted EOS before the cap. Criterion
            (a) fires when this is False (the run is non-terminating). If
            `num_generated_tokens` and `max_new_tokens` are both supplied,
            reaching the cap is also treated as non-termination (cross-check).
        tokens: optional model-token sequence for the seq-rep-n metric
            (criterion c). Falls back to whitespace words.
        rep_n: n-gram order for the rep metric (default 4 = seq-rep-4).
        rep_n_cutoff: criterion (c) threshold; failure if rep-n exceeds it.
        num_generated_tokens, max_new_tokens: optional, for the cap
            cross-check on criterion (a).

    Returns:
        FailureResult. `failed` is True if ANY criterion fires; `reasons`
        lists which of {"a","b","c"} fired.
    """
    reasons: List[str] = []

    # (a) Non-termination: no EOS before the cap.
    hit_cap = (
        num_generated_tokens is not None
        and max_new_tokens is not None
        and num_generated_tokens >= max_new_tokens
    )
    if (not eos_emitted) or hit_cap:
        reasons.append("a")

    # (b) Structural loop -- loop flag ONLY (no clean_text / no recovery).
    loop = check_for_loops(text)
    loop_pattern = None
    loop_repeats = 0
    if loop.looped:
        reasons.append("b")
        loop_pattern = loop.pattern
        loop_repeats = loop.repeats

    # (c) rep-4 over the cutoff.
    rep_value = seq_rep_n(text, n=rep_n, tokens=tokens)
    if rep_value > rep_n_cutoff:
        reasons.append("c")

    return FailureResult(
        failed=bool(reasons),
        reasons=reasons,
        rep_n=rep_value,
        rep_n_cutoff=rep_n_cutoff,
        loop_pattern=loop_pattern,
        loop_repeats=loop_repeats,
    )
