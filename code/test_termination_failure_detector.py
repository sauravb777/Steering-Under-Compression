"""Unit tests for termination_failure_detector (prereg §8).

Run from D:\\Claude\\projects\\steering-quantization with D:\\Claude\\Tools on
sys.path (the module's import shim also locates Tools relative to itself):

    python -m pytest test_termination_failure_detector.py -q
"""

import pytest

from termination_failure_detector import (
    detect_termination_failure,
    seq_rep_n,
    DEFAULT_REP4_CUTOFF,
)

CLEAN = (
    "First I consider the premises carefully. Then I weigh the evidence "
    "for each option, comparing their relative strengths. Finally I reach a "
    "conclusion and explain why it follows from the reasoning above."
)


# --- seq_rep_n metric ----------------------------------------------------

def test_seq_rep_n_zero_on_short_input():
    assert seq_rep_n("one two three", n=4) == 0.0

def test_seq_rep_n_zero_on_unique_text():
    assert seq_rep_n(CLEAN, n=4) == 0.0

def test_seq_rep_n_high_on_repetition():
    text = "the cat sat down " * 10
    assert seq_rep_n(text, n=4) > 0.8

def test_seq_rep_n_accepts_token_sequence():
    toks = [1, 2, 3, 4] * 10
    assert seq_rep_n(tokens=toks, n=4) > 0.8


# --- criterion (a): non-termination -------------------------------------

def test_a_no_eos_is_failure():
    r = detect_termination_failure(CLEAN, eos_emitted=False)
    assert r.failed and "a" in r.reasons

def test_a_eos_clean_is_not_failure():
    r = detect_termination_failure(CLEAN, eos_emitted=True)
    assert not r.failed and r.reasons == []

def test_a_cap_crosscheck_fires():
    r = detect_termination_failure(
        CLEAN, eos_emitted=True,
        num_generated_tokens=2048, max_new_tokens=2048,
    )
    assert r.failed and "a" in r.reasons

def test_a_under_cap_does_not_fire():
    r = detect_termination_failure(
        CLEAN, eos_emitted=True,
        num_generated_tokens=500, max_new_tokens=2048,
    )
    assert not r.failed


# --- criterion (b): structural loop -------------------------------------

def test_b_structural_loop_is_failure():
    text = "Let me think. " + "main entry point " * 8
    r = detect_termination_failure(text, eos_emitted=True)
    assert r.failed and "b" in r.reasons
    assert r.loop_repeats >= 2

def test_b_clean_text_no_loop():
    r = detect_termination_failure(CLEAN, eos_emitted=True)
    assert "b" not in r.reasons


# --- criterion (c): rep-4 cutoff ----------------------------------------

def test_c_high_rep4_is_failure():
    text = ("alpha beta gamma delta " * 6)
    r = detect_termination_failure(text, eos_emitted=True)
    assert r.failed
    assert r.rep_n > DEFAULT_REP4_CUTOFF

def test_c_clean_below_cutoff():
    r = detect_termination_failure(CLEAN, eos_emitted=True)
    assert r.rep_n <= DEFAULT_REP4_CUTOFF
    assert "c" not in r.reasons


def test_c_uses_supplied_tokens_over_text():
    # S3 regression: the harness now passes model TOKEN ids. Text is clean (no
    # repeated word 4-gram) but the tokens are highly repetitive; criterion (c)
    # must fire from the TOKENS (prereg sec.8), proving detect_termination_failure
    # scores rep-4 on the supplied tokens, not the decoded text.
    clean_text = "each of these particular words appears exactly once with no repeats"
    repetitive_tokens = [1, 2, 3, 4] * 10
    r_tokens = detect_termination_failure(
        clean_text, eos_emitted=True, tokens=repetitive_tokens)
    assert "c" in r_tokens.reasons and r_tokens.rep_n > DEFAULT_REP4_CUTOFF
    # Same clean text WITHOUT tokens -> word fallback -> (c) does NOT fire.
    r_text = detect_termination_failure(clean_text, eos_emitted=True)
    assert "c" not in r_text.reasons


# --- rep-4 robustness sweep {0.4, 0.5, 0.6} -----------------------------

def test_rep4_sweep_monotonic():
    units = ["w%d" % i for i in range(20)] + ["x", "y", "z", "q"] * 5
    text = " ".join(units)
    fired = {c: seq_rep_n(text, n=4) > c for c in (0.4, 0.5, 0.6)}
    # a higher cutoff can never fire when a lower one does not
    assert not (fired[0.6] and not fired[0.4])

@pytest.mark.parametrize("cutoff", [0.4, 0.5, 0.6])
def test_cutoff_is_configurable(cutoff):
    text = "the cat sat down " * 10
    r = detect_termination_failure(text, eos_emitted=True, rep_n_cutoff=cutoff)
    assert r.rep_n_cutoff == cutoff
    assert "c" in r.reasons  # heavy repetition exceeds all three


# --- multiple criteria can fire together --------------------------------

def test_multiple_reasons():
    text = "loop loop loop loop loop loop "
    r = detect_termination_failure(text, eos_emitted=False)
    assert r.failed
    assert "a" in r.reasons
    assert "b" in r.reasons or "c" in r.reasons


# --- detection-only invariant: no clean_text leakage --------------------

def test_result_has_no_recovery_fields():
    r = detect_termination_failure("anything here", eos_emitted=True)
    assert not hasattr(r, "clean_text")
