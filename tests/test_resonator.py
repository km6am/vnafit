"""Lumped vs distributed resonators, and the confound between the two.

A helical is a shorted quarter-wave line, not a parallel L||C.  The two agree
near f0 and diverge steadily outside it, and the divergence is ANTISYMMETRIC: a
line is less detuned than its lumped stand-in below f0 and more detuned above.

That matters far more than it sounds, because the antisymmetry is the same shape
a frequency-dependent coupling produces.  Fit a distributed resonator with a
lumped model and the coupling law will happily absorb the error -- and you will
conclude the coupling drifts with frequency when it does not.  These tests pin
the signature down and then demonstrate the confound deliberately.
"""
import numpy as np
import pytest

from vnafit import coupled as M

F0 = 146.75e6
KS = [0.0147]
QE = 110.0
QU = 900.0


def test_the_two_resonators_agree_in_the_passband():
    f = np.linspace(F0 * 0.97, F0 * 1.03, 401)
    a = M._detune(f, F0, "lumped")
    b = M._detune(f, F0, "quarter")
    ok = np.abs(f - F0) > 1e-6 * F0
    assert np.abs(b[ok] / a[ok] - 1).max() < 0.02


def test_the_divergence_is_antisymmetric():
    """The discriminating signature: a line's low skirt sits ABOVE a lumped fit
    and its high skirt BELOW.  Predicting the sign on both sides at once is what
    makes this a test rather than one more free knob."""
    for r, expect_smaller in ((0.70, True), (0.85, True),
                              (1.15, False), (1.26, False)):
        f = np.array([F0 * r])
        a = abs(M._detune(f, F0, "lumped")[0])
        b = abs(M._detune(f, F0, "quarter")[0])
        if expect_smaller:
            assert b < a, f"at {r:.2f}*f0 the line should be LESS detuned"
        else:
            assert b > a, f"at {r:.2f}*f0 the line should be MORE detuned"


def test_the_line_resonates_again_at_three_f0():
    """Physical, and the reason a lumped model cannot be trusted wideband.

    Note the direction, which I first got backwards: the detuning goes to ZERO
    at 3*f0, because zero detuning IS resonance.  A shorted quarter-wave line is
    also three-quarters of a wave long at 3*f0 and resonates again there.  The
    lumped stand-in is far off resonance at that frequency and knows nothing
    about it.
    """
    assert abs(M._detune(np.array([3.0 * F0]), F0, "quarter")[0]) < 1e-9
    assert abs(M._detune(np.array([3.0 * F0]), F0, "lumped")[0]) > 2.5
    # and it is a genuine pole of the detuning midway between, at 2*f0
    assert abs(M._detune(np.array([2.0 * F0]), F0, "quarter")[0]) > 1e12


def test_unknown_resonator_is_refused():
    with pytest.raises(ValueError, match="unknown resonator"):
        M.response(np.array([F0]), [F0, F0], KS, QE, QE, QU, resonator="helix")


def _synth(resonator, coupling, f):
    s11, s21 = M.response(f, [F0, F0 * 1.004], KS, QE, QE * 1.4, QU,
                          coupling=coupling, resonator=resonator)
    return s11, s21


def test_round_trip_through_the_right_resonator():
    f = np.linspace(F0 * 0.72, F0 * 1.26, 900)
    s11, s21 = _synth("quarter", "constant", f)
    seed = dict(f0s=[F0] * 2, ks=list(KS), Qe1=QE, Qen=QE * 1.4)
    right = M.fit(f, s11, s21, 2, QU, seed=seed, resonator="quarter")
    wrong = M.fit(f, s11, s21, 2, QU, seed=seed, resonator="lumped")
    assert right["rms"] < 0.15, right["rms"]
    assert wrong["rms"] > 10 * right["rms"]


def test_a_lumped_model_invents_a_coupling_law_that_is_not_there():
    """The confound, demonstrated on data whose truth is known.

    The data is generated from a quarter-wave resonator with a genuinely
    CONSTANT coupling.  Fit it with a lumped resonator and `helical_tapped`
    comes out ahead of `constant` -- a frequency-dependent coupling that does
    not exist, manufactured entirely by the wrong resonator.  Fit it with the
    right resonator and `constant` wins, as it should.

    This is what happened on the live 2 m helical sweep: with a lumped
    resonator `helical_tapped` looked 0.33 dB better than `constant`, and with
    the quarter-wave resonator `constant` was 0.52 dB better instead.
    """
    f = np.linspace(F0 * 0.72, F0 * 1.26, 900)
    s11, s21 = _synth("quarter", "constant", f)
    seed = dict(f0s=[F0] * 2, ks=list(KS), Qe1=QE, Qen=QE * 1.4)
    got = {}
    for reso in ("lumped", "quarter"):
        for law in ("constant", "helical_tapped"):
            got[(reso, law)] = M.fit(f, s11, s21, 2, QU, seed=seed,
                                     coupling=law, resonator=reso)["rms"]
    assert got[("lumped", "helical_tapped")] < got[("lumped", "constant")], (
        "the confound did not reproduce; this test is no longer testing it")
    assert got[("quarter", "constant")] < got[("quarter", "helical_tapped")], (
        "with the right resonator, the non-existent law must not win")
    assert got[("quarter", "constant")] < got[("lumped", "helical_tapped")], (
        "fixing the resonator must beat papering over it with a coupling law")
