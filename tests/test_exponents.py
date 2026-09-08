"""Measuring the coupling exponents by RETUNING, instead of by skirt-fitting.

Why this exists: on a narrowband filter the skirt-fit route is dead.  The
coupling laws differ only far down the skirts, and on a 2 m helical measured
with a NanoVNA-H4 they separate by ~1 dB over the usable band against a ~1.2 dB
residual.  Averaging does not help -- the instrument's own residual crosstalk,
measured bare with 8 averages at the narrowest IF, is about -86 dB, and settling
it that way needs roughly -100 dB.

Retuning has 80 dB of headroom instead of 3, because k and Qe are read in the
passband.

THE TRAP THESE TESTS GUARD, which I walked into while writing them: the retuning
exponents are NOT the within-sweep exponents in COUPLING_LAWS.  Those describe
how k drifts across one sweep at fixed hardware; these describe how k moves when
the resonator capacitance changes.  A cap-coupled filter is (+1, -2) across a
sweep and (+2, -3) under retuning.  Comparing a measured retuning slope against
COUPLING_LAWS would name the wrong topology with total confidence.
"""
import numpy as np
import pytest

from vnafit import coupled as M
from vnafit import mna
from vnafit.netlist import parse

QU = 500.0

# One physical filter each, retuned ONLY by the resonator capacitance -- which
# is what a tuning screw does.  Coupling element and port element stay fixed.
CAP_COUPLED = (""".param c=39p cm=0.9p cs=4.3p L=60n
.port 1 p1 0
.port 2 p2 0
Cin p1 n1 {cs}
L1 n1 0 {L} Q=500
C1 n1 0 {c}
Cm n1 n2 {cm}
L2 n2 0 {L} Q=500
C2 n2 0 {c}
Cout p2 n2 {cs}""", 39e-12, 60e-9)

TAPPED_MAGNETIC = (""".param c=3.6p L=330n nt=0.06 k=0.012
.port 1 p1 0
.port 2 p2 0
X1 p1 0 n1 0 n={nt}
L1 n1 0 {L} Q=500
C1 n1 0 {c}
L2 n2 0 {L} Q=500
C2 n2 0 {c}
K1 L1 L2 {k}
X2 p2 0 n2 0 n={nt}""", 3.6e-12, 330e-9)


def _retuned_sweeps(text, cbase, L, mults=(1.5, 1.2, 1.0, 0.82, 0.68)):
    nl = parse(text)
    out = []
    for m in mults:
        c = cbase * m
        f0 = 1 / (2 * np.pi * np.sqrt(L * c))
        f = np.linspace(f0 * 0.90, f0 * 1.10, 801)
        S = mna.build(nl, {"c": c}, w_ref=2 * np.pi * f0).solve(f)
        out.append((f, S[:, 0, 0], S[:, 1, 0]))
    return out


@pytest.mark.parametrize("name,spec,want", [
    ("capacitive", CAP_COUPLED, (2.0, -3.0)),
    ("helical_tapped", TAPPED_MAGNETIC, (0.0, -1.0)),
])
def test_retuning_recovers_the_topology(name, spec, want):
    res = M.measure_exponents(_retuned_sweeps(*spec), 2, QU)
    assert abs(res["kexp"] - want[0]) < 0.35, f"kexp {res['kexp']:+.2f} want {want[0]:+.0f}"
    assert abs(res["qexp"] - want[1]) < 0.40, f"qexp {res['qexp']:+.2f} want {want[1]:+.0f}"
    got, _, dist, law = M.nearest_topology(res["kexp"], res["qexp"])
    assert got == name, f"identified {got}, expected {name}"
    assert law == name


def test_the_two_topologies_are_far_apart_under_retuning():
    """The whole reason to prefer this over the skirt fit.

    Under retuning the two topologies sit at (+2, -3) and (0, -1) -- a distance
    of 2.8 in exponent space, measured in the passband.  Across a sweep they sit
    at (+1, -2) and (-1, 0), which is the same distance but has to be read off
    the skirts, where a real fixture has only leakage.
    """
    a = M.measure_exponents(_retuned_sweeps(*CAP_COUPLED), 2, QU)
    b = M.measure_exponents(_retuned_sweeps(*TAPPED_MAGNETIC), 2, QU)
    sep = np.hypot(a["kexp"] - b["kexp"], a["qexp"] - b["qexp"])
    assert sep > 2.0, f"topologies only {sep:.2f} apart in exponent space"


def test_retuning_exponents_are_not_the_within_sweep_ones():
    """Guards the category error directly.

    If someone ever 'simplifies' measure_exponents to compare against
    COUPLING_LAWS, this fails: the cap-coupled filter retunes at kexp near +2,
    which COUPLING_LAWS does not contain at all.
    """
    res = M.measure_exponents(_retuned_sweeps(*CAP_COUPLED), 2, QU)
    within = M.COUPLING_LAWS["capacitive"]
    assert abs(res["kexp"] - within[0]) > 0.8, (
        "retuning and within-sweep exponents came out the same; one of them is "
        "being computed wrongly")
    assert M.RETUNE_LAWS["capacitive"][2] == "capacitive"


def test_two_tunings_give_a_slope_but_no_uncertainty():
    sw = _retuned_sweeps(*TAPPED_MAGNETIC, mults=(1.4, 0.7))
    res = M.measure_exponents(sw, 2, QU)
    assert np.isfinite(res["kexp"]) and np.isnan(res["kexp_se"])
    with pytest.raises(ValueError, match="at least two"):
        M.measure_exponents(sw[:1], 2, QU)
