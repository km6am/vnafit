"""Which netlist parameters can a 2-port measurement actually determine?

The netlist has more numbers in it than (f0, k, Qe, Qu) -- every L, C, R, M.
That does not mean the extra numbers are measurable, and reporting one that is
not is the single failure mode this project has already paid for once.

The impedance scale is the parameter at issue: raise every L and lower every C
at fixed f0, k, Qe and Qu, and ask whether |S21| notices.

RESULT, measured here, not argued:

  magnetic (tap)        3e-11 dB for a 100x change in L.  Machine precision.
                        The scale is an EXACT invariance, at every frequency,
                        not merely a narrowband one.  L is unmeasurable.

  capacitive (series C) 0.05-0.12 dB in-band, saturating near 0.28 dB wideband
                        for an unbounded change in L.  Broken, but only just.

The mechanism is the whole story.  A tap is an ideal autotransformer, so every
impedance in the network scales together and the network is invariant.  A series
input cap works against the fixed 50 ohm, which does NOT scale -- and the size
of the breaking is exactly u^2 = (w0*Cs*Z0)^2.  As L rises, Cs falls, u -> 0,
and the series cap becomes indistinguishable from an ideal transformer again.

The practical reading, against a NanoVNA's ~0.05 dB trace noise: in the magnetic
case L is unidentifiable, full stop; in the capacitive case it is WEAKLY
identifiable at best, and only from a wide span with good SNR.  Fit in
(f0, k, Qe) coordinates unless a profile shows otherwise.

An earlier estimate of "12 dB in-band" for the capacitive case was wrong: that
test changed Cs without retuning the resonator, so most of what it measured was
a detuned filter rather than a changed impedance level.
"""
import numpy as np
import pytest

from vnafit import coupled as M
from vnafit import mna, synth

F0, KS, QE, QU = 145.4e6, [0.0121], 46.0, 670.0
FSPAN = np.linspace(F0 * 0.6, F0 * 1.4, 1201)


def _s21(nl):
    return M.db(mna.build(nl, w_ref=2 * np.pi * F0).solve(FSPAN)[:, 1, 0])


def _delta(topo, la, lb):
    fn = synth.TOPOLOGIES[topo]
    a = _s21(fn([F0, F0], KS, QE, QE, QU, L=la))
    b = _s21(fn([F0, F0], KS, QE, QE, QU, L=lb))
    band = a > a.max() - 45
    return np.abs(a - b).max(), np.abs(a - b)[band].max()


def test_tapped_ports_make_the_impedance_scale_an_exact_invariance():
    """100x in L, at fixed (f0, k, Qe, Qu), must change nothing at all.

    Not "nothing much" -- nothing.  If this ever starts failing at the 1e-6
    level, something has introduced a fixed impedance into the network and the
    docstring above no longer describes it.
    """
    wide, inband = _delta("magnetic", 33e-9, 3300e-9)
    assert wide < 1e-9, f"expected exact invariance, got {wide:.2e} dB"
    assert inband < 1e-9


def test_a_physical_port_element_breaks_the_invariance_but_only_at_order_u2():
    """A series input cap does break it -- and by a knowable, small amount.

    Asserting both ends matters.  A lower bound proves the netlist is not
    secretly ignoring L; an upper bound stops anyone reading the difference as
    "L is measurable here".
    """
    wide, inband = _delta("capacitive", 30e-9, 120e-9)
    assert wide > 1e-4, "series-cap ports should NOT be scale-invariant"
    assert wide < 1.0, f"{wide:.3f} dB is far more than the u^2 mechanism allows"
    # and it is ~1e9 times larger than the tapped case, which is the real point
    tap_wide, _ = _delta("magnetic", 30e-9, 120e-9)
    assert wide > 1e6 * max(tap_wide, 1e-15)


def test_the_breaking_vanishes_as_u_goes_to_zero():
    """The mechanism test: the effect is entirely u^2 = (w0*Cs*Z0)^2.

    As L rises, the series cap needed for a given Qe shrinks, u falls, and the
    cap converges to an ideal transformer -- so the response must converge too.
    Successive doublings of L past that point must produce ever-smaller changes.
    """
    Ls = [150e-9, 300e-9, 600e-9, 1200e-9]
    us = [2 * np.pi * F0 * synth.series_cap_for_qe(F0, L, QE)[0] * 50.0 for L in Ls]
    assert us == sorted(us, reverse=True), "u should fall as L rises"

    steps = [_delta("capacitive", a, b)[0] for a, b in zip(Ls, Ls[1:])]
    assert steps == sorted(steps, reverse=True), (
        f"successive doublings should matter less and less, got {steps}")
    assert steps[-1] < 0.02, (
        f"the response has not converged as u -> 0: last step {steps[-1]:.4f} dB")


def test_the_breaking_is_below_or_near_instrument_noise():
    """The honest bound on what a real sweep could extract.

    A NanoVNA-H4 trace is good to roughly 0.05 dB.  A 4x error in L moves the
    in-band response by about that, so L is at best weakly identifiable even in
    the topology where it is identifiable at all.  This test exists so that
    number stays in front of anyone who adds `.fit L ...` to a netlist.
    """
    _, inband = _delta("capacitive", 30e-9, 120e-9)
    assert inband < 0.20, f"in-band sensitivity {inband:.3f} dB"
    assert inband > 0.005, "sanity: there should be *some* dependence"


@pytest.mark.parametrize("topo", ["magnetic", "capacitive"])
def test_f0_k_and_qe_are_held_by_construction(topo):
    """The invariance claims only mean anything if the synthesiser really is
    holding (f0, k, Qe) fixed while L moves.  Check that, rather than assuming
    it -- an L-dependent detuning would masquerade as sensitivity to L."""
    fn = synth.TOPOLOGIES[topo]
    Ls = (30e-9, 120e-9) if topo == "capacitive" else (33e-9, 3300e-9)
    f = np.linspace(F0 - 6e6, F0 + 6e6, 601)
    seed = dict(f0s=[F0, F0], ks=list(KS), Qe1=QE, Qen=QE)
    got = []
    for L in Ls:
        S = mna.build(fn([F0, F0], KS, QE, QE, QU, L=L),
                      w_ref=2 * np.pi * F0).solve(f)
        r = M.fit(f, S[:, 0, 0], S[:, 1, 0], 2, QU, seed=seed,
                  coupling=synth.PREDICTED_LAW[topo])
        got.append(r)
    for key in ("Qe1", "Qen"):
        assert abs(got[0][key] / got[1][key] - 1) < 0.02, f"{key} moved with L"
    assert abs(got[0]["ks"][0] / got[1]["ks"][0] - 1) < 0.02, "k moved with L"
    assert abs(got[0]["f0s"][0] - got[1]["f0s"][0]) < 2e-4 * F0, "f0 moved with L"
