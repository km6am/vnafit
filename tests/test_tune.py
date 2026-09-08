"""Tuning advice that the impedance scale cannot move.

The fitter walks a long way along the impedance scale -- L up, every C down by
the same factor -- because the response barely changes along it.  A readout in
picofarads is worthless there: a live fit asked for a 32 pF trimmer to be
changed to 301 pF, which was a scale artefact and not a tuning error.
"""
import numpy as np
import pytest

from vnafit import spice as SP, tune

LT = "examples/tinyfilter_ltspice.net"
FITTED = dict(cm=1.89476e-12, cin=6.17911e-12, c1=3.16671e-11,
              c2=3.60571e-11, c3=3.19575e-11)


def _load():
    nl, _r = SP.load_any(LT)
    target = nl.resolve_params()
    fitted = dict(target)
    fitted.update(FITTED)
    return nl, fitted, target


def _gauge(p, a):
    """Move along the impedance scale: L up by a, every C down by a."""
    out = dict(p)
    out["l"] *= a
    out["qr"] *= a                     # R = wL/Q tracks L under constant Q
    for c in ("cm", "cin", "c1", "c2", "c3"):
        out[c] /= a
    return out


def _pct(nl, fitted, target):
    r, s = tune.resonators(nl, fitted), tune.resonators(nl, target)
    return [((a["f0"] / b["f0"]) ** 2 - 1) * a["ctot"] / a["ctrim"]
            for a, b in zip(r, s)]


def test_the_middle_resonator_advice_is_exactly_invariant():
    """It touches no port impedance, so nothing in its advice knows the scale."""
    nl, fitted, target = _load()
    ref = _pct(nl, fitted, target)[1]
    for a in (0.25, 2.0, 8.3, 30.0):
        got = _pct(nl, _gauge(fitted, a), target)[1]
        assert abs(got - ref) < 1e-9, (a, got, ref)


def test_an_end_resonator_degrades_gracefully_instead_of_catastrophically():
    """The end capacitor works into the port impedance, which does NOT scale --
    the one term that makes the impedance level identifiable at all.  So an end
    resonator's advice is not exactly invariant; it just stops being nonsense.
    """
    nl, fitted, target = _load()
    ref = _pct(nl, fitted, target)[0]
    err = {a: abs(_pct(nl, _gauge(fitted, a), target)[0] - ref)
           for a in (2.0, 8.3)}
    assert max(err.values()) < 0.02, err        # under 2 percentage points

    # against what the picofarad readout did over the same move
    naive = {a: abs((target["c1"] / _gauge(fitted, a)["c1"] - 1) -
                    (target["c1"] / fitted["c1"] - 1)) for a in (2.0, 8.3)}
    assert min(naive.values()) > 1.0            # over 100 percentage points
    assert min(naive.values()) / max(err.values()) > 50


def test_the_advice_names_the_right_resonator_to_turn():
    nl, fitted, target = _load()
    lines = tune.advice(nl, fitted, target)
    assert lines and lines[0].startswith("resonator")
    body = "\n".join(lines)
    assert "C1" in body and "C2" in body and "C3" in body
    # C2 is the one 416 kHz low; C3 is within 72 kHz and should not be flagged
    turn = [l for l in lines if "<- turn" in l]
    assert any(l.strip().startswith("C1") for l in turn), body
    assert not any(l.strip().startswith("C3") for l in turn), body


def test_the_fixed_parts_are_reported_as_ratios_not_values():
    """k = Cm/sqrt(Ctot_i*Ctot_j) is a ratio of capacitances, so it says
    something about the build rather than about where the fit parked.

    Not EXACTLY invariant on this filter: both couplings land on an end
    resonator, whose total capacitance includes the end cap working into the
    port impedance -- the term that does not scale.  So it drifts a few percent
    where the raw capacitance drifts by hundreds.
    """
    nl, fitted, target = _load()
    ks = np.array([k for _n, k in tune.couplings(nl, fitted,
                                                 tune.resonators(nl, fitted))])
    for a in (0.25, 8.3):
        g = _gauge(fitted, a)
        gk = np.array([k for _n, k in tune.couplings(nl, g,
                                                     tune.resonators(nl, g))])
        drift = np.abs(gk / ks - 1).max()
        raw = abs(g["cm"] / fitted["cm"] - 1)
        assert drift < 0.05, (a, drift)
        assert raw / drift > 15, (a, raw, drift)


def test_a_netlist_with_no_resonators_says_nothing_rather_than_guessing():
    nl, _r = SP.load_any("examples/topologies/lowpass_cheb5.net")
    p = nl.resolve_params()
    assert tune.resonators(nl, p) == []
    assert tune.advice(nl, p, p) == []
