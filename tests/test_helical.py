"""The helical bench, built on the ordinary machinery.

The point of this module is that there is almost nothing in it: `coupled` does
the synthesis, `synth` writes the topology, and everything after that is the
same solver, card, fitter and window every other netlist gets.  These tests
check the join, and the one thing that is genuinely helical -- that the
impedance level is a gauge and must never be fitted.
"""
import numpy as np
import pytest

from vnafit import helical, mna, structure
from vnafit.coupled import db


def _response(nl, n=3001):
    f = np.linspace(nl.ac[2], nl.ac[3], n)
    S = mna.build(nl, w_ref=2 * np.pi * nl.targets["f0"]).solve(f)
    return f, db(S[:, 1, 0])


@pytest.mark.parametrize("order", (1, 2, 3, 4))
def test_every_order_builds_and_hits_its_design_bandwidth(order):
    nl = helical.design(order, 146e6, 2e6, ripple_db=0.1, qu=670)
    assert len(nl.elements) == 3 * order + 1        # L, C per section; K; 2 taps
    f, d = _response(nl)
    i = int(np.argmax(d))
    inside = np.flatnonzero(d >= d[i] - 3)
    bw = f[inside[-1]] - f[inside[0]]
    want = nl.targets["bw3"]
    assert abs(bw / want - 1) < 0.07, f"{bw/1e6:.3f} against {want/1e6:.3f} MHz"
    # The BAND's centre, not the tallest point: a Chebyshev has `order` ripple
    # peaks and argmax picks whichever is a hair higher, which on a 2-pole sits
    # 480 kHz off centre and means nothing.
    centre = 0.5 * (f[inside[0]] + f[inside[-1]])
    # As a FRACTION of the bandwidth: a single resonator is 13 MHz wide, and
    # |S21| is symmetric in f/f0 - f0/f rather than in f, so a fixed hertz
    # tolerance asks a 1-pole and a 4-pole for quite different accuracies.
    assert abs(centre - 146e6) < 0.03 * bw, f"centred at {centre/1e6:.4f} MHz"


def test_more_sections_are_more_selective_and_lossier():
    """The trade the order choice exists to make."""
    got = []
    for order in (1, 2, 3, 4):
        nl = helical.design(order, 146e6, 2e6, 0.1, qu=670)
        f, d = _response(nl)
        i = int(np.argmax(d))
        edge = d[np.argmin(np.abs(f - (146e6 + 8e6)))]
        got.append((-d[i], d[i] - edge))
    il = [a for a, _s in got]
    rej = [s for _a, s in got]
    assert il == sorted(il), f"insertion loss should rise with order: {il}"
    assert rej == sorted(rej), f"rejection should rise with order: {rej}"


@pytest.mark.parametrize("order", (1, 2, 3, 4))
def test_the_impedance_gauge_is_never_offered_for_fitting(order):
    """With tapped ports every impedance scales together, so raising L and
    lowering C at fixed f0/k/Qe/Qu changes nothing measurable.  A fit that
    frees it slides along that flat direction and reports its bound."""
    nl = helical.design(order, 146e6, 2e6, 0.1)
    assert "lref" not in [f.param for f in nl.fits]

    # and the analysis finds it independently, from the circuit alone
    st = structure.analyse(nl, np.linspace(nl.ac[2], nl.ac[3], 300))
    gauge = [st.names[i] for v in st.nullvecs
             for i, c in enumerate(v) if abs(c) > 0.15]
    assert "lref" in gauge, f"the gauge was not found: {gauge}"


@pytest.mark.parametrize("order", (2, 3, 4))
def test_every_resonator_and_every_gap_is_separately_measurable(order):
    """The whole point of an alignment tool: a detuned section has to be
    visible as ITS OWN frequency, not absorbed into a shared one."""
    nl = helical.design(order, 146e6, 2e6, 0.1)
    st, _notes = structure.analyse_reduced(
        nl, np.linspace(nl.ac[2], nl.ac[3], 300))
    good = set(st.measurable())
    assert {f"f0{i}" for i in range(1, order + 1)} <= good
    assert {f"k{i}" for i in range(1, order)} <= good


def test_a_detuned_section_changes_the_response():
    nl = helical.design(3, 146e6, 2e6, 0.1)
    off = helical.design(3, 146e6, 2e6, 0.1, f0s=[146e6, 145.2e6, 146e6])
    _f, a = _response(nl)
    _f, b = _response(off)
    assert np.abs(a - b).max() > 3.0, "an 800 kHz detune should be obvious"


def test_the_order_is_bounded_and_says_so():
    with pytest.raises(ValueError, match="between 1 and 4"):
        helical.design(5)
    with pytest.raises(ValueError, match="between 1 and 4"):
        helical.design(0)
    with pytest.raises(ValueError, match="need 3 frequencies"):
        helical.design(3, f0s=[146e6, 146e6])


def test_choosing_an_order_in_the_window_rebuilds_and_loads_it():
    import tkinter as tk
    from vnafit.gui import App
    from vnafit.helicalwin import HelicalWindow
    try:
        root = tk.Tk()
    except tk.TclError:                                     # pragma: no cover
        pytest.skip("no display")
    root.withdraw()
    app = App(root, "examples/tinyfilter.net")
    try:
        w = HelicalWindow(root, app)
        assert "2 sections" in w.out.get("1.0", "end")
        w.v["order"].set("4")
        assert "4 sections" in w.out.get("1.0", "end"), "the order did not take"
        assert "f01, f02, f03, f04" in w.out.get("1.0", "end")

        w.load()
        assert len(app.nl.elements) == 13
        assert app.netlist_path is None, "a synthesised model has no file"
        assert set(app.rows) >= {"f01", "f04", "k3", "qu"}
    finally:
        root.destroy()


# ------------------------------------------------------------ the bench steps
@pytest.mark.parametrize("stage", helical.STAGES)
def test_each_bench_step_is_the_same_filter_with_the_rest_stripped_out(stage):
    nl = helical.design(3, 146e6, 2e6, 0.1, qu=670, stage=stage, section=1)
    want = {"align": 4, "couple": 4, "gap": 7, "measure": 10}[stage]
    assert len(nl.elements) == want
    assert nl.fits, "a stage with nothing to settle is not a bench step"
    assert helical.stage_hint(nl)


def test_each_step_measures_what_its_hint_says_it_measures():
    """The check that matters: build the stage, solve it, and confirm the
    number on screen is the number that comes out."""
    qu, order = 670.0, 3
    t = helical.design(order, 146e6, 2e6, 0.1, qu=qu).targets

    def width_and_split(nl):
        f = np.linspace(nl.ac[2], nl.ac[3], 12001)
        S = mna.build(nl, w_ref=2 * np.pi * 146e6).solve(f)
        d = db(S[:, 1, 0])
        i = int(np.argmax(d))
        ins = np.flatnonzero(d >= d[i] - 3)
        peaks = f[1:-1][(d[1:-1] > d[:-2]) & (d[1:-1] > d[2:])]
        split = abs(peaks[1] - peaks[0]) if len(peaks) >= 2 else None
        return f[ins[-1]] - f[ins[0]], split

    d = lambda **kw: helical.design(order, 146e6, 2e6, 0.1, qu=qu, **kw)

    bw, _ = width_and_split(d(stage="align", section=1))
    want = 146e6 / qu * (1 + 2 / helical.PROBE_Q)
    assert abs(bw / want - 1) < 0.03, f"align {bw/1e6:.4f} vs {want/1e6:.4f}"

    bw, _ = width_and_split(d(stage="couple", section=1))
    # 1/QL = 1/Qu + 1/Qe.  NOT f0/Qe -- with Qu/Qe about 9 that is 11% out and
    # aiming at it over-couples by the same amount.
    want = 146e6 * (1 / qu + 1 / t["Qe1"])
    assert abs(bw / want - 1) < 0.03, f"couple {bw/1e6:.4f} vs {want/1e6:.4f}"
    assert abs(bw / t["sl_bw1"] - 1) > 0.08, "the naive target should differ"

    _bw, split = width_and_split(d(stage="gap", section=1))
    assert split is not None
    assert abs(split / t["split"][0] - 1) < 0.03


def test_a_gap_needs_two_resonators():
    with pytest.raises(ValueError, match="gap needs two"):
        helical.design(1, stage="gap")


def test_the_window_offers_the_bench_sequence():
    import tkinter as tk
    from vnafit.gui import App
    from vnafit.helicalwin import HelicalWindow
    try:
        root = tk.Tk()
    except tk.TclError:                                     # pragma: no cover
        pytest.skip("no display")
    root.withdraw()
    app = App(root, "examples/tinyfilter.net")
    try:
        w = HelicalWindow(root, app)
        w.v["order"].set("3")
        for stage, expect in (("align", "Set section"), ("couple", "tap"),
                              ("gap", "aperture"), ("measure", "whole filter")):
            w.v["stage"].set(stage)
            assert expect in w.hint.cget("text"), (stage, w.hint.cget("text"))
        w.v["stage"].set("gap")
        w.load()
        assert len(app.nl.elements) == 7
        assert app.nl.stage == ("gap", 1)
    finally:
        root.destroy()
