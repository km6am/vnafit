"""Structural identifiability, computed from the netlist with no measurement.

Which parameters a sweep can determine is a property of the circuit, so it is
computable up front.  These tests pin down the two things that decide whether
the computation is right, both of which were wrong in the first version.
"""
import numpy as np
import pytest

from vnafit import structure as ST
from vnafit.netlist import parse

TAPPED = """tapped helical
.param f0=145.4Meg L=330n qu=670 k=0.0121 qe=46
.port 1 p1 0
.port 2 p2 0
X1 p1 0 n1 0 n={sqrt(50/(2*pi*f0*L*qe))}
L1 n1 0 {L} Q={qu}
C1 n1 0 {1/(4*pi*pi*f0*f0*L)}
L2 n2 0 {L} Q={qu}
C2 n2 0 {1/(4*pi*pi*f0*f0*L)}
K1 L1 L2 {k}
X2 p2 0 n2 0 n={sqrt(50/(2*pi*f0*L*qe))}
.ac lin 300 100Meg 200Meg"""

LADDER = """ladder with a derived inductance
.param al=5.2n turns=24 lsh={al*turns*turns} ql=180 unused=1Meg
.port 1 p1 0
.port 2 p2 0
C1 p1 a 3300p
L1 a 0 {lsh} Q={ql}
C2 a b 1000p
L2 b 0 {lsh} Q={ql}
C3 b p2 3300p
.ac dec 300 300k 60Meg"""

# Two degeneracies at once, and neither is expressible as a single SVD row:
# `al`/`turns` (only their product `lsh` acts), and the middle node's impedance
# scale (L2 up, C2 down, cm up leaves the response alone -- an INTERNAL node has
# a scale freedom that the end nodes do not, because `cin` works against the
# fixed 50 ohm).  Rank is 7 of 9, so the report must cost exactly two
# coordinates.  Reading the SVD rows one at a time cost six.
TWO_GAUGES = """two overlapping degeneracies
.param L=68n qu=150 cm=1.2p cin=5.0p c1=20p c2=22p c3=20p
.param al=3.5n turns=4 lsh={al*turns*turns}
.port 1 p1 0 Z0=50
.port 2 p2 0 Z0=50
Cin p1 n1 {cin}
L1  n1 0 {L} Q={qu}
C1  n1 0 {c1}
Cm1 n1 n2 {cm}
L2  n2 0 {lsh} Q={qu}
C2  n2 0 {c2}
Cm2 n2 n3 {cm}
L3  n3 0 {L} Q={qu}
C3  n3 0 {c3}
Cout p2 n3 {cin}
.ac lin 401 60Meg 300Meg"""


def test_a_tap_makes_the_impedance_scale_a_gauge():
    """The finding this project derived by hand, now found automatically from
    the netlist with no measurement at all."""
    st = ST.analyse(parse(TAPPED))
    assert st.verdict["l"][0] == "gauge"
    for other in ("f0", "k", "qe", "qu"):
        assert st.verdict[other][0] == "determined"
    assert st.rank == 4 and set(st.ranks) == {4}


def test_parameters_that_only_act_through_a_product_are_flagged():
    """`al` and `turns` reach the circuit only as `al*turns^2`, so no sweep can
    separate them -- which also means my earlier claim that fitting a sweep
    'identifies the core' held only because the turn count was fixed."""
    st = ST.analyse(parse(LADDER))
    assert st.verdict["al"][0] == "degenerate"
    assert st.verdict["turns"][0] == "degenerate"
    assert st.verdict["al"][1] == st.verdict["turns"][1], "same group"
    assert st.verdict["ql"][0] == "determined"


def test_a_parameter_no_element_uses_is_caught():
    st = ST.analyse(parse(LADDER))
    assert st.verdict["unused"][0] == "gauge"
    assert st.drives("unused") == []


def test_derived_parameters_must_be_allowed_to_recompute():
    """Overriding a derived parameter as well pins the dependency being
    perturbed.  Doing that made `al` and `turns` look individually free."""
    nl = parse(LADDER)
    assert "lsh" not in ST.independent(nl)
    base = nl.resolve_params()
    moved = nl.resolve_params({"al": base["al"] * 2})
    assert abs(moved["lsh"] / base["lsh"] - 2) < 1e-12, "lsh did not recompute"


def test_the_rank_does_not_depend_on_the_step_size():
    """At h=1e-2 an exactly degenerate pair showed a relative singular value of
    1.2e-2 -- big enough to read as a real direction.  It was truncation error
    and fell linearly with h.  A fixed step and a fixed threshold gets the rank
    wrong, so the analysis must agree across step sizes."""
    nl = parse(LADDER)
    names = ST.independent(nl)
    f = np.logspace(np.log10(3e5), np.log10(6e7), 300)
    p0 = {n: nl.resolve_params()[n] for n in names}
    svs = []
    for h in (1e-3, 1e-4, 1e-5):
        s = np.linalg.svd(ST.jacobian(nl, names, f, p0, h), compute_uv=False)
        svs.append(s / s[0])
    # the spurious direction shrinks by ~10x for each 10x in h
    assert svs[0][2] > 3 * svs[1][2] > 9 * svs[2][2]
    # and the real ones do not move
    assert abs(svs[0][1] / svs[2][1] - 1) < 1e-3


def test_the_invariant_is_named_from_the_netlist():
    """`al` and `turns` collapse to `lsh`, which the netlist already names,
    found by moving along the null direction and seeing what does not budge."""
    st = ST.analyse(parse(LADDER))
    coords, notes = ST.suggest_coordinates(st)
    assert "lsh" in coords and "ql" in coords
    assert "al" not in coords and "turns" not in coords
    assert "unused" not in coords
    assert any("lsh" in n for n in notes)


def test_the_reduced_analysis_has_no_degeneracies_left():
    st, notes = ST.analyse_reduced(parse(LADDER))
    assert sorted(st.names) == ["lsh", "ql"]
    assert st.rank == 2 and not st.null
    assert all(v[0] == "determined" for v in st.verdict.values())


def test_element_params_stops_at_the_chosen_coordinates():
    """After collapsing, expanding all the way to the leaves would report `al`
    and `turns` again -- the very names the report just said not to use -- and
    `lsh` would appear to drive nothing."""
    nl = parse(LADDER)
    leaves = ST.element_params(nl)
    assert leaves["L1"] == {"al", "turns", "ql"}
    reduced = ST.element_params(nl, ["lsh", "ql"])
    assert reduced["L1"] == {"lsh", "ql"}
    assert reduced["C1"] == set(), "a literal element has no parameters"


def test_describe_null_reports_the_invariant_not_the_direction():
    st = ST.analyse(parse(LADDER))
    said = [ST.describe_null(st.names, v)[1] for v in st.nullvecs]
    joined = " ".join(said)
    assert "al * turns^2" in joined, joined
    assert "1/al" not in joined, "printed the reciprocal of the invariant"


def test_a_netlist_with_no_params_says_so():
    with pytest.raises(ValueError, match="no .param"):
        ST.analyse(parse(".port 1 a 0\n.port 2 b 0\nR1 a b 50\nR2 b 0 1e9"))


ASYM = """per-resonator tuning
.param f01=145.4Meg f02=145.4Meg L=330n qu=670 k=0.0121 qe1=46 qe2=46
.port 1 p1 0
.port 2 p2 0
X1 p1 0 n1 0 n={sqrt(50/(2*pi*f01*L*qe1))}
L1 n1 0 {L} Q={qu}
C1 n1 0 {1/(4*pi*pi*f01*f01*L)}
L2 n2 0 {L} Q={qu}
C2 n2 0 {1/(4*pi*pi*f02*f02*L)}
K1 L1 L2 {k}
X2 p2 0 n2 0 n={sqrt(50/(2*pi*f02*L*qe2))}
.ac lin 300 135.4Meg 155.4Meg"""


def test_one_shared_f0_cannot_express_a_detune():
    """A model with a single f0 for both resonators is symmetric by
    construction, so no fit of it can ever report a detune -- which made the
    netlist strictly less expressive than the coupling-matrix model it was meant
    to supersede, where f0 is a vector.
    """
    from vnafit import mna
    sym = parse(TAPPED)
    assert "f01" not in ST.independent(sym) and "f0" in ST.independent(sym)
    caps = [e for e in sym.elements if e.kind == "C"]
    srcs = {e.args["value"].src for e in caps}
    assert len(srcs) == 1, "both resonator capacitors share one expression"


def test_per_resonator_tuning_makes_the_detune_identifiable():
    st, _notes = ST.analyse_reduced(parse(ASYM))
    for n in ("f01", "f02", "qe1", "qe2", "k", "qu"):
        assert st.verdict[n][0] == "determined", (n, st.verdict[n])
    assert st.drives("f01") == ["C1", "X1"]
    assert st.drives("f02") == ["C2", "X2"]


def test_the_restart_check_discards_out_of_window_trials():
    """Perturbed too hard, the passband walks off the edge of the analysis
    window, every parameter loses leverage in band, and the rank drops for a
    reason that has nothing to do with the circuit.  Observed as rank
    [4, 5, 6] on a filter whose true rank is 6, every low trial's peak sitting
    on a window boundary.
    """
    nl = parse(ASYM)
    # The property that was broken is STABILITY of the rank across trials, not
    # its value: the raw analysis carries the gauge `L`, so 6 of 7 is correct
    # here.  It used to come back [4, 5, 6].
    for j in (0.05, 0.10, 0.20):
        st = ST.analyse(nl, jitter=j)
        assert len(set(st.ranks)) == 1, (j, st.ranks)
        assert st.ranks[0] == len(st.names) - 1, (j, st.ranks, st.names)

    # The guard fires when it should, and the count is reported rather than the
    # trials being silently dropped.
    violent = ST.analyse(nl, jitter=0.50)
    assert violent.skipped > 5, violent.skipped
    assert "discarded as out of window" in violent.report()

    # It does NOT rescue arbitrarily large jitter, and should not pretend to:
    # at +-65% the passband can sit inside the window while the filter has been
    # distorted into something else, and a parameter genuinely loses leverage.
    # The jitter exists to check a degeneracy is not an accident of the nominal
    # point, not to explore the parameter space.
    assert ST.analyse(nl, jitter=0.10).skipped < violent.skipped


def test_the_tinyfilter_example_matches_figure_3a_of_its_note():
    """Figure 3a of the K6STR note gives the final component values outright.

    The previous version of this test asserted a 5.01 MHz bandwidth "the note
    gives" -- a number the note does not contain.  It came from an earlier
    version of the netlist that DERIVED the values from an assumed 4 MHz ripple
    bandwidth, itself inferred from the note's ~3 dB insertion loss, which is
    circular: the loss is set mostly by unloaded Q.  The coupling came out half
    right and the model 4.65 MHz wide against a design that is 9.75.

    So this checks the values against the figure, and the response against
    Figure 2b, which shows a flat top from about 141 to 150 MHz.
    """
    import os
    from vnafit import netlist as _nl, sweep
    from vnafit.coupled import db
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    nl = _nl(os.path.join(here, "examples", "tinyfilter.net"))
    p = nl.resolve_params()
    assert p["cin"] == 6.27e-12 and p["cm"] == 1.78e-12       # Figure 3a
    assert p["c1"] == p["c3"] == 3.204e-11 and p["c2"] == 3.606e-11
    assert abs(p["l"] - 3e-8) < 1e-20

    f = np.linspace(120e6, 180e6, 6001)
    _f, S = sweep(nl, f, {"qu": 1e9})           # lossless: the design geometry
    d = db(S[:, 1, 0])
    inside = np.flatnonzero(d >= d.max() - 3.0)
    lo, hi = f[inside[0]] / 1e6, f[inside[-1]] / 1e6
    assert 9.0 < hi - lo < 10.5, f"3 dB BW {hi-lo:.2f} MHz; Figure 2b shows ~10"
    assert 140.0 < lo < 142.0 and 149.5 < hi < 151.5, (lo, hi)
    pb = (f >= 144e6) & (f <= 148e6)
    assert d[pb].max() - d[pb].min() < 0.45, "ripple far from the 0.25 dB design"

    # The note says ~3 dB of insertion loss.  These values at Qu = 198 give
    # 1.5 -- so Qu = 198 is NOT the note's, it was reverse-engineered from the
    # loss by an earlier version of this file, and the measurement agrees with
    # the loss rather than with the Qu: fitting a real unit gives Qu = 113.
    _f, S = sweep(nl, f)
    il = -db(S[:, 1, 0])[np.argmin(np.abs(f - 146e6))]
    assert 1.2 < il < 1.8, f"insertion loss {il:.2f} dB at Qu=198"
    _f, S = sweep(nl, f, {"qu": 113.0})
    il = -db(S[:, 1, 0])[np.argmin(np.abs(f - 146e6))]
    assert abs(il - 2.5) < 0.4, f"at the measured Qu the loss is {il:.2f} dB"


def test_series_end_caps_make_the_inductance_identifiable():
    """The contrast that ties the whole identifiability thread together.

    On the tapped helical the impedance scale is an exact gauge and L is
    unmeasurable.  The TinyFilter feeds through series capacitors into the fixed
    50 ohm, which does not scale, so L is structurally identifiable instead.

    Structurally.  On real data it is another matter -- fitting L on the live
    unit halved it and doubled every capacitor, at condition number 5.6e3, along
    exactly the scale direction.  Necessary, not sufficient.
    """
    import os
    from vnafit import netlist as _nl
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    st = ST.analyse(_nl(os.path.join(here, "examples", "tinyfilter.net")))
    assert st.verdict["l"][0] == "determined"
    assert not st.null, st.null

    tapped = ST.analyse(parse(TAPPED))
    assert tapped.verdict["l"][0] == "gauge"


def test_the_null_basis_is_rotated_until_it_can_be_named():
    """A null direction is a subspace; the SVD's basis for it is arbitrary.

    Here the SVD returns two dense rows, each touching four parameters, and
    neither is the `al*turns^2` gauge -- but their span contains it.  Rotating
    to the sparsest basis recovers it exactly: exponents -2 : +1 in log space.
    """
    st = ST.analyse(parse(TWO_GAUGES))
    assert len(st.null) == 2
    dense = max((np.abs(st.Vt[k]) > 0.15).sum() for k in st.null)
    assert dense >= 3, "the raw SVD rows were already sparse; test is vacuous"

    i, j = st.names.index("al"), st.names.index("turns")
    hit = [v for v in st.nullvecs
           if abs(v[i]) > 0.15 and abs(v[j]) > 0.15
           and np.abs(np.delete(v, [i, j])).max() < 0.05]
    assert len(hit) == 1, "the al/turns gauge was not isolated"
    assert np.isclose(hit[0][i] / hit[0][j], -2.0, atol=0.02)


def test_a_degeneracy_of_size_m_costs_one_coordinate_not_m():
    """Rank 7 of 9 means seven combinations are determined, so seven parameters
    must come back.  Dropping every parameter that appears in a null direction
    returned ONE, because two directions between them touched six names."""
    nl = parse(TWO_GAUGES)
    st = ST.analyse(nl)
    assert st.rank == 7
    coords, notes = ST.suggest_coordinates(st)
    assert len(coords) == 7, coords
    assert "lsh" in coords and "al" not in coords and "turns" not in coords
    # the surviving coordinates really are determinable, not merely proposed
    red, _notes = ST.analyse_reduced(nl)
    assert red.rank == len(red.names) == 7
    assert all(v[0] == "determined" for v in red.verdict.values())
    assert any("gauge choice and not a measurement" in n for n in notes), notes


def test_a_weak_direction_is_not_mistaken_for_a_null_one():
    """The step-size test and the jittered-rank count are independent measures
    of the same number, and they used to be allowed to disagree in silence.

    On this netlist a singular value of 1.1e-3 shrank by 5x over a 100x change
    in step and was called null.  Truncation error falls as O(h^p) with p >= 1,
    so 5x over 100x is p = 0.35 -- a real direction, not an artefact.
    """
    st = ST.analyse(parse(TWO_GAUGES))
    assert not st.rank_dispute, st.rank_dispute
    assert st.rank == int(np.median(st.ranks))
