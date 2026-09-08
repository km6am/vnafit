"""The role heuristic: which elements a filter designer buys as one part.

The claim being tested is narrow and checkable -- the heuristic must recover
the SAME parameterisation a person wrote by hand, and must refuse to tie the
one thing a person left separate.
"""
import numpy as np
import pytest

from vnafit import roles, spice as SP, structure
from vnafit.netlist import load

LT = "examples/tinyfilter_ltspice.net"
HAND = "examples/tinyfilter.net"
BAND = np.linspace(125e6, 180e6, 400)


def test_roles_are_read_off_the_ladder():
    nl, _r = SP.load_any(LT, group=False)
    role = roles.classify(nl)
    assert role["Cin"] == role["Cout"] == "end"
    assert role["Cm1"] == role["Cm2"] == "coupling"
    assert all(role[f"L{i}"] == "resonator_l" for i in (1, 2, 3))
    assert all(role[f"RL{i}"] == "loss_r" for i in (1, 2, 3))
    assert all(role[f"C{i}"] == "resonator_c" for i in (1, 2, 3))


def test_the_tuning_capacitors_are_never_tied():
    """C1 and C3 are equal in the design and are the two end TRIMMERS.  Tying
    them -- which merging by equal value does -- makes a detuned end resonator
    undetectable, and on the real unit those resonators sit 1044 kHz apart."""
    nl, _r = SP.load_any(LT, group=False)
    tied = {e.name for _n, els, _w in roles.suggest(nl) for e in els}
    assert not ({"C1", "C2", "C3"} & tied), tied


def _partition(nl):
    """Which elements each parameter drives, ignoring what it is called."""
    out = {}
    for e in nl.elements:
        v = e.args.get("value")
        for n in getattr(v, "names", ()):
            out.setdefault(n, set()).add(e.name)
    return {frozenset(v) for v in out.values()}


def test_the_heuristic_recovers_the_hand_written_parameterisation():
    """The two files are not element-identical -- the hand-written one carries
    loss as `Q=` on the inductor where SPICE needs a separate resistor -- so the
    claim is about the grouping of what they share."""
    auto, _r = SP.load_any(LT)
    hand = load(HAND)
    shared = {frozenset(g) for g in _partition(auto) if not any(
        n.startswith("RL") for n in g)}
    assert shared == _partition(hand), shared

    # and the loss resistors, which the hand-written file expresses as one `qu`
    assert frozenset({"RL1", "RL2", "RL3"}) in _partition(auto)


def test_grouping_turns_an_unfittable_import_into_a_fittable_one():
    """The number that matters: 2 of 12 measurable becomes 7 of 7."""
    raw, _r = SP.load_any(LT, group=False)
    grouped, _r2 = SP.load_any(LT)
    a, _n = structure.analyse_reduced(raw, BAND)
    b, _n2 = structure.analyse_reduced(grouped, BAND)
    assert len(a.measurable()) <= 2
    assert len(b.measurable()) == len(b.names) == 7


def test_a_hand_written_netlist_is_never_regrouped():
    """The heuristic is about a file that never said what its parameters are.
    Applying it over an author's own choices would overrule them."""
    nl, report = SP.load_any(HAND)
    assert report is None
    assert set(nl.params) == {"l", "qu", "cm", "cin", "c1", "c2", "c3"}


def test_equal_values_are_required_not_just_equal_roles():
    """The heuristic decides which ROLES may be tied; it never decides that two
    different values are secretly the same part."""
    nl, _r = SP.load_any(LT, group=False)
    for e in nl.elements:
        if e.name == "Cout":
            nl.params[next(iter(e.args["value"].names))] = 9.9e-12
    names = {n for n, _e, _w in roles.suggest(nl)}
    assert "cin" not in names, "tied two capacitors that are not the same value"
    assert "cm" in names, "the untouched pair should still group"


TOPOLOGIES = {
    "lowpass_cheb5":       {"L1": "series_l", "C2": "shunt_c", "L5": "series_l"},
    "highpass_cheb5":      {"C1": "series_c", "L2": "shunt_l", "C5": "series_c"},
    "elliptic_lowpass":    {"L2": "series_l", "Cb2": "bridge_c", "C1": "shunt_c"},
    "notch_trap":          {"L1": "trap_l", "C1": "trap_c", "Lser": "series_l"},
    "crystal_ladder4":     {"Lm1": "mesh_l", "Cm1": "mesh_c", "Cc1": "shunt_coupling"},
    "bandpass_ind_coupled": {"L1": "resonator_l", "C1": "resonator_c",
                             "Lm1": "coupling_l", "Cin": "end"},
}


@pytest.mark.parametrize("name,expect", sorted(TOPOLOGIES.items()))
def test_each_topology_is_read_correctly(name, expect):
    """Six shapes an RF filter actually comes in, each classified from its
    topology alone -- no names, no hints, raw SPICE."""
    nl, _r = SP.load_any(f"examples/topologies/{name}.net", group=False)
    role = roles.classify(nl)
    for el, want in expect.items():
        assert role[el] == want, f"{name}: {el} read as {role[el]}, not {want}"


@pytest.mark.parametrize("name", sorted(TOPOLOGIES))
def test_no_topology_ties_a_resonator_tuning_capacitor(name):
    """The one rule the whole heuristic rests on.  Tie the element someone
    adjusts and a detuned resonator stops being visible, which is the fault the
    tool exists to find."""
    nl, _r = SP.load_any(f"examples/topologies/{name}.net", group=False)
    role = roles.classify(nl)
    tied = {e.name for _n, els, _w in roles.suggest(nl) for e in els}
    for el, r in role.items():
        if r in roles.NEVER:
            assert el not in tied, f"{name}: tied {el}, which is a {r}"


@pytest.mark.parametrize("name", sorted(TOPOLOGIES))
def test_grouping_never_breaks_a_topology(name):
    """It may not help -- on a symmetric ladder the analysis already finds the
    symmetry on its own -- but it must never cost rank."""
    path = f"examples/topologies/{name}.net"
    raw, _a = SP.load_any(path, group=False)
    grouped, _b = SP.load_any(path)
    f = np.linspace(raw.ac[2], raw.ac[3], 300)
    a, _n = structure.analyse_reduced(raw, f)
    b, _m = structure.analyse_reduced(grouped, f)
    assert len(b.measurable()) == len(b.names), (
        f"{name}: grouping left a coordinate that cannot be measured")
    assert len(b.names) <= len(a.names)


def test_a_crystal_ladder_is_solvable_at_all():
    """Series L-C mesh arms with capacitive shunt coupling have no DC path to
    ground in the middle, which the engine used to refuse outright."""
    from vnafit import mna
    from vnafit.coupled import db
    nl, _r = SP.load_any("examples/topologies/crystal_ladder4.net")
    f = np.linspace(9.97e6, 10.03e6, 601)
    S = mna.build(nl, w_ref=2 * np.pi * 1e7).solve(f)
    u = np.einsum("fij,fik->fjk", S.conj(), S)
    assert np.abs(u - np.eye(2)[None]).max() < 1e-9, "lossless, so S must be unitary"
    d = db(S[:, 1, 0])
    # The passband is ~31 kHz wide inside a 60 kHz window and is not centred in
    # it, so asserting a skirt depth at both edges is asserting the window, not
    # the filter.
    assert d.max() > -0.1, "no passband"
    assert d.max() - d.min() > 30, "no selectivity"
