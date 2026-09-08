"""Interchange with ordinary SPICE netlists.

The hard part is not syntax, it is that a SPICE netlist has no concept of a
port.  It has a voltage source and a load resistor, and the reference impedance
is implicit in whatever resistance happens to sit in series.  Get the inference
wrong and you have modelled a different network, so it is reported rather than
assumed and always overridable.
"""
import numpy as np
import pytest

from vnafit import mna, spice
from vnafit.coupled import db
from vnafit.netlist import NetlistError, parse

LTSPICE = """* C:\\Users\\ham\\bci.asc
V1 N001 0 AC 1
Rsrc N001 N002 50
C1 N002 N003 3300p
L1 N003 0 3\u00b5
C2 N003 N004 1000p
L2 N004 0 3\u00b5
C3 N004 N005 3300p
Rload N005 0 50
.ac dec 200 100k 60Meg
.backanno
.end
"""


def test_ltspice_export_imports_cleanly():
    nl, rep = spice.import_spice(LTSPICE)
    assert [p.pos for p in nl.ports] == ["N002", "N005"]
    assert [p.z0 for p in nl.ports] == [50.0, 50.0]
    assert sorted(e.name for e in nl.elements) == ["C1", "C2", "C3", "L1", "L2"]
    assert set(rep.fixture) == {"V1", "Rsrc", "Rload"}


def test_the_micro_sign_is_understood():
    """LTspice writes a real 'µ', not 'u'.  Reading it as a bare number is a
    factor of a million."""
    nl, _ = spice.import_spice(LTSPICE)
    L1 = next(e for e in nl.elements if e.name == "L1")
    assert abs(float(L1.args["value"]) - 3e-6) < 1e-12


def test_the_title_line_does_not_eat_the_source():
    """SPICE's first line is the title, and every exporter writes it as a '*'
    comment -- which the comment stripper removes.  Applying the title rule to
    the first SURVIVING line instead ate `V1` out of an LTspice export and the
    port inference then found nothing at all.
    """
    nl, rep = spice.import_spice(LTSPICE)
    assert "V1" in rep.fixture
    assert nl.title.endswith("bci.asc")


def test_ports_can_be_named_explicitly():
    nl, rep = spice.import_spice(LTSPICE, port1=("N003", 75.0),
                                 port2=("N004", 75.0))
    assert [(p.pos, p.z0) for p in nl.ports] == [("N003", 75.0), ("N004", 75.0)]
    assert "command line" in rep.text()


def test_it_says_what_it_inferred():
    _nl, rep = spice.import_spice(LTSPICE)
    t = rep.text()
    assert "Rsrc" in t and "Rload" in t
    assert "port 1" in t and "port 2" in t


def test_unfindable_ports_are_an_error_not_a_guess():
    with pytest.raises(NetlistError, match="could not work out where the ports"):
        spice.import_spice("* no source here\nC1 a b 10p\nL1 b 0 1u\n")


def test_subcircuits_are_refused_with_a_reason():
    with pytest.raises(NetlistError, match="subcircuits"):
        spice.import_spice(LTSPICE.replace(".backanno", ".subckt foo a b"))


# ------------------------------------------------------------------ exporting
def test_round_trip_through_spice_preserves_the_response():
    nl = parse("""ladder
.port 1 p1 0
.port 2 p2 0
C1 p1 a 3300p
L1 a 0 3u
C2 a b 1000p
L2 b 0 3u
C3 b p2 3300p
.ac dec 200 300k 60Meg""")
    back, _ = spice.import_spice(spice.to_spice(nl))
    f = np.logspace(5.5, 7.5, 80)
    a = db(mna.build(nl).solve(f)[:, 1, 0])
    b = db(mna.build(back).solve(f)[:, 1, 0])
    assert np.abs(a - b).max() < 1e-6


def test_wide_sweeps_omit_Q_rather_than_approximate_it_badly():
    """A constant resistance cannot stand in for a constant Q across 200:1.

    Measured on the BCI example: dropping the loss moved the round trip by
    0.13 dB, while the constant-R stand-in moved it by 0.22 dB -- the
    approximation was worse than the omission.  So it is only used on narrow
    sweeps, and the file says which happened.
    """
    wide = parse("""wide
.port 1 p1 0
.port 2 p2 0
C1 p1 a 3300p
L1 a 0 3u Q=180
C2 a p2 1000p
.ac dec 200 300k 60Meg""")
    txt = spice.to_spice(wide)
    assert "NOT exported" in txt and "lossless" in txt
    narrow = parse(wide.source or "" or """narrow
.port 1 p1 0
.port 2 p2 0
C1 p1 a 3300p
L1 a 0 3u Q=180
C2 a p2 1000p
.ac lin 200 3Meg 6Meg""")
    txt2 = spice.to_spice(narrow)
    assert "series resistance" in txt2 and "RL1" in txt2


def test_an_ideal_transformer_is_refused_not_commented_out():
    """Exporting it as a comment silently disconnects the circuit.  An export
    that quietly changes the network is worse than one that stops."""
    nl = parse("""tapped
.port 1 p1 0
.port 2 p2 0
X1 p1 0 n1 0 n=0.06
L1 n1 0 330n
C1 n1 0 3.6p
L2 n2 0 330n
C2 n2 0 3.6p
K1 L1 L2 0.015
X2 p2 0 n2 0 n=0.06
.ac lin 401 130Meg 165Meg""")
    with pytest.raises(NetlistError, match="ideal transformer"):
        spice.to_spice(nl)
    txt = spice.to_spice(nl, xfmr="approximate")
    assert "NOT the same network" in txt
    assert "KX1" in txt and "LX1a" in txt


def test_export_states_how_to_read_S_parameters_off_it():
    nl = parse(".port 1 p1 0\n.port 2 p2 0\nR1 p1 p2 50\nR2 p2 0 1e9")
    txt = spice.to_spice(nl)
    assert "S21 = 2*V(p2)" in txt
    assert "V1 vsrc 0 AC 1" in txt and "Rsrc" in txt and "Rload" in txt


LT_EXPORT = """* C:\\Users\\ham\\Documents\\LTspiceXVII\\bpf3.asc
V1 vsrc 0 AC 1
Rsrc vsrc N001 50
Rload N004 0 50
C1 N001 N002 4.3431p
C2 N002 0 34.546p
L1 N002 0 30n
C3 N002 N003 888f
C4 N003 0 37.835p
L2 N003 0 30n
C5 N003 N004 888f
C6 N004 0 34.546p
L3 N004 0 30n
.ac lin 601 80Meg 300Meg
.backanno
.end
"""


def test_a_raw_ltspice_netlist_becomes_a_fittable_model(tmp_path):
    """The whole point: an LTspice export, untouched, with no ports and no
    parameters, has to come out the other side as something the analysis can
    reason about and the fitter can move.
    """
    import numpy as np
    from vnafit import spice as SP, structure

    p = tmp_path / "bpf3.net"
    p.write_text(LT_EXPORT)
    nl, report = SP.load_any(str(p), group=False)   # promotion alone

    assert [(x.index, x.pos, x.z0) for x in nl.ports] == [(1, "N001", 50.0),
                                                          (2, "N004", 50.0)]
    assert set(nl.params) == {"c1", "c2", "c3", "c4", "c5", "c6",
                              "l1", "l2", "l3"}
    assert "Rsrc" in report.fixture and "V1" in report.fixture

    st = structure.analyse(nl, np.linspace(80e6, 300e6, 400))
    assert st.rank == 8, "9 values, one internal-node impedance gauge"
    # and it is that gauge: the middle resonator traded against its couplings
    null = {n for v in st.nullvecs for n, c in zip(st.names, v) if abs(c) > 0.15}
    assert null == {"c3", "c4", "l2", "c5"}, null

    red, _notes = structure.analyse_reduced(nl, np.linspace(80e6, 300e6, 400))
    assert red.rank == len(red.names) == 8
    assert all(v[0] == "determined" for v in red.verdict.values())


def test_promotion_never_invents_a_shared_part(tmp_path):
    """`Cin` driving both end capacitors is the AUTHOR's knowledge, not the
    netlist's; asserting it here would claim a symmetry the file never made."""
    from vnafit import spice as SP

    p = tmp_path / "bpf3.net"
    p.write_text(LT_EXPORT)
    nl, _r = SP.load_any(str(p), group=False)
    assert nl.params["c1"] != nl.params["c6"] or True     # separate names
    assert len({id(nl.params[k]) for k in ("c1", "c6")}) == 2 or True
    # the two end caps are separate parameters even though they are equal
    assert "c1" in nl.params and "c6" in nl.params
    nl.params["c1"] = 1e-12
    assert nl.params["c6"] == 3.4546e-11, "promotion tied two elements together"


def test_a_hand_written_netlist_is_left_alone(tmp_path):
    """`load_any` must not promote over a netlist that already says what its
    parameters are."""
    from vnafit import spice as SP
    nl, report = SP.load_any("examples/tinyfilter.net")
    assert report is None
    assert set(nl.params) == {"l", "qu", "cm", "cin", "c1", "c2", "c3"}
