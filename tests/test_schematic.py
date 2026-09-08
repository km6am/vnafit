"""Schematic layout.

The placement is tested, not the pixels.  `plan()` decides what is a spine
element, what is a rung, what is a bridge and what could not be placed at all,
and those are the decisions that make a drawing right or wrong; whether a
capacitor's plates are 0.13 units apart is not worth a test.
"""
import pytest

from vnafit import schematic as SCH
from vnafit.netlist import parse

LADDER = """5-element high-pass
.port 1 p1 0
.port 2 p2 0
C1 p1 a 3300p
L1 a 0 3u
C2 a b 1000p
L2 b 0 3u
C3 b p2 3300p"""

TAPPED = """tapped, magnetically coupled
.port 1 p1 0
.port 2 p2 0
X1 p1 0 n1 0 n=0.06
L1 n1 0 330n
C1 n1 0 3.6p
L2 n2 0 330n
C2 n2 0 3.6p
K1 L1 L2 0.015
X2 p2 0 n2 0 n=0.06"""

CROSS = """3-pole with a bridge
.port 1 p1 0
.port 2 p2 0
X1 p1 0 n1 0 n=0.06
L1 n1 0 330n
C1 n1 0 3.6p
Cm1 n1 n2 44f
L2 n2 0 330n
C2 n2 0 3.6p
Cm2 n2 n3 44f
L3 n3 0 330n
C3 n3 0 3.6p
Cb n1 n3 20f
X2 p2 0 n3 0 n=0.06"""


def names(pairs):
    return sorted(e.name for e, *_ in pairs)


def test_a_plain_ladder_lays_out_as_one():
    lay = SCH.plan(parse(LADDER))
    assert lay.spine == ["p1", "a", "b", "p2"]
    assert names(lay.series) == ["C1", "C2", "C3"]
    assert names(lay.shunt) == ["L1", "L2"]
    assert lay.bridges == [] and lay.leftover == []


def test_couplings_connect_the_ladder():
    """Two magnetically coupled resonators sit side by side on the signal path
    even though no wire joins them.  Without treating K as a connection there is
    no ungrounded route from port to port at all, and the entire circuit lands
    in the leftover row -- which is what the first version did.
    """
    lay = SCH.plan(parse(TAPPED))
    assert lay.spine == ["p1", "n1", "n2", "p2"]
    assert names(lay.series) == ["X1", "X2"]
    assert names(lay.shunt) == ["C1", "C2", "L1", "L2"]
    assert lay.leftover == []
    assert len(lay.couplings) == 1
    assert 1 in lay.gaps, "the K link carries no in-line component"


def test_the_spine_is_the_longest_path_not_the_shortest():
    """A bridging cross-coupling is a shortcut across the resonators, so the
    shortest path skips the middle one and dumps it in the leftover row.  The
    signal path goes through every resonator, so the longest path is right.
    """
    lay = SCH.plan(parse(CROSS))
    assert lay.spine == ["p1", "n1", "n2", "n3", "p2"]
    assert names(lay.bridges) == ["Cb"]
    assert names(lay.shunt) == ["C1", "C2", "C3", "L1", "L2", "L3"]
    assert lay.leftover == []


def test_nothing_is_ever_silently_omitted():
    """An element the ladder cannot place must still be reported."""
    nl = parse(LADDER + "\nR9 x y 1e9\nR8 x 0 1e9")
    lay = SCH.plan(nl)
    assert sorted(e.name for e in lay.leftover) == ["R8", "R9"]
    total = (len(lay.series) + len(lay.shunt) + len(lay.bridges)
             + len(lay.leftover))
    assert total == len([e for e in nl.elements if e.kind != "K"])


def test_two_ports_are_required():
    with pytest.raises(ValueError, match="two ports"):
        SCH.plan(parse(".port 1 a 0\nR1 a 0 50"))


def test_engineering_format():
    assert SCH.eng(3.3e-9, "F") == "3.3 nF"
    assert SCH.eng(3.0e-6, "H") == "3 µH"
    assert SCH.eng(2e-14, "F") == "20 fF"
    assert SCH.eng(50.0, "Ω") == "50 Ω"
    assert SCH.eng(0, "F") == "0F"
    assert "?" in SCH.eng(float("nan"), "F")


def test_it_actually_renders(tmp_path):
    out = tmp_path / "s.png"
    for text in (LADDER, TAPPED, CROSS):
        lay = SCH.save(parse(text), str(out))
        assert out.exists() and out.stat().st_size > 4000
        assert lay.spine[0] == "p1" and lay.spine[-1] == "p2"


def test_an_outline_encloses_its_own_element_and_nothing_else():
    """The rule the boxes kept breaking: an outline must contain the whole of
    the element it names, and no part of any other element.

    The first version drew one rectangle round the bounding box of every
    element a parameter touched.  On a ladder those are almost never adjacent
    -- `cin` drives the two END capacitors -- so its rectangle swallowed the
    nine elements in between.  The second cut the ground symbol in half,
    because the box stopped where the ground glyph started.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from vnafit.netlist import load
    from vnafit import schematic

    import glob
    from vnafit import spice as SP
    for path in sorted(glob.glob("examples/*.net")):
        # load_any, not load: examples/ now holds a raw SPICE netlist too, and
        # that is exactly the shape this rule most needs to hold for.
        fig, _ax, lay = schematic.draw(SP.load_any(path)[0])
        names = sorted(lay.boxes)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                if a in lay.shapes or b in lay.shapes:
                    continue          # an arc is outlined as an arc
                x0, y0, x1, y1 = lay.boxes[a]
                a0, b0, a1, b1 = lay.boxes[b]
                over = (min(x1, a1) - max(x0, a0), min(y1, b1) - max(y0, b0))
                assert not (over[0] > 1e-9 and over[1] > 1e-9), (
                    f"{path}: {a} and {b} outlines overlap by {over}")
        plt.close(fig)


def test_a_shunt_outline_clears_the_ground_symbol_and_the_spine():
    """sym_ground reaches 0.21 below the leg's end, and the spine wire runs at
    y0; an outline that stops at either lands a line across a drawn glyph."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from vnafit.netlist import load
    from vnafit import schematic

    fig, ax, lay = schematic.draw(load("examples/tinyfilter.net"))
    # Derived, not a magic number: the ground sits at the bottom of the leg and
    # sym_ground reaches 0.21 below it, so the outline has to reach further.
    ground = min(b[1] for b in lay.boxes.values()) + 0.25
    x0, y0, x1, y1 = lay.boxes["L1"]
    assert y0 <= ground - 0.21, "the outline cuts through the ground symbol"
    assert y1 < 0.0, "the outline runs along the spine wire"
    plt.close(fig)


def test_a_coupling_is_outlined_as_an_arc_not_a_box():
    """A rectangle spanning two coupled inductors necessarily contains whatever
    is drawn between them."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from vnafit.netlist import load
    from vnafit import schematic

    fig, ax, lay = schematic.draw(load("examples/helical_2m.net"))
    ks = [n for n in lay.boxes if n.upper().startswith("K")]
    assert ks, "this example is supposed to have a coupling in it"
    for k in ks:
        assert lay.shapes.get(k, (None,))[0] == "arc", k
    plt.close(fig)


CHAINED = """a lossy inductor the way LTspice writes one
.port 1 p1 0 Z0=50
.port 2 p2 0 Z0=50
Cin p1 n1 4.3431p
L1 n1 l1q 30n
RL1 l1q 0 0.139
C1 n1 0 34.546p
Cm1 n1 n2 0.8877p
L2 n2 l2q 30n
RL2 l2q 0 0.139
C2 n2 0 37.835p
Cout p2 n2 4.3431p
.ac lin 401 125Meg 180Meg
"""


def test_a_shunt_leg_may_be_a_chain_and_all_of_it_gets_drawn():
    """A leg to ground can hold more than one element: LTspice writes a lossy
    inductor as an inductor to a private node and a resistor from there down.

    Matching single elements only put every inductor and every loss resistor
    into `leftover`, where nothing draws them -- six of thirteen components
    missing from the picture, and no outline for the parameters driving them.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from vnafit.netlist import parse
    from vnafit import schematic

    nl = parse(CHAINED)
    lay = schematic.plan(nl)
    assert not lay.leftover, [e.name for e in lay.leftover]
    chains = {tuple(e.name for e in ch) for _i, ch in lay.legs}
    assert ("L1", "RL1") in chains and ("L2", "RL2") in chains, chains

    fig, _ax, lay = schematic.draw(nl)
    assert set(lay.boxes) == {e.name for e in nl.elements}, \
        "an element with no box cannot be outlined or clicked"
    # stacked, not overlapping: the chain members must be separable by eye
    l1, rl1 = lay.boxes["L1"], lay.boxes["RL1"]
    assert l1[1] > rl1[3], "L1 and RL1 outlines overlap"
    plt.close(fig)


def test_two_arms_in_one_slot_are_not_drawn_on_top_of_each_other():
    """An elliptic section puts a capacitor ACROSS a series inductor.  Both are
    series arms between the same pair of nodes, and drawn on the spine they
    landed on each other -- symbols and labels both, "340 nH" and "33 pF"
    overprinted into a smear.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from vnafit import schematic, spice as SP

    nl, _r = SP.load_any("examples/topologies/elliptic_lowpass.net", group=False)
    fig, _ax, lay = schematic.draw(nl)
    l2, cb2 = lay.boxes["L2"], lay.boxes["Cb2"]
    assert cb2[1] > l2[3], "the parallel arm is not lifted clear of the spine"
    over = (min(l2[2], cb2[2]) - max(l2[0], cb2[0]),
            min(l2[3], cb2[3]) - max(l2[1], cb2[1]))
    assert not (over[0] > 0 and over[1] > 0), over
    plt.close(fig)


def test_a_one_element_leg_gets_a_box_the_size_of_its_element():
    """A leg's outline used to stretch down to the common ground line, so a
    plain shunt capacitor beside a two-element leg got a box as tall as the
    whole leg -- taller than its neighbour, and it looked like it enclosed it.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from vnafit import schematic, spice as SP

    nl, _r = SP.load_any("examples/tinyfilter_ltspice.net", group=False)
    fig, _ax, lay = schematic.draw(nl)
    h = lambda n: lay.boxes[n][3] - lay.boxes[n][1]
    assert h("C1") < 1.4 * h("L1"), "the single-element leg's box is stretched"
    plt.close(fig)


def test_every_port_shows_what_its_50_ohm_is_measured_against():
    """A port is a PAIR of terminals and the drawing showed one.

    P1 and P2 sat on the spine as bare circles, so the return -- ground in every
    ordinary measurement -- was absent from a schematic otherwise full of ground
    symbols, and the circuit did not look closed.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import glob
    from vnafit import schematic, spice as SP

    for path in sorted(glob.glob("examples/*.net")
                       + glob.glob("examples/topologies/*.net")):
        nl, _r = SP.load_any(path)
        fig, _ax, lay = schematic.draw(nl)
        legs = len(lay.legs)
        assert len(lay.grounds) == legs + 2, (
            f"{path}: {len(lay.grounds)} grounds for {legs} legs and 2 ports")
        for port, x in zip(nl.ports, (lay.grounds[-2][0], lay.grounds[-1][0])):
            assert port.neg in schematic.GND
        plt.close(fig)
