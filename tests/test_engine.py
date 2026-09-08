"""Engine correctness: invariants first, closed forms second.

The invariants matter more than the closed forms.  A closed-form test proves
the engine is right for one circuit; `S^H S == I` on a random lossless ladder
proves it is right about energy, and catches essentially every stamp sign error
at once.
"""
import numpy as np
import pytest

from vnafit import mna
from vnafit.netlist import NetlistError, parse

FREQ = np.linspace(80e6, 260e6, 61)
WREF = 2 * np.pi * 145e6


def built(text, **kw):
    kw.setdefault("w_ref", WREF)
    return mna.build(parse(text), **kw)


# --------------------------------------------------------------- invariants
LOSSLESS = [
    # a shunt-C / series-L ladder
    """.port 1 a 0
       .port 2 c 0
       L1 a b 100n
       C1 b 0 22p
       L2 b c 100n
       C2 a 0 10p
       C3 c 0 10p
       R9 b 0 1e15""",
    # two magnetically coupled parallel resonators
    """.param L=330n C=3.6p
       .port 1 a 0
       .port 2 b 0
       L1 a 0 {L}
       C1 a 0 {C}
       L2 b 0 {L}
       C2 b 0 {C}
       K1 L1 L2 0.03""",
    # capacitively coupled, three sections
    """.param L=30n C=39.6p
       .port 1 a 0
       .port 2 c 0
       L1 a 0 {L}
       C1 a 0 {C}
       Cm1 a b 1.2p
       L2 b 0 {L}
       C2 b 0 {C}
       Cm2 b c 1.2p
       L3 c 0 {L}
       C3 c 0 {C}""",
    # an ideal transformer in the middle
    """.port 1 a 0
       .port 2 b 0
       L1 a 0 200n
       X1 a 0 b 0 n=2.5
       L2 b 0 50n""",
    # a transmission line
    """.port 1 a 0
       .port 2 b 0
       T1 a 0 b 0 Z0=75 TD=1.1n
       R9 a 0 1e15
       R8 b 0 1e15""",
]


@pytest.mark.parametrize("text", LOSSLESS, ids=range(len(LOSSLESS)))
def test_lossless_is_unitary(text):
    """S^H S == I when nothing dissipates.

    The strongest single check available: unitarity ties every entry of S to
    every other through conservation of energy, so a wrong sign or a
    transposed stamp cannot survive it.  The 1e15 ohm resistors are there only
    to satisfy the DC-path rule; they dissipate nothing at these levels.
    """
    S = built(text).solve(FREQ)
    I = np.eye(S.shape[1])
    err = np.abs(np.conj(S.transpose(0, 2, 1)) @ S - I).max()
    assert err < 1e-9, f"unitarity violated by {err:.2e}"


@pytest.mark.parametrize("text", LOSSLESS, ids=range(len(LOSSLESS)))
def test_matrix_is_complex_symmetric(text):
    """A == A.T (symmetric, NOT Hermitian) for every reciprocal element.

    Equivalent to reciprocity, and it fails loudly on a mis-signed stamp.
    """
    A = built(text).matrix(FREQ)
    assert np.abs(A - A.transpose(0, 2, 1)).max() < 1e-12


@pytest.mark.parametrize("text", LOSSLESS, ids=range(len(LOSSLESS)))
def test_reciprocity(text):
    S = built(text).solve(FREQ)
    assert np.abs(S[:, 0, 1] - S[:, 1, 0]).max() < 1e-12


def test_passivity_with_loss():
    """Adding loss must never make the network gain."""
    S = built(""".port 1 a 0
                 .port 2 b 0
                 L1 a 0 330n Q=180
                 C1 a 0 3.6p
                 L2 b 0 330n Q=180
                 C2 b 0 3.6p
                 K1 L1 L2 0.03""").solve(FREQ)
    ev = np.linalg.eigvalsh(np.conj(S.transpose(0, 2, 1)) @ S)
    assert ev.max() <= 1 + 1e-12
    assert ev.max() < 1.0            # it really does dissipate


def test_loss_lowers_the_peak_and_leaves_the_skirts():
    """Q affects the peak, not the stopband -- the claim the constant-Q
    convention rests on.  If this ever fails, the docstring in mna.py is wrong.

    The ports are coupled in through small series caps rather than strapped
    straight across the resonators.  That is not decoration: with a bare 50 ohm
    across a 330 nH coil the external Q is about 6, the filter is grossly
    overcoupled, and Qu barely shows up at all -- the first version of this test
    asserted a peak drop that the circuit could not produce.
    """
    net = """.port 1 p 0
             .port 2 q 0
             Cs1 p a 1.5p
             L1 a 0 330n {q}
             C1 a 0 3.6p
             Cm a b 0.2p
             L2 b 0 330n {q}
             C2 b 0 3.6p
             Cs2 q b 1.5p"""
    f = np.linspace(120e6, 170e6, 801)
    hi = built(net.format(q="Q=1e9")).solve(f)[:, 1, 0]
    lo = built(net.format(q="Q=200")).solve(f)[:, 1, 0]
    db = lambda x: 20 * np.log10(np.abs(x))
    assert db(hi).max() - db(lo).max() > 0.5          # peak drops
    skirt = np.abs(f - f[np.argmax(np.abs(hi))]) > 15e6
    assert np.abs(db(hi) - db(lo))[skirt].max() < 0.05  # skirts do not move


# -------------------------------------------------------------- closed forms
def test_series_impedance():
    """S21 = 2Z0/(2Z0+Z), S11 = Z/(Z+2Z0) for a series element."""
    for zval, txt in [(50.0, "R1 a b 50"), (None, "L1 a b 100n")]:
        b = built(f".port 1 a 0\n.port 2 b 0\n{txt}\nR8 a 0 1e15\nR9 b 0 1e15")
        f = np.array([145.4e6])
        Z = 50.0 if zval else 1j * 2 * np.pi * f[0] * 100e-9
        S = b.solve(f)[0]
        assert abs(S[1, 0] - 100.0 / (100.0 + Z)) < 1e-9
        assert abs(S[0, 0] - Z / (Z + 100.0)) < 1e-9


def test_shunt_capacitor():
    """S21 = 2/(2 + jwC*Z0) for a shunt C across a thru line."""
    f = np.array([145.4e6])
    C = 10e-12
    S = built(".port 1 a 0\n.port 2 a 0\nC1 a 0 10p\nR9 a 0 1e15").solve(f)[0]
    y = 1j * 2 * np.pi * f[0] * C * 50.0
    assert abs(S[1, 0] - 2.0 / (2.0 + y)) < 1e-9


def test_quarter_wave_line():
    """A quarter-wave 50 ohm line into 50 ohm: S11 = 0, S21 = -j."""
    f0 = 145.4e6
    S = built(f".port 1 a 0\n.port 2 b 0\nT1 a 0 b 0 Z0=50 TD={1/(4*f0):.12g}\n"
              "R8 a 0 1e15\nR9 b 0 1e15").solve(np.array([f0]))[0]
    assert abs(S[0, 0]) < 1e-9
    assert abs(S[1, 0] - (-1j)) < 1e-8


def test_ideal_transformer():
    """An n:1 ideal transformer between equal ports:
    S11 = (n^2-1)/(n^2+1), S21 = 2n/(n^2+1)."""
    n = 2.5
    S = built(f".port 1 a 0\n.port 2 b 0\nX1 a 0 b 0 n={n}").solve(np.array([145e6]))[0]
    assert abs(S[0, 0] - (n * n - 1) / (n * n + 1)) < 1e-9
    assert abs(abs(S[1, 0]) - 2 * n / (n * n + 1)) < 1e-9


def test_series_rlc_notch():
    """A series RLC to ground notches at f0 with a known depth.

    Note the node chain a-m-p-0: R must be IN SERIES with L and C.  Written as
    `R1 a m` it sits in parallel with L1 instead and the closed form below is
    for a different circuit entirely.
    """
    L, C, R = 100e-9, 12e-12, 2.0
    f0 = 1 / (2 * np.pi * np.sqrt(L * C))
    S = built(f".port 1 a 0\n.port 2 a 0\nL1 a m {L}\nR1 m n2 {R}\nC1 n2 0 {C}"
              ).solve(np.array([f0]))[0]
    # at resonance the branch is R; shunt R across a thru -> S21 = 2R/(2R+Z0)
    assert abs(S[1, 0] - 2 * R / (2 * R + 50.0)) < 2e-6


def test_unequal_port_impedances():
    """A bare thru between 50 and 75 ohm ports: S11 = (75-50)/(75+50)."""
    S = built(".port 1 a 0 Z0=50\n.port 2 a 0 Z0=75\nR9 a 0 1e15"
              ).solve(np.array([145e6]))[0]
    assert abs(S[0, 0] - 25.0 / 125.0) < 1e-8
    assert abs(S[1, 1] + 25.0 / 125.0) < 1e-8
    assert abs(S[1, 0] - S[0, 1]) < 1e-12                 # still reciprocal
    assert abs(abs(S[0, 0]) ** 2 + abs(S[1, 0]) ** 2 - 1) < 1e-9   # still lossless


# ------------------------------------------------------------------ refusals
def test_indefinite_coupling_matrix_raises():
    """k12 = k13 = +0.9 with k23 = -0.9 is energy-generating.

    Every pairwise |k| < 1, so a naive check passes it and the engine returns
    |S21| > 1 with no complaint.  Positive-definiteness of Lmat is the real
    condition, and it must raise.
    """
    with pytest.raises(NetlistError, match="positive definite"):
        built(""".port 1 a 0
                 .port 2 c 0
                 L1 a 0 100n
                 L2 b 0 100n
                 L3 c 0 100n
                 R9 b 0 1e15
                 K1 L1 L2 0.9
                 K2 L1 L3 0.9
                 K3 L2 L3 -0.9""")


def test_three_way_coupling_that_is_valid_is_accepted():
    b = built(""".port 1 a 0
                 .port 2 c 0
                 L1 a 0 100n
                 L2 b 0 100n
                 L3 c 0 100n
                 C1 a 0 12p
                 C2 b 0 12p
                 C3 c 0 12p
                 K1 L1 L2 0.05
                 K2 L2 L3 0.05
                 K3 L1 L3 0.004""")
    S = b.solve(FREQ)
    assert np.abs(S).max() <= 1 + 1e-9


def test_floating_node_is_named_not_gminned():
    """A winding coupled to the circuit by K alone, and by nothing else.

    That is a real nullspace at every frequency, and the fix is a named error
    rather than a GMIN conductance -- a 1e-12 entry in a matrix of O(1) entries
    destroys the conditioning the scaling just bought.
    """
    with pytest.raises(NetlistError, match="magnetic coupling is not a connection"):
        built(".port 1 a 0\n.port 2 a 0\nL1 a 0 100n\nL2 b c 100n\n"
              "K1 L1 L2 0.5\nR9 a 0 1e15")


def test_a_capacitor_is_a_connection_at_every_frequency_we_solve_at():
    """This used to raise.  A capacitor is an open circuit at DC, but this is an
    AC solver that refuses f <= 0, and excluding capacitors rejected the crystal
    ladder -- series L-C mesh arms with capacitive shunt coupling have no DC
    path to ground anywhere in the middle of the filter."""
    b = built(".port 1 a 0\n.port 2 a 0\nC1 a b 10p\nL1 b m 100n\nR9 a 0 1e15")
    S = b.solve(np.linspace(100e6, 200e6, 51))
    assert np.all(np.isfinite(S))


def test_dc_is_refused_rather_than_nan():
    b = built(".port 1 a 0\n.port 2 b 0\nL1 a b 100n\nR9 b 0 1e15")
    with pytest.raises(ValueError, match="AC solver"):
        b.solve(np.array([0.0]))


def test_useful_errors():
    with pytest.raises(NetlistError, match="not an inductor"):
        parse(".port 1 a 0\nR1 a 0 50\nK1 R1 R1 0.5")
    with pytest.raises(NetlistError, match="duplicate element"):
        parse(".port 1 a 0\nR1 a 0 50\nR1 a 0 60")
    with pytest.raises(NetlistError, match="ports must be numbered"):
        parse(".port 1 a 0\n.port 3 a 0\nR1 a 0 50")
    with pytest.raises(NetlistError, match="no .port"):
        parse("R1 a 0 50")


# ------------------------------------------------------------- conditioning
def test_conditioning_is_physics_not_units():
    """After normalisation the condition number should track loaded Q, not the
    nH/pF unit mismatch.  A high-Q 2-pole at 145 MHz has Q_L of order 50, so
    anything past 1e8 means the scaling is not doing its job."""
    b = built(""".port 1 a 0
                 .port 2 b 0
                 L1 a 0 330n Q=800
                 C1 a 0 3.6p
                 L2 b 0 330n Q=800
                 C2 b 0 3.6p
                 K1 L1 L2 0.012""")
    f = np.linspace(140e6, 150e6, 401)
    assert b.cond(f).max() < 1e8


def test_first_line_is_the_title_even_when_it_looks_like_an_element():
    """SPICE's rule, followed exactly, and the reason it is followed.

    A cleverer heuristic -- "treat the first line as a title only if it does not
    look like an element" -- failed on the first real file it met.  The title
    "K9DP QRP BCI filter, version 1" starts with K and has several fields, so it
    was read as a mutual-coupling element and the parser complained that "QRP"
    is not an inductor.  The convention is unambiguous; guessing is not.
    """
    nl = parse("K9DP QRP BCI filter, version 1 (332/102, 24 turns)\n"
               ".port 1 a 0\n.port 2 a 0\nR1 a 0 1e15")
    assert nl.title.startswith("K9DP QRP BCI filter")
    assert len(nl.elements) == 1

    # A leading directive is still not a title, so fragments keep working.
    assert parse(".port 1 a 0\n.port 2 a 0\nR1 a 0 1e15").title == ""
