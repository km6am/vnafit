"""Does the netlist engine agree with the coupling-matrix model where it should,
and disagree where it should?

Agreement alone would prove nothing -- a netlist engine that was secretly a
coupling-matrix model would pass it.  So the suite is in three parts:

  T1  the mapping is right      -- both models agree in the passband
  T2  the topology is SEEN      -- each netlist tracks its OWN predicted
                                   frequency law into the skirts and visibly
                                   fails to track the other one
  T3  the netlist is richer     -- it produces responses the coupling matrix
                                   provably cannot

T2 is the one that can fail informatively.  If a netlist matched BOTH laws
equally well, it would mean the span is too narrow to separate f^+1 from f^-1,
or the residual is dominated by something else -- not that the coupling physics
is the same.  So it also reports how far apart the two topologies actually are.
"""
import numpy as np
import pytest

from vnafit import coupled as M
from vnafit import mna, synth
from vnafit.netlist import parse

F0 = 145.4e6
KS = [0.0121]
QE = 46.0
QU = 670.0


def s21_db(nl, f):
    return M.db(mna.build(nl, w_ref=2 * np.pi * F0).solve(f)[:, 1, 0])


# ------------------------------------------------------------------ T1: mapping
@pytest.mark.parametrize("topo", ["magnetic", "capacitive"])
def test_mapping_recovers_its_own_parameters(topo):
    """Synthesise a netlist from (f0, k, Qe, Qu), then fit the coupling model to
    what the netlist actually does.  If the closed forms in synth.py are right,
    the fit comes back with the numbers we started from.

    This is the test that catches an algebra error in the mapping, and it is
    also the first thing that has ever checked `response()` against an
    independent calculation rather than against itself.
    """
    L = 330e-9 if topo == "magnetic" else 60e-9
    nl = synth.TOPOLOGIES[topo]([F0, F0], KS, QE, QE, QU, L=L)
    f = np.linspace(F0 - 6e6, F0 + 6e6, 601)
    S = mna.build(nl, w_ref=2 * np.pi * F0).solve(f)
    seed = dict(f0s=[F0, F0], ks=list(KS), Qe1=QE, Qen=QE)
    law = synth.PREDICTED_LAW[topo]
    r = M.fit(f, S[:, 0, 0], S[:, 1, 0], 2, QU, seed=seed, coupling=law)

    # Fitting under the law the topology predicts must beat fitting under
    # constant-k, even this close to the centre.  If it does not, either the
    # mapping or the predicted exponent pair is wrong.
    r_const = M.fit(f, S[:, 0, 0], S[:, 1, 0], 2, QU, seed=seed)
    assert r["rms"] <= r_const["rms"] + 1e-9, (
        f"{topo}: its predicted law {law} fits worse than constant-k "
        f"({r['rms']:.3f} vs {r_const['rms']:.3f} dB)")
    assert r["rms"] < 0.25, f"coupling model cannot even fit the netlist ({r['rms']:.3f} dB)"
    assert abs(r["ks"][0] / KS[0] - 1) < 0.06, f"k off by {r['ks'][0]/KS[0]-1:+.1%}"
    assert abs(r["Qe1"] / QE - 1) < 0.10, f"Qe1 off by {r['Qe1']/QE-1:+.1%}"
    for got in r["f0s"]:
        assert abs(got - F0) < 0.01 * F0


@pytest.mark.parametrize("topo", ["magnetic", "capacitive"])
def test_models_agree_across_the_passband(topo):
    """Inside the 3 dB bandwidth the two models must be interchangeable.

    This is the claim that lets the coupling matrix stay in the toolkit at all:
    the netlist is a better model of the skirts, not a different filter.
    """
    L = 330e-9 if topo == "magnetic" else 60e-9
    nl = synth.TOPOLOGIES[topo]([F0, F0], KS, QE, QE, QU, L=L)
    f = np.linspace(F0 - 2e6, F0 + 2e6, 401)
    net = s21_db(nl, f)
    _, cm = M.response(f, [F0, F0], KS, QE, QE, QU,
                       coupling=synth.PREDICTED_LAW[topo])
    # Compare shapes: a constant offset is a mapping-level insertion-loss
    # question, not a shape disagreement, and it is what T1 already checks.
    d = net - M.db(cm)
    assert np.std(d) < 0.25, f"passband shapes differ by {np.std(d):.3f} dB rms"


# ------------------------------------------- T2: the topology must be visible
def _skirt_rms(nl, law, f, floor_db=45.0):
    """rms disagreement over the skirts only, where the laws differ.

    Restricted to points within `floor_db` of the peak.  Below that a real
    measurement is into the instrument's crosstalk floor, so scoring there
    would be scoring noise -- the same mistake the original fit residual makes.
    """
    net = s21_db(nl, f)
    _, cm = M.response(f, [F0, F0], KS, QE, QE, QU, coupling=law)
    cm = M.db(cm)
    peak = net.max()
    band = (net > peak - floor_db) & (np.abs(f - F0) > 4e6)
    return float(np.sqrt(np.mean((net[band] - cm[band]) ** 2)))


@pytest.mark.parametrize("topo", ["magnetic", "capacitive"])
def test_each_netlist_tracks_its_own_law(topo):
    """The discriminating test.

    A capacitively-coupled netlist must follow `capacitive` into the skirts and
    visibly fail to follow `inductive`, and vice versa.  If one netlist matched
    both, the engine would not be seeing the topology and the whole premise of
    this project would be unfounded.
    """
    L = 330e-9 if topo == "magnetic" else 60e-9
    nl = synth.TOPOLOGIES[topo]([F0, F0], KS, QE, QE, QU, L=L)
    f = np.linspace(F0 * 0.6, F0 * 1.4, 1201)

    own = synth.PREDICTED_LAW[topo]
    other = synth.PREDICTED_LAW["capacitive" if topo == "magnetic" else "magnetic"]
    r_own = _skirt_rms(nl, own, f)
    r_other = _skirt_rms(nl, other, f)
    r_const = _skirt_rms(nl, "constant", f)

    assert r_own < r_other, (
        f"{topo}: its own law {own} fits the skirts no better than {other} "
        f"({r_own:.2f} vs {r_other:.2f} dB) -- the engine is not seeing topology")
    assert r_own < r_const, (
        f"{topo}: its own law is no better than constant-k "
        f"({r_own:.2f} vs {r_const:.2f} dB)")


def test_the_two_topologies_are_actually_distinguishable():
    """Power check for the test above, and the one usually skipped.

    If the two topologies' responses differ by less than a real sweep's noise,
    then the discrimination test is passing on a difference nobody could ever
    measure -- and neither could the live experiment that motivated the
    frequency laws in the first place.  Report the separation, and require it to
    be large compared with a NanoVNA's ~0.05 dB trace noise.
    """
    f = np.linspace(F0 * 0.6, F0 * 1.4, 1201)
    a = s21_db(synth.magnetic([F0, F0], KS, QE, QE, QU, L=330e-9), f)
    b = s21_db(synth.capacitive([F0, F0], KS, QE, QE, QU, L=60e-9), f)
    band = (a > a.max() - 45) | (b > b.max() - 45)
    sep = np.abs(a - b)[band].max()
    assert sep > 1.0, (
        f"the two topologies differ by only {sep:.2f} dB over this span; "
        "the discrimination test is not measuring anything")


# -------------------------------------- T3: what the coupling matrix cannot do
def test_cross_coupling_makes_a_transmission_zero_the_coupling_matrix_cannot():
    """A cross-coupling from resonator 1 to resonator 3 opens a second path
    across the filter, and where the two paths cancel there is a finite
    transmission zero.

    A strictly tridiagonal coupling matrix has no entry for that path and cannot
    produce the zero at any parameter value, so this is structure the netlist
    ADDS rather than a parameter it refits.  It is also the case that retires
    `model.py`'s "the sign of k is not observable": that holds for a tridiagonal
    A, where flipping a resonator's basis vector absorbs the sign, and stops
    holding the moment a non-adjacent coupling exists -- here the sign of K9
    decides which side of the passband the notch lands on.
    """
    base = """.port 1 p1 0
.port 2 p2 0
L1 n1 0 330n Q=670
C1 n1 0 3.63p
L2 n2 0 330n Q=670
C2 n2 0 3.63p
L3 n3 0 330n Q=670
C3 n3 0 3.63p
Cm1 n1 n2 0.044p
Cm2 n2 n3 0.044p
X1 p1 0 n1 0 n=0.0602
X2 p2 0 n3 0 n=0.0602
{bridge}"""
    f = np.linspace(130e6, 165e6, 7001)
    def run(bridge):
        return M.db(mna.build(parse(base.format(bridge=bridge)),
                              w_ref=2 * np.pi * F0).solve(f)[:, 1, 0])

    plain = run("")
    up = run("K9 L1 L3 0.001")

    iz = int(np.argmin(up))
    fz = f[iz]
    assert 0 < iz < len(f) - 1, "the notch is at a sweep edge, so it is a skirt"
    assert up[iz] < plain[iz] - 20, (
        f"no transmission zero at {fz/1e6:.2f} MHz: cross-coupled {up[iz]:.1f} dB "
        f"vs plain {plain[iz]:.1f} dB")
    # a zero is a sharp local minimum, not a slope
    shoulder = min(up[max(0, iz - 400)], up[min(len(f) - 1, iz + 400)])
    assert shoulder > up[iz] + 20, "the minimum is not a notch"

    # and the SIGN of the cross-coupling picks the side -- which is exactly what
    # a tridiagonal coupling matrix cannot represent.
    down = run("K9 L1 L3 -0.001")
    assert (f[np.argmin(down)] - F0) * (fz - F0) < 0, (
        "flipping the cross-coupling did not move the zero to the other side")


def test_shorted_line_has_a_spurious_passband_no_lumped_model_shows():
    """A helical resonator is a shorted quarter-wave line, so it resonates again
    at 3*f0.  The lumped L||C stand-in does not, which is the reason a wideband
    helical overlay needs TLIN rather than L and C.
    """
    td = 1.0 / (4 * F0)
    nl = __import__("vnafit.netlist", fromlist=["parse"]).parse(
        f""".port 1 p1 0
.port 2 p2 0
T1 a 0 0 0 Z0=100 TD={td:.12g}
R9 a 0 1e15
C1 p1 a 1p
C2 p2 a 1p""")
    f = np.linspace(0.5 * F0, 4.0 * F0, 4001)
    d = M.db(mna.build(nl, w_ref=2 * np.pi * F0).solve(f)[:, 1, 0])
    # peaks near f0 and near 3*f0
    near = lambda c: d[np.abs(f - c) < 0.08 * F0].max()
    assert near(F0) > d.min() + 20
    assert near(3 * F0) > d.min() + 20, "no 3*f0 resonance -- TLIN is not distributed"


# ----------------------------------------- T5: does the data support ANY law?
def _law_data(topo, f, qu=QU, L=None, noise_db=None, seed=1):
    L = L if L is not None else (330e-9 if topo == "magnetic" else 60e-9)
    nl = synth.TOPOLOGIES[topo]([F0, F0], KS, QE, QE, qu, L=L)
    S = mna.build(nl, w_ref=2 * np.pi * F0).solve(f)
    s11, s21 = S[:, 0, 0], S[:, 1, 0]
    if noise_db is not None:
        rng = np.random.default_rng(seed)
        n = 10 ** (noise_db / 20)
        jit = lambda: n * (rng.normal(size=len(f)) + 1j * rng.normal(size=len(f)))
        s11, s21 = s11 + jit(), s21 + jit()
    return s11, s21


def test_the_verdict_follows_the_criterion_not_the_span():
    """The half that matters, and my first version of this test had it wrong.

    I assumed a narrow sweep is automatically underpowered.  It is not: on
    noiseless data a narrow sweep discriminates perfectly well, because the
    criterion is separation versus RESIDUAL and with no noise and no model error
    the residual is nearly zero.  Span is only a proxy for the thing that
    matters, and a test written around the proxy tests the wrong claim.

    So this asserts the contract itself, across a grid of spans and noise
    levels: the tool names a winner exactly when the laws separate by more than
    three times the residual, and refuses otherwise.  Both outcomes have to
    occur in the grid, or the test proves nothing.
    """
    outcomes = []
    for half in (0.03, 0.10, 0.35):
        for noise_db in (None, -60.0, -40.0):
            f = np.linspace(F0 * (1 - half), F0 * (1 + half), 500)
            s11, s21 = _law_data("capacitive", f, noise_db=noise_db)
            r = M.compare_laws(f, s11, s21, 2, QU)
            best_sep = max((v for k, v in r["separation"].items()
                            if k != "constant"), default=0.0)
            expect = best_sep > r["need_separation"]
            assert r["powered"] == expect, (
                f"span +-{half:.0%} noise {noise_db}: powered={r['powered']} but "
                f"separation {best_sep:.3f} vs need {r['need_separation']:.3f}")
            if not r["powered"]:
                assert r["best"] is None, "refusing but still naming a winner"
            outcomes.append(r["powered"])
    assert any(outcomes), "nothing in the grid was resolvable"
    assert not all(outcomes), "nothing in the grid was refused"


def test_a_wide_clean_sweep_picks_the_right_law():
    """And when it does name a winner, it must be the right one."""
    f = np.linspace(F0 * 0.45, F0 * 2.2, 1400)
    s11, s21 = _law_data("capacitive", f, noise_db=-70.0)
    r = M.compare_laws(f, s11, s21, 2, QU)
    assert r["powered"] and r["best"] == "capacitive", (r["powered"], r["order"])


def test_separation_grows_with_span():
    """The mechanism: the exponents only change the skirts, so discrimination
    is bought with span (or with a lower floor, which is the same thing)."""
    seps = []
    for half in (0.03, 0.15, 0.5):
        f = np.linspace(F0 * (1 - half), F0 * (1 + half), 900)
        s11, s21 = _law_data("capacitive", f)
        res = M.compare_laws(f, s11, s21, 2, QU, laws=["constant", "capacitive"])
        seps.append(res["separation"]["capacitive"])
    assert seps == sorted(seps), f"separation did not grow with span: {seps}"
    assert seps[-1] > 8 * seps[0]
