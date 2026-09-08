"""Fitting, and whether the tool is honest about what it fitted.

The round-trip tests are the easy half.  The half that matters is
`test_the_gauge_parameter_is_reported_as_not_measured`: a fit that returns a
confident number for a parameter the data cannot see is exactly the failure this
project has already paid for, and the tool is supposed to catch it by itself
rather than rely on someone remembering.
"""
import numpy as np
import pytest

from vnafit import fit, identify, mna
from vnafit.netlist import parse

F0 = 145.4e6
WREF = 2 * np.pi * F0

# Capacitively coupled: the series input caps work against the fixed 50 ohm, so
# the impedance level is (weakly) identifiable here.
CAPACITIVE = """.param f0=145.4Meg L=60n cm=0.5p cs=2.2p
.port 1 p1 0
.port 2 p2 0
Cin p1 n1 {cs}
L1 n1 0 {L} Q=670
C1 n1 0 {1/(4*pi*pi*f0*f0*L) - cm - cs}
Cm1 n1 n2 {cm}
L2 n2 0 {L} Q=670
C2 n2 0 {1/(4*pi*pi*f0*f0*L) - cm - cs}
Cout p2 n2 {cs}
.fit cm 0.35p 0.1p 2p
.fit cs 3.0p 0.5p 8p
.fit L  90n  20n  400n
"""

# Tapped and magnetically coupled: the tap is an ideal autotransformer whose
# ratio is written in terms of L, so every impedance scales together and L is an
# EXACT gauge freedom.  k and qe are real; L is not.
TAPPED = """.param f0=145.4Meg L=330n qe=46 k=0.0121 qu=670
.port 1 p1 0
.port 2 p2 0
X1 p1 0 n1 0 n={sqrt(50/(2*pi*f0*L*qe))}
L1 n1 0 {L} Q={qu}
C1 n1 0 {1/(4*pi*pi*f0*f0*L)}
L2 n2 0 {L} Q={qu}
C2 n2 0 {1/(4*pi*pi*f0*f0*L)}
K1 L1 L2 {k}
X2 p2 0 n2 0 n={sqrt(50/(2*pi*f0*L*qe))}
.fit k  0.015 0.002 0.1
.fit qe 60    10    300
.fit L  900n  30n   9u
"""

TRUTH_CAP = {"cm": 0.5e-12, "cs": 2.2e-12, "l": 60e-9}
TRUTH_TAP = {"k": 0.0121, "qe": 46.0, "l": 330e-9}


def synth_measurement(text, truth, f, noise_db=-72.0, seed=3):
    nl = parse(text)
    S = mna.build(nl, truth, w_ref=WREF).solve(f)
    rng = np.random.default_rng(seed)
    n = 10 ** (noise_db / 20)
    jit = lambda: n * (rng.normal(size=len(f)) + 1j * rng.normal(size=len(f)))
    return nl, S[:, 0, 0] + jit(), S[:, 1, 0] + jit()


# ---------------------------------------------------------------- round trips
def test_recovers_known_values():
    f = np.linspace(125e6, 168e6, 401)
    nl, s11, s21 = synth_measurement(CAPACITIVE, TRUTH_CAP, f)
    p = fit.Problem(nl, f, s11, s21, sigma_db=0.05, sigma_s11=0.005)
    r = p.run()
    assert r.rms < 0.2
    for name, want in TRUTH_CAP.items():
        assert abs(r.values[name] / want - 1) < 0.05, (
            f"{name}: got {r.values[name]:.4g}, want {want:.4g}")


def test_null_fit_to_its_own_noiseless_output():
    """Fit the model to exactly what the model produces.

    Catches bounds, seeding and parameterisation bugs, which is the class of
    failure the older tool's notes record hitting repeatedly.
    """
    f = np.linspace(125e6, 168e6, 301)
    nl = parse(CAPACITIVE)
    S = mna.build(nl, TRUTH_CAP, w_ref=WREF).solve(f)
    p = fit.Problem(nl, f, S[:, 0, 0], S[:, 1, 0], sigma_db=0.05, sigma_s11=0.005)
    r = p.run(theta0=np.log([TRUTH_CAP[n] for n in p.names]))
    assert r.rms < 1e-6, f"null fit rms {r.rms:.3g} dB"


# ------------------------------------------------------- the honesty machinery
def test_profile_and_svd_agree_where_the_problem_is_well_conditioned():
    """Two independent estimates of the same interval.

    The profile refits everything else at each node; the SVD is a local
    quadratic from the Jacobian.  Where the problem is near-linear they must
    agree, and agreement is the mutual validation that both are implemented
    right.  Where they DISAGREE, the nonlinearity is real and the profile wins.
    """
    f = np.linspace(125e6, 168e6, 401)
    nl, s11, s21 = synth_measurement(CAPACITIVE, TRUTH_CAP, f)
    p = fit.Problem(nl, f, s11, s21, sigma_db=0.05, sigma_s11=0.005)
    r = p.run()
    rep = identify.svd_report(p, r)
    for name in p.names:
        pr = identify.profile(p, name, r, npts=17)
        svd_spread = float(np.exp(1.96 * rep.sd[p.names.index(name)]))
        assert abs(pr.verdict()[1] / svd_spread - 1) < 0.15, (
            f"{name}: profile says x/{pr.verdict()[1]:.3f}, "
            f"SVD says x/{svd_spread:.3f}")


def test_the_gauge_parameter_is_reported_as_not_measured():
    """The headline case.

    In the tapped topology L is an exact gauge freedom: the fit will happily
    return a number for it, and that number is the seed.  The tool must say so
    -- through the profile refusing to leave the confidence band, and through
    the SVD flagging a degenerate direction -- WITHOUT anyone having to know in
    advance that this is the tapped case.

    Meanwhile k and qe, which are real, must still come back identified.  A tool
    that called everything unmeasured would pass a one-sided version of this.
    """
    f = np.linspace(125e6, 168e6, 401)
    nl, s11, s21 = synth_measurement(TAPPED, TRUTH_TAP, f)
    p = fit.Problem(nl, f, s11, s21, sigma_db=0.05, sigma_s11=0.005)
    r = p.run()
    assert r.rms < 0.25, f"the tapped model should still fit the data ({r.rms:.3f} dB)"

    pr_l = identify.profile(p, "l", r, span=np.log(20.0), npts=15)
    assert pr_l.verdict()[0] == "NOT MEASURED", (
        f"L is an exact gauge freedom here but the profile called it "
        f"{pr_l.verdict()[0]} (x/{pr_l.verdict()[1]:.3f})")
    assert "assumption, not a result" in pr_l.report()

    for name, want in (("k", 0.0121), ("qe", 46.0)):
        pr = identify.profile(p, name, r, npts=15)
        assert pr.verdict()[0] != "NOT MEASURED", f"{name} should be identified"
        assert abs(r.values[name] / want - 1) < 0.06

    rep = identify.svd_report(p, r)
    assert rep.cond > rep.COND_DEGENERATE, (
        f"condition number {rep.cond:.3g} does not flag the gauge direction")
    assert "degenerate" in rep.report()
    # and the soft direction really is the one carrying L
    assert abs(rep.v_min[p.names.index("l")]) > 0.3, (
        f"the softest direction does not point along L: {rep.v_min}")


def test_covariance_is_refused_at_a_bound():
    """A parameter pinned at a bound has no meaningful uncertainty, so the tool
    must refuse rather than quote the curvature of a wall."""
    f = np.linspace(125e6, 168e6, 201)
    nl, s11, s21 = synth_measurement(CAPACITIVE, TRUTH_CAP, f)
    # bounds that exclude the truth, so the fit must end up on a rail
    for spec in nl.fits:
        if spec.param == "cm":
            spec.lo, spec.hi, spec.seed = 1.2e-12, 2.0e-12, 1.5e-12
    p = fit.Problem(nl, f, s11, s21)
    r = p.run()
    assert r.at_bound.any()
    assert "AT BOUND" in r.report()
    with pytest.raises(ValueError, match="at a bound"):
        identify.svd_report(p, r)


# ------------------------------------------------------------ residual hygiene
def test_deep_points_are_de_weighted_rather_than_guessed_away():
    """The noise model, and the reason there is no floor heuristic.

    A flat dB residual weights a -80 dB point like the passband peak.  The first
    fix here was an auto-estimated hard floor, and it failed twice in ways worth
    keeping on the record: a quantile estimator picked the deepest transmission
    ZERO (real filter behaviour, and the most informative part of the trace),
    and an out-of-band-plateau estimator threw away 40% of a narrower sweep that
    simply never reached the floor.  Both were guessing an INSTRUMENT property
    from the data.

    Propagating the additive noise into dB does the job without a threshold: a
    point at the floor carries ~9 dB of error and contributes almost nothing.
    """
    s21 = np.array([1.0, 10 ** (-20 / 20), 10 ** (-40 / 20),
                    10 ** (-70 / 20), 10 ** (-90 / 20)], complex)
    sig = fit.sigma_db_per_point(s21, 10 ** (fit.DEFAULT_FLOOR_DBC / 20), 0.05)
    assert sig[0] < 0.06, "the peak should be known to the relative term"
    assert sig[1] < 0.06
    assert 0.2 < sig[2] < 0.5, "a -40 dB point should be a few tenths of a dB"
    assert 8.0 < sig[3] < 10.0, "a point AT the floor should be ~9 dB uncertain"
    assert sig[4] <= sig[3] * 1.001, "below the floor it cannot get worse"
    assert np.all(np.diff(sig) > 0) or sig[4] == sig[3]


def test_the_floor_is_stated_not_inferred():
    f = np.linspace(125e6, 168e6, 301)
    nl, s11, s21 = synth_measurement(CAPACITIVE, TRUTH_CAP, f)
    loud = fit.Problem(nl, f, s11, s21, floor_dbc=-45.0)
    quiet = fit.Problem(nl, f, s11, s21, floor_dbc=-85.0)
    assert loud.sigma_db.max() > quiet.sigma_db.max(), (
        "a worse stated floor must widen the uncertainty on deep points")
    assert "noise model" in loud.run().report()


def test_noise_sigma_from_repeat_sweeps():
    rng = np.random.default_rng(0)
    true_sigma = 0.037
    base = rng.normal(size=4000)
    a = base + rng.normal(scale=true_sigma, size=4000)
    b = base + rng.normal(scale=true_sigma, size=4000)
    assert abs(fit.noise_sigma(a, b) / true_sigma - 1) < 0.05


def test_fit_refuses_when_nothing_is_declared_free():
    f = np.linspace(125e6, 168e6, 101)
    nl = parse(CAPACITIVE.split(".fit")[0])
    S = mna.build(nl, TRUTH_CAP, w_ref=WREF).solve(f)
    with pytest.raises(ValueError, match="nothing to fit"):
        fit.Problem(nl, f, S[:, 0, 0], S[:, 1, 0])


def test_the_reported_band_is_the_one_actually_used(tmp_path):
    """`fit_band` asks for peak +/- 10 bandwidths.  On a 10 MHz filter that is a
    200 MHz request, and against a 125-180 MHz sweep it selects every point --
    so reporting the REQUEST told the user their fit ran over 43-250 MHz of data
    they never swept.
    """
    import numpy as np
    from vnafit import fit as F, mna
    from vnafit.netlist import load

    nl = load("examples/tinyfilter.net")
    f = np.linspace(125e6, 180e6, 401)
    S = mna.build(nl, w_ref=2 * np.pi * 146e6).solve(f)
    prob = F.Problem(nl, f, s11=S[:, 0, 0], s21=S[:, 1, 0], free=["cm"],
                     band="auto")

    assert prob.keep.all(), "this test needs a window wider than the sweep"
    assert prob.band[0] < f[0] and prob.band[1] > f[-1]
    assert prob.band_used == (f[0], f[-1])

    text = prob.run(restarts=0).report()
    assert "125.000-180.000 MHz" in text, text
    assert "the sweep is the limit here" in text
