"""Coupled-resonator synthesis and fitting for an n-pole in-line bandpass filter.

Vendored from the author's earlier helical-filter tuning tool, where it was
developed and
proved.  Kept here rather than replaced: the netlist engine is a better FORWARD
model, but this module's synthesis (gvalues/targets), its model-free measures
(peak_and_bw, k_from_split, qu_from_loose, qu_from_reflection), its calibration
helpers and qu_leverage are all model-agnostic, and it remains the coordinate
system the bench procedure is written in.  It is also the independent second
opinion the netlist engine is checked against -- see tests/test_reconcile.py.

Changed from the original: the coupling/port frequency exponents are no longer
bundled into named pairs.  See COUPLING_LAWS below.

Model (unnormalised, one node per resonator):
    x_i  = f/f0i - f0i/f
    A_ii = 1/Qu + 1j*x_i   (+1/Qe1 on the first, +1/Qen on the last)
    A_i,i+1 = A_i+1,i = -1j*k_i
    S21 = 2/sqrt(Qe1*Qen) * inv(A)[n-1,0]
    S11 = 1 - 2/Qe1 * inv(A)[0,0]
Only |S21| and |S11| are used, so the sign of k is not observable.
"""
import numpy as np
from scipy.optimize import least_squares

# ---------------------------------------------------------------- synthesis
def gvalues(n, ripple_db=0.0):
    """Lowpass prototype g1..gn plus g_{n+1}. ripple_db <= 0 gives Butterworth."""
    if ripple_db <= 0:
        return [2*np.sin((2*k-1)*np.pi/(2*n)) for k in range(1, n+1)], 1.0
    beta = np.log(1/np.tanh(ripple_db/17.37))
    gam = np.sinh(beta/(2*n))
    a = [np.sin((2*k-1)*np.pi/(2*n)) for k in range(1, n+1)]
    b = [gam**2 + np.sin(k*np.pi/n)**2 for k in range(1, n+1)]
    g = [2*a[0]/gam]
    for k in range(2, n+1):
        g.append(4*a[k-2]*a[k-1]/(b[k-2]*g[k-2]))
    gn1 = 1.0 if n % 2 else (1/np.tanh(beta/4))**2
    return g, gn1

def ripple_to_bw3(n, ripple_db):
    """3 dB bandwidth divided by ripple bandwidth."""
    if ripple_db <= 0:
        return 1.0
    eps = np.sqrt(10**(ripple_db/10) - 1)
    return float(np.cosh(np.arccosh(1/eps)/n))

def targets(n, f0_hz, bw_hz, ripple_db=0.0, Qu=None):
    """Design targets. bw_hz is the ripple bandwidth for Chebyshev, 3 dB for Butterworth."""
    g, gn1 = gvalues(n, ripple_db)
    fbw = bw_hz/f0_hz
    k = [fbw/np.sqrt(g[i]*g[i+1]) for i in range(n-1)]
    t = dict(n=n, f0=f0_hz, ripple=ripple_db, fbw=fbw, g=g, gn1=gn1,
             k=k, Qe1=g[0]/fbw, Qen=g[-1]*gn1/fbw,
             bw3=bw_hz*ripple_to_bw3(n, ripple_db),
             split=[ki*f0_hz for ki in k])
    t['sl_bw1'] = f0_hz/t['Qe1']            # singly-loaded 3 dB BW, port 1
    t['sl_bwn'] = f0_hz/t['Qen']
    t['rl'] = (None if ripple_db <= 0 else
               -10*np.log10((10**(ripple_db/10)-1)/(10**(ripple_db/10))))
    if Qu:
        t['il'] = 4.343*sum(g)/(fbw*Qu)
    return t

# ---------------------------------------------------------------- forward model
# How the couplings scale with frequency.  The usual narrowband model treats k
# and Qe as constants, which is exact only at the centre; a real coupling element
# is a reactance and drifts either side of it.  Exponents are (k, Qe):
#   capacitive  k = J/b with J = wCm  ->  k ~ f;  a series input cap transforms
#               50 ohm as 1/(w^2 Cs^2 Z0)          ->  Qe ~ 1/f^2
#   inductive   the duals of those                 ->  k ~ 1/f,  Qe ~ f^2
# Both exponents are PREDICTED by the topology, not fitted.  Measured on a live
# N6ARA TinyFilter over +/-50 MHz, with every other parameter pinned at its
# passband value: constant k gives 6.41 dB rms and a lopsided skirt error
# (-8.0 dB low, +3.8 high); "capacitive" gives 2.49 dB and -1.5/-1.5.  Letting
# the exponents float instead reaches 2.10 dB but runs to the bounds - the
# fixed-exponent model is the better one on parsimony, not on residual.
# UNBUNDLED from the original, which offered only the three named pairs.  The
# two exponents are set by two INDEPENDENT physical choices -- how the
# resonators are coupled, and how the ports are coupled -- and a real filter can
# mix them.  A tapped helical is exactly that case: magnetic coupling between
# parallel resonators gives k ~ 1/f (kexp -1), but the feed is a tap on the
# coil, an autotransformer that is frequency-independent to first order, giving
# Qe ~ constant (qexp 0).  That pair is neither "capacitive" nor "inductive",
# and forcing it into one of them mis-models the skirts on the very filter this
# project was built for.
#
# CONFIDENCE, because this matters: the kexp=-1 half is solid, derived two ways.
# The qexp=0 half assumes a tap behaves as an ideal autotransformer, ignoring
# leakage inductance and the moving voltage distribution on a distributed
# resonator.  It is a prediction, not a measurement -- one wideband helical
# sweep settles it.
#
# Note also that these exponents are for the PARALLEL-resonator / J-inverter
# picture, which is what both filters here are (the TinyFilter is three shunt
# L||C sections; a helical is a shorted quarter-wave line, antiresonant at f0).
# The same physical coupling element gives the OPPOSITE exponent in the
# series-resonator / K-inverter picture, so a crystal ladder would invert these.
COUPLING_LAWS = dict(constant=(0.0, 0.0), capacitive=(1.0, -2.0),
                     inductive=(-1.0, 2.0), helical_tapped=(-1.0, 0.0))


# How a resonator detunes away from f0.  The coupling matrix's diagonal is
# 1/Qu + j*x, and x is the only place the resonator's own nature enters.
#
#   "lumped"   x = f/f0 - f0/f              a parallel L||C
#   "quarter"  x = -(4/pi)*cot(pi*f/(2*f0)) a SHORTED QUARTER-WAVE LINE, i.e. a
#                                           helical or a coax resonator
#
# The two agree to 1.4% inside +-3% of f0 and diverge steadily outside it: at
# 0.7*f0 the line is 11% LESS detuned than the lumped stand-in, and at 1.26*f0
# it is 18% MORE.  That asymmetry is the signature -- a distributed resonator
# puts its low skirt above a lumped fit and its high skirt below, and predicting
# the sign on both sides at once is what makes it a discriminating test rather
# than one more free knob.
RESONATORS = ("lumped", "quarter")


def _detune(f, f0, resonator="lumped"):
    if resonator == "lumped":
        return f/f0 - f0/f
    if resonator == "quarter":
        # cot goes singular at the 3*lambda/4 resonance (f = 3*f0), which is
        # physical: that is where the line resonates again.
        return -(4.0/np.pi)/np.tan(np.pi*f/(2.0*f0))
    raise ValueError(f"unknown resonator {resonator!r}; known: {RESONATORS}")


def response(f, f0s, ks, Qe1, Qen, Qu, coupling="constant", resonator="lumped"):
    """Batched over frequency.  Only the first column of A^-1 is needed
    (S21 wants element n-1, S11 wants element 0), so this solves rather than
    inverting - about 100x faster than the per-point loop it replaces.

    `coupling` names an entry in COUPLING_LAWS, or is an explicit (kexp, qexp)
    pair of exponents.  "constant" is the classic narrowband model; the others
    let k and Qe drift with frequency the way the real elements do.  It changes
    the SKIRTS, barely the passband - inside the 3 dB bandwidth every law agrees
    to well under 0.1 dB.
    """
    n = len(f0s)
    f = np.atleast_1d(np.asarray(f, float))
    m = len(f)
    if isinstance(coupling, str):
        if coupling not in COUPLING_LAWS:
            raise ValueError(f"unknown coupling law {coupling!r}; known: "
                             f"{', '.join(COUPLING_LAWS)} -- or pass (kexp, qexp)")
        kexp, qexp = COUPLING_LAWS[coupling]
    else:
        kexp, qexp = coupling               # an explicit (kexp, qexp) pair
    fr = float(np.mean(f0s))
    qe1 = Qe1*(f/fr)**qexp if qexp else np.full(m, float(Qe1))
    qen = Qen*(f/fr)**qexp if qexp else np.full(m, float(Qen))
    ksc = (f/fr)**kexp if kexp else 1.0
    A = np.zeros((m, n, n), complex)
    for i in range(n):
        A[:, i, i] = 1/Qu + 1j*_detune(f, f0s[i], resonator)
    A[:, 0, 0] += 1/qe1
    A[:, n-1, n-1] += 1/qen
    for i in range(n-1):
        A[:, i, i+1] = A[:, i+1, i] = -1j*ks[i]*ksc
    e0 = np.zeros((m, n, 1), complex); e0[:, 0, 0] = 1.0   # numpy 2 wants a
    col0 = np.linalg.solve(A, e0)[:, :, 0]                 # stack of column vectors
    return 1 - 2/qe1*col0[:, 0], 2/np.sqrt(qe1*qen)*col0[:, n-1]

def db(x):
    return 20*np.log10(np.maximum(np.abs(x), 1e-12))

# ---------------------------------------------------------------- fitting
def fit(f, s11, s21, n, Qu, seed=None, window=None, coupling="constant",
        resonator="lumped"):
    """Fit f0_1..f0_n, k_1..k_{n-1}, Qe1, Qen with Qu HELD FIXED.

    Qu is pinned deliberately: with the ports connected it is poorly
    identifiable, and letting it float lets it absorb real variation in k
    and Qe.  Measure it once in single-resonator mode and pass it in.

    `coupling` is the frequency law for k and Qe -- a name from COUPLING_LAWS
    or an explicit (kexp, qexp) pair.  It is an ASSUMPTION about the topology,
    not something the fit determines, so it is an argument rather than a free
    parameter: letting the exponents float reaches a lower residual by running
    them to the bounds, which is a worse model on parsimony.  Fitting the same
    data under two laws and comparing residuals is how you test the assumption.
    """
    f = np.asarray(f, float)
    if window:
        m = (f >= window[0]) & (f <= window[1])
        f, s11, s21 = f[m], s11[m], s21[m]
    d21, m11 = db(s21), np.abs(s11)
    if seed is None:
        fc = f[np.argmax(np.abs(s21))]
        seed = dict(f0s=[fc]*n, ks=[0.02]*(n-1), Qe1=50.0, Qen=50.0)
    p0 = list(seed['f0s']) + list(seed['ks']) + [seed['Qe1'], seed['Qen']]
    lo = [f.min()]*n + [1e-4]*(n-1) + [2, 2]
    hi = [f.max()]*n + [0.5]*(n-1) + [5000, 5000]
    xs = [1e6]*n + [5e-3]*(n-1) + [20, 20]
    def unpack(p):
        return list(p[:n]), list(p[n:2*n-1]), p[2*n-1], p[2*n]
    def resid(p):
        a, b = response(f, *unpack(p), Qu, coupling=coupling, resonator=resonator)
        return np.concatenate([db(b) - d21, (np.abs(a) - m11)*40])
    r = least_squares(resid, p0, bounds=(lo, hi), x_scale=xs, max_nfev=20000)
    f0s, ks, Qe1, Qen = unpack(r.x)
    a, b = response(f, f0s, ks, Qe1, Qen, Qu, coupling=coupling,
                    resonator=resonator)
    return dict(f0s=f0s, ks=ks, Qe1=Qe1, Qen=Qen, Qu=Qu,
                rms=float(np.sqrt(np.mean((db(b)-d21)**2))), f=f, s11=a, s21=b)

# ---------------------------------------------------------------- direct measures
def peak_and_bw(f, s21, level=3.0):
    """Peak dB, peak frequency, and bandwidth using the OUTERMOST crossings
    (ripple-safe: scanning outward from the peak truncates on a passband dip)."""
    m = db(s21); pk = m.max(); th = pk - level
    a = np.flatnonzero(m >= th)
    if len(a) < 2:
        return pk, f[np.argmax(m)], np.nan
    ip = lambda j, kk: f[j] + (f[kk]-f[j])*(th-m[j])/(m[kk]-m[j])
    lo = ip(a[0]-1, a[0]) if a[0] > 0 else f[0]
    hi = ip(a[-1]+1, a[-1]) if a[-1] < len(m)-1 else f[-1]
    return pk, f[np.argmax(m)], hi-lo

def qu_from_loose(f, s21):
    """Unloaded Q of a single resonator, both probes loose.
    Qu = Q_L / (1 - 10^(A/20)); at -35 dB the probe correction is under 2%."""
    pk, f0, bw = peak_and_bw(f, s21)
    if not np.isfinite(bw) or bw <= 0:
        return None
    QL = f0/bw
    corr = 1 - 10**(pk/20)
    return dict(f0=f0, bw=bw, QL=QL, peak_db=pk, Qu=QL/corr if corr > 0 else np.nan)

def normalise_reflection(f, s11, s21, oob_db=-25.0, deg=2):
    """Correct a reflection *tracking* error using the filter's own stopband.

    Far from its passband a bandpass filter reflects essentially everything, so
    |S11| there should be ~1.  Fitting a smooth envelope to |S11| over the
    stopband and dividing by it removes a frequency-dependent scale error.

    This is a one-term magnitude correction only.  It does not touch directivity,
    source match or the reference plane, so the result is good enough to display
    a return loss and to separate Qe1 from Qe3, but it is not a substitute for a
    proper SOL calibration.  Returns (s11_corrected, envelope) or (s11, None) if
    there is not enough stopband in the sweep.
    """
    d = db(s21)
    m = d < d.max() + oob_db
    if m.sum() < 8:
        return s11, None
    c = np.polyfit(f[m]*1e-6, np.log(np.abs(s11[m])), deg)   # in MHz, so the
    env = np.exp(np.polyval(c, f*1e-6))                      # fit travels
    return s11/env, (c, float(f[m].min()), float(f[m].max()))


def apply_reflection_cal(f, s11, cal):
    """Apply a stored reflection-tracking envelope to another sweep."""
    c, _, _ = cal
    return s11/np.exp(np.polyval(c, f*1e-6))


def s11_unphysical(s11, tol=1.05, frac=0.10):
    """A passive DUT cannot reflect more than it receives."""
    return float(np.mean(np.abs(s11) > tol)) > frac


def bw_edges(f, s21, level):
    """Outermost crossings of (peak - level) dB.  Outermost rather than scanning
    away from the peak, so passband ripple cannot truncate the result.
    `complete` is False when a crossing runs off the end of the sweep, in which
    case lo/hi are clamped to the sweep limits and the width is a lower bound."""
    m = db(s21); pk = m.max(); th = pk - level
    a = np.flatnonzero(m >= th)
    if len(a) < 2:
        return None
    ip = lambda j, kk: f[j] + (f[kk]-f[j])*(th-m[j])/(m[kk]-m[j])
    lo_in, hi_in = a[0] > 0, a[-1] < len(m)-1
    lo = ip(a[0]-1, a[0]) if lo_in else f[0]
    hi = ip(a[-1]+1, a[-1]) if hi_in else f[-1]
    return dict(level=level, lo=lo, hi=hi, bw=hi-lo,
                complete=bool(lo_in and hi_in), lo_in=bool(lo_in), hi_in=bool(hi_in))


def qu_from_reflection(f, s11):
    """Unloaded Q of a single resonator from ONE port - no second connection.

    For a resonator coupled to one port,
        |S11|^2 = (a^2 + x^2)/(b^2 + x^2),  a = 1/Qu - 1/Qe,  b = 1/Qu + 1/Qe = 1/Q_L
    so |S11| dips to |a|/b at resonance and returns to 1 far away.  The points
    where |S11|^2 = (1 + |S11|min^2)/2 sit at x = +/-1/Q_L, hence Q_L = f0/BW,
    and the dip depth gives the coupling factor beta = Qu/Qe:
        |S11|min = |1 - beta| / (1 + beta)
    Then Qu = Q_L (1 + beta) and Qe = Qu/beta.

    beta has two roots - under-coupled (Qe > Qu) and over-coupled (Qe < Qu) -
    and |S11| magnitude alone cannot tell them apart.  With a deliberately loose
    probe you are under-coupled, which is the root returned; `beta_over` carries
    the other one.  The BANDWIDTH is a shape measurement and survives a poor
    reflection calibration; the DEPTH does not, so beta and the split of Qu
    against Qe are only as good as the cal.
    """
    a = np.abs(s11)
    i = int(np.argmin(a))
    if i == 0 or i == len(a)-1:
        return None
    gmin = float(a[i]); f0 = float(f[i])
    th = np.sqrt((1.0 + gmin**2)/2.0)
    lo = hi = None
    for j in range(i, 0, -1):
        if a[j-1] >= th:
            lo = f[j-1] + (f[j]-f[j-1])*(th-a[j-1])/(a[j]-a[j-1]); break
    for j in range(i, len(a)-1):
        if a[j+1] >= th:
            hi = f[j] + (f[j+1]-f[j])*(th-a[j])/(a[j+1]-a[j]); break
    if lo is None or hi is None:
        return None
    bw = hi-lo
    QL = f0/bw
    beta_u = (1-gmin)/(1+gmin)          # under-coupled root
    beta_o = (1+gmin)/max(1e-9, 1-gmin) # over-coupled root
    return dict(f0=f0, gmin=gmin, rl_db=-20*np.log10(max(gmin, 1e-9)), bw=bw, QL=QL,
                beta=beta_u, beta_over=beta_o,
                Qu=QL*(1+beta_u), Qe=QL*(1+beta_u)/max(beta_u, 1e-9),
                Qu_over=QL*(1+beta_o), Qe_over=QL*(1+beta_o)/beta_o)


def k_from_split(f, s21, min_prom=0.5, max_drop=6.0):
    """k from the two-hump split of a coupled pair, or None if there is no split.

        k = (f2^2 - f1^2)/(f2^2 + f1^2)

    This is the one measurement of k that needs no model and no Qu - which makes
    it the anchor when the fit is otherwise free to trade k against Qu.  So it
    has to be right or silent, never confidently wrong.

    The naive version took every local maximum and kept the two highest, which
    on a noisy trace pairs the real peak with a noise ripple tens of MHz away:
    on a filter whose true k is 0.0057 it returned 0.062.  A genuine split has
    two peaks of comparable height with a real dip between them, so require
    prominence, require the pair to be within `max_drop` of the global peak, and
    return None for a single hump rather than inventing a pair.
    """
    from scipy.signal import find_peaks
    m = db(s21)
    idx, props = find_peaks(m, prominence=min_prom)
    if len(idx) < 2:
        return None
    top = m.max()
    idx = np.array([i for i in idx if m[i] >= top - max_drop])
    if len(idx) < 2:
        return None
    pick = idx[np.argsort(m[idx])[-2:]]
    i1, i2 = sorted(pick)
    trough = m[i1:i2+1].min()
    if min(m[i1], m[i2]) - trough < min_prom:      # no real dip between them
        return None
    f1, f2 = float(f[i1]), float(f[i2])
    return dict(f1=f1, f2=f2, split=f2-f1, k=(f2*f2-f1*f1)/(f2*f2+f1*f1),
                f0=float(np.sqrt(f1*f2)),
                depth=float(min(m[i1], m[i2]) - trough))


QU_FIT_POINTS = 200         # per tap sweep in fit_qu_shared

# ------------------------------------------------- coupling from pinned ends
def qu_leverage(Qe1, Qen, Qu):
    """Share of each resonator's loss that the PORTS account for.

    S21 sees Qu only through the total resonator loss 1/Qu + 1/Qe.  With tight
    taps 1/Qe swamps 1/Qu and the two-port measurement is blind to the copper -
    a free Qu in the fit will then run wherever the bounds let it, which looks
    like a result and is not one.  Above ~0.9 treat any fitted Qu as unmeasured
    and use the one-port dip instead.
    """
    Qe = 2.0/(1.0/Qe1 + 1.0/Qen)          # harmonic mean of the two ends
    return (1.0/Qe)/(1.0/Qe + 1.0/Qu)


def fit_pinned_ends(f, s21, n, f0_1, f0_n, Qe1, Qen, Qu, seed=None, band=None):
    """Fit the couplings with the two END resonators pinned at values measured
    one port at a time.

    The ends are exactly where a whole-filter fit is least able to separate Qe
    from Qu, so measuring them from their own reflection dips and holding them
    removes the worst-conditioned part of the problem.  For n = 2 the only free
    parameter left is k12.  For n > 2 the interior resonator frequencies are
    free as well - nothing measures those one port at a time - but they are far
    better conditioned than the end Qe values ever were.

    `band` (Hz) keeps the fit near the peak; see fit_k_pinned for why.
    """
    from scipy.optimize import least_squares
    f = np.asarray(f, float); d = db(s21)
    m = np.ones(len(f), bool)
    if band:
        m = np.abs(f - f[int(np.argmax(d))]) <= band
        if m.sum() < 6*n: m = np.ones(len(f), bool)
    ff, dd = f[m], d[m]
    nk, nmid = n-1, n-2
    seed = seed or {}
    p0 = list(seed.get('ks') or [0.02]*nk) + \
         list(seed.get('f0mid') or [0.5*(f0_1+f0_n)]*nmid)
    lo = [1e-6]*nk + [ff.min()]*nmid
    hi = [2.0]*nk + [ff.max()]*nmid
    xs = [5e-3]*nk + [1e6]*nmid
    p0 = [min(max(v, lo[i]), hi[i]) for i, v in enumerate(p0)]
    def split(p):
        return [abs(x) for x in p[:nk]], [f0_1] + list(p[nk:]) + [f0_n]
    def res(p):
        ks, f0s = split(p)
        _, sm = response(ff, f0s, ks, Qe1, Qen, Qu)
        return db(sm) - dd
    r = least_squares(res, p0, bounds=(lo, hi), x_scale=xs)
    ks, f0s = split(r.x)
    return dict(k=ks[0], ks=ks, f0s=f0s, n_pts=int(m.sum()),
                rms=float(np.sqrt(np.mean(r.fun**2))))


def fit_k_pinned(f, s21, f0s, Qe1, Qen, Qu, band=None, k0=3e-3):
    """Fit ONLY the inter-resonator coupling, with f0 and Qe held at values
    measured some other way - the one-port dip at each end.  Everything else in
    the two-port response is then known, so k is the single free parameter and
    stays well conditioned even where a full six-parameter fit wanders.

    `band` (Hz) keeps the fit within +/- that much of the peak.  Skirts carry
    almost no information about k and, on a real filter, carry structure this
    model has no term for - a stray transmission zero from mixed electric and
    magnetic coupling, or the instrument's own crosstalk floor - so including
    them biases k rather than constraining it.
    """
    from scipy.optimize import least_squares
    f = np.asarray(f, float); d = db(s21)
    m = np.ones(len(f), bool)
    if band:
        m = np.abs(f - f[int(np.argmax(d))]) <= band
        if m.sum() < 12: m = np.ones(len(f), bool)
    ff, dd = f[m], d[m]
    def res(p):
        _, s = response(ff, f0s, [abs(p[0])], Qe1, Qen, Qu)
        return db(s) - dd
    r = least_squares(res, [k0], bounds=([1e-6], [2.0]), x_scale=[5e-4])
    return dict(k=abs(r.x[0]), rms=float(np.sqrt(np.mean(r.fun**2))), n=int(m.sum()))


def compare_roots(f, s21, f0s, cands, n=2, band=None, seed=None):
    """Resolve the one-port depth ambiguity using S21.

    A dip of a given depth fits both an over- and an under-coupled resonator,
    and |S11| alone cannot separate them; the usual tie-breaker is whether the
    locus encircles the origin, which needs a reflection cal good enough to
    trust the phase - a 1 dB dip does not give one.  S21 does: the two roots
    swap Qu and Qe, leaving the resonator loss unchanged but changing the Qe
    product by orders of magnitude.  Fitting k under each therefore predicts
    very different peak levels AND bandwidths, and only one matches.

    `cands` is [(label, Qe1, Qen, Qu), ...] and `f0s` carries the two END
    frequencies.  Returns them sorted best-first with the fitted couplings and
    rms; `decisive` is set when the winner is clear.
    """
    out = []
    for lab, Qe1, Qen, Qu in cands:
        try:
            r = fit_pinned_ends(f, s21, n, f0s[0], f0s[-1], Qe1, Qen, Qu,
                                seed=seed, band=band)
        except Exception:
            continue
        out.append(dict(label=lab, Qe1=Qe1, Qen=Qen, Qu=Qu, **r))
    out.sort(key=lambda d: d['rms'])
    if len(out) > 1:
        out[0]['decisive'] = out[1]['rms'] > 2.0*out[0]['rms'] + 0.5
    return out


def dip_model(f, f0, QL, gmin):
    """|S11| of the single resonator that qu_from_reflection assumes.

        |S11|^2 = (a^2 + x^2)/(b^2 + x^2),   b = 1/Q_L,  |a| = gmin*b,
        x = f/f0 - f0/f

    Both roots give the same MAGNITUDE - they differ only in the sign of a - so
    this is one curve, and it is parameter-free once the dip's centre, width and
    depth are read off.  Drawing it against the measurement is the check that
    the dip really is one resonator: a second resonator, or a neighbour bleeding
    through the coupling, shows up as the trace departing from this shape.
    """
    f = np.asarray(f, float)
    x = f/f0 - f0/f
    b = 1.0/QL
    a = gmin*b
    return np.sqrt((a*a + x*x)/(b*b + x*x))


def fit_s11_ends(f, s11, n, ks, Qu, seed=None, seed_rms=None,
                 coupling="constant", warm_only=False):
    """Both end couplings and every resonator frequency, from S11 alone.

    A one-port sweep of a multi-resonator filter shows every resonator: the near
    one directly, the rest through the couplings.  That is usually treated as a
    nuisance - qu_from_reflection fits ONE Lorentzian to the deepest dip and
    ignores the others - but the extra dips are information, not contamination.
    With the couplings known (captured while aligning) and Qu pinned, the only
    unknowns left are the n frequencies and the two end Qe, and the dips carry
    two positions, two widths and two depths to determine them.

    Two things this buys over reading a single dip:

      * No over/under-coupled ambiguity.  A lone dip's depth fits two values of
        beta and nothing in the magnitude separates them; the pair does.  On a
        46/32 filter, 81 of 100 random starts converged and every one landed on
        the truth - there is no mirror solution.
      * Both ends at once, from one port.  No reversing the filter.

    Weakly identified when the resonators are co-tuned AND the ends are equal:
    the total loading is pinned but its split between the ends goes soft (51/51
    also fits as 46/58).  Detune them slightly to separate the two.

    Parameterised in 1/Qe, not Qe.  With port 2 left OPEN the far tap is
    unterminated and its true Qe is infinite - unreachable in Q-space, so the
    optimiser used to pin it at the 5000 bound and dump the residual into Qe1
    (-3% at loose taps, -38% at tight).  As a conductance that state is 0:
    interior, and evenly scaled against a tight tap.

    Seeded from BOTH the dip positions and their midpoint.  The dips of a
    coupled pair are the split normal modes, not the resonator frequencies -
    two resonators co-tuned at 145.400 put dips at 145.03/145.78.  Seeding
    only from the dips starts 370 kHz off on every frequency and converges to
    a basin that manufactures the split out of a wrong (Qe1, Qen) instead.
    That failure is centred on Qe = 1/k - the target - and reads HIGH:
    +65% at Qe 250, +156% at 172, +294% at 120.  With both seeds tried, error
    is 0% from Qe 25 to 2000, port 2 open or terminated.
    """
    from scipy.optimize import least_squares
    from scipy.signal import find_peaks
    f = np.asarray(f, float); d = db(s11)
    G_MIN, G_MAX = 0.0, 1/3.0        # 1/Qe: 0 = unterminated, 1/3 = absurd
    WARM_CEIL = 0.25                 # dB rms a warm start may never exceed
    lo = [f.min()]*n + [G_MIN, G_MIN]
    hi = [f.max()]*n + [G_MAX, G_MAX]
    xs = [1e6]*n + [5e-3, 5e-3]

    def _q(g):
        return float(1.0/max(g, 1e-9))

    def run(p0):
        p0 = [min(max(v, lo[i]), hi[i]) for i, v in enumerate(p0)]
        def res(p):
            a, _ = response(f, list(p[:n]), ks, _q(p[n]), _q(p[n+1]), Qu,
                            coupling)
            return db(a) - d
        r = least_squares(res, p0, bounds=(lo, hi), x_scale=xs, max_nfev=400)
        return list(r.x), float(np.sqrt(np.mean(r.fun**2)))

    def _out(b, restarted):
        return dict(f0s=b[0][:n], Qe1=_q(b[0][n]), Qen=_q(b[0][n+1]),
                    rms=b[1], restarted=restarted)

    # The dips are the split normal modes.  Seed from them AND from their
    # midpoint (the co-tuned case), and let the residual choose.
    idx, _ = find_peaks(-d, prominence=0.1)
    if len(idx) >= n:
        pick = idx[np.argsort(d[idx])[:n]]
        f0s = sorted(f[pick], reverse=True)
    else:
        c = f[int(np.argmin(d))]
        f0s = [c]*n
    fseeds = [list(f0s)]
    if n == 2:
        # De-embed the split.  Two coupled resonators detuned by d show modes
        # separated by S = sqrt(dt^2 + (k*f0)^2), so the frequencies that put
        # dips where we see them are mid -/+ dt/2, dt = sqrt(S^2-(k*f0)^2).
        # Seeding straight from the dip positions instead starts every f0 half
        # a split off and converges to a basin that fakes the split with a
        # wrong (Qe1, Qen) - +156% at the target, and worse either side.
        # Co-tuned resonators fall out of this as dt = 0, so it subsumes the
        # plain midpoint seed.
        mid = float(np.mean(f0s)); S = abs(f0s[0] - f0s[1])
        kf = abs(ks[0])*mid
        dt = float(np.sqrt(max(S*S - kf*kf, 0.0)))   # NOT `d` - that is db(s11)
        fseeds.append([mid + dt/2, mid - dt/2])
        if S > 0:
            fseeds.append([mid, mid])
    elif n > 1 and max(f0s) - min(f0s) > 0:
        fseeds.append([float(np.mean(f0s))]*n)
    # Which end is which.  The dips are sorted descending, so every seed above
    # puts the HIGHER resonator on port 1.  When the port-1 resonator is
    # actually the lower one the fit would have to swap two frequencies past
    # each other to reach the truth, and it does not - it settles at 713 for a
    # true 500.  Reflection alone cannot tell the order, so try both.
    for fs in list(fseeds):
        rv = list(fs)[::-1]
        if rv != list(fs):
            fseeds.append(rv)
    uniq = []                                 # de-dup: when the resonators are
    for fs in fseeds:                         # co-tuned several seeds coincide,
        if not any(max(abs(a-b) for a, b in zip(fs, u)) < 5e3 for u in uniq):
            uniq.append(list(fs))             # and each costs a grid of solves
    fseeds = uniq
    best = None
    if seed:                                  # warm start from the last sweep
        s = list(seed)
        g1, g2 = 1.0/max(s[n], 1e-9), 1.0/max(s[n+1], 1e-9)
        best = run(list(s[:n]) + [g1, g2])
        # The wrong-basin failure is a FREQUENCY-seed failure, so re-probing the
        # other frequency seeds at the warm conductances is enough to escape it
        # - two more solves, not thirty-two.  Live sweeps stay ~40 ms instead of
        # ~380 ms, which is the difference between tuning and waiting.
        for fs in fseeds:
            cand = run(list(fs) + [g1, g2])
            if cand[1] < best[1]:
                best = cand
        # Accept the warm answer only if it is as good as this filter was
        # ALREADY achieving.  An absolute threshold cannot work: the correct
        # fit sits at 0.00 dB rms on synthetic data and 0.07 on a real noisy
        # sweep, and any constant that tolerates the second also tolerates
        # being stuck in the wrong basin (which costs 0.08-0.29 dB, and read
        # +542% at Qe 80).  Relative to the last good rms, both cases are
        # separated cleanly.
        # Capped, so a bad rms cannot latch in as the new baseline and make
        # every later sweep accept the wrong basin (43% -> 3646% over one run).
        tol = 0.02 if seed_rms is None else min(
            max(3.0*seed_rms, seed_rms + 0.02), WARM_CEIL)
        if best[1] <= tol or warm_only:
            # warm_only callers (the sensitivity bracket) want a bounded cost
            # far more than a perfect fit: they run this several times per
            # sweep, and letting each of them fall through to the cold search
            # turns a 0.1 s frame into a 3.5 s one exactly when the sweeps are
            # already bad.
            return _out(best, False)
    for fs in fseeds:                         # else sweep the conductance plane
        for g1 in (1/30., 1/150., 1/700.):    # port 1 always couples - if it
            for g2 in (1/30., 1/150., 1/700., G_MIN):   # did not there is no dip
                cand = run(list(fs) + [g1, g2])
                if best is None or cand[1] < best[1]:
                    best = cand
    return _out(best, True)


def fit_qu_shared(traces, k, Qen, seed=None):
    """Unloaded Q from the SAME resonator measured at two or three different tap
    settings, with the reflection calibration fitted rather than trusted.

    `traces` is [(f, |s11|), ...] - one sweep per tap position, port 1 on the
    end being moved.  The far end may be terminated or open (pass Qen) and the
    other resonator does NOT have to be detuned.

    Why not just read one dip.  Qu = QL(1+beta) needs beta, which comes from the
    dip DEPTH, which is exactly what a reflection calibration gets wrong.  On a
    single sweep a 3% magnitude error moves Qu by 6% and a 10% error by 20%, and
    nothing in that sweep says which you have.  Moving the tap does not change
    Qu and does not change the calibration, so a family of sweeps over-determines
    both: fitting one shared Qu, one shared cal scale AND slope, and a free Qe
    and f0 per sweep returns Qu to 0% with a 6% cal error present, at any noise
    level from 0.02 to 0.20 dB rms.

    The cal SLOPE has to be in there.  A residual tilt of 2%/MHz - unremarkable
    at VHF on cheap cables - reads as +15% on Qu if it is not modelled, and
    +39% at 5%/MHz.  Fitting it costs nothing when the tilt is really zero.

    Two tap settings are enough; three is better.  Spread them: 3000/1500/800
    works, and so does a narrow 2000/1600/1200.  The taps may pull f0 as they
    move - each sweep gets its own f0, so that is expected, not a problem.

    Seed matters, and it must come from the MEASUREMENT.  From random starts
    only 6 of 80 land on the truth; seeded from the Qe the couple step actually
    read, 20 of 20 do, even with the seed 50% out.  Seeded from the DESIGN Qe
    instead, 0 of 20.  The wrong basins are not silent - they sit at 5.7x the
    rms of the right one - so `rms` is the check, and it is returned.
    """
    from scipy.optimize import least_squares
    n = len(traces)
    if n < 2:
        raise ValueError("need at least two tap settings")
    # Decimate.  The information is in the SHAPE of each dip, not the point
    # count - 200 points per sweep against ~10 parameters is already far
    # over-determined, and the full 601 made the multi-start take 25 s where
    # this takes about two.
    dec = []
    for f, m in traces:
        f = np.asarray(f, float); m = np.asarray(m, float)
        if len(f) > QU_FIT_POINTS:
            i = np.linspace(0, len(f)-1, QU_FIT_POINTS).round().astype(int)
            f, m = f[i], m[i]
        dec.append((f, m))
    traces = dec
    fr = float(np.mean([np.mean(f) for f, _ in traces]))

    def unpack(p):
        return (p[0], list(p[1:1+n]), list(p[1+n:1+2*n]),
                p[1+2*n], p[2+2*n], p[3+2*n])          # Qu, Qe[], f0[], cal, k, tilt

    def res(p):
        qu, qes, f0s, c, kk, tl = unpack(p)
        out = []
        for (f, m), qe, f0 in zip(traces, qes, f0s):
            a, _ = response(f, [f0, fr], [kk], qe, Qen, qu)
            out.append(c*(1.0 + tl*(f - fr)/1e6)*np.abs(a) - m)
        return np.concatenate(out)

    qls = []
    if seed is None:
        qes, f0s = [], []
        for f, m in traces:
            r = qu_from_reflection(f, m.astype(complex))
            qes.append(r['Qe'] if r else 2000.0)
            f0s.append(r['f0'] if r else fr)
            if r:
                qls.append(r['QL'])
        # Qu >= QL always, and with a backed-off tap Qu is only a little above
        # it, so the median loaded Q is the right order for a seed.  (Seeding
        # from the median Qe instead is a different quantity entirely and lands
        # in the wrong basin on loose taps and noisy sweeps.)
        seed = dict(Qu=float(np.median(qls)) if qls else 600.0,
                    Qe=qes, f0=f0s)
    lo = [50.] + [20.]*n + [fr-5e6]*n + [0.7, 1e-5, -0.3]
    hi = [5e4] + [1e6]*n + [fr+5e6]*n + [1.3, 0.5,  0.3]
    xs = [200.] + [500.]*n + [1e5]*n + [.05, 2e-3, .01]

    def run(qu0, qscale):
        p0 = ([qu0] + [q*qscale for q in seed['Qe']] + list(seed['f0'])
              + [1.0, float(k), 0.0])
        p0 = [min(max(v, lo[i]), hi[i]) for i, v in enumerate(p0)]
        return least_squares(res, p0, bounds=(lo, hi), x_scale=xs, max_nfev=4000)

    # The wrong basins sit at ~5.7x the rms of the right one, so a ladder with
    # the residual as referee is safe.  It has to move Qe as well as Qu: the Qe
    # seed comes from each dip's depth, so it is biased by exactly the
    # calibration error this fit exists to remove, and on slack taps that bias
    # is enough to land in a basin 13% high.
    best = None
    for q0 in (seed.get('Qu', 600.), 400., 1200., 3000.):
        for qs in (1.0, 0.4, 2.5):
            try:
                cand = run(q0, qs)
            except Exception:
                continue
            if best is None or cand.cost < best.cost:
                best = cand
    if best is None:
        raise RuntimeError("Qu fit did not converge from any start")
    r = best
    qu, qes, f0s, c, kk, tl = unpack(r.x)
    # How deep the shallowest dip got.  Below about 5 dB (beta < 0.3) the fit
    # starts landing in a basin ~12% high: the dip carries too little shape to
    # separate Qu from the calibration.  So "as loose as possible" is the wrong
    # instinct - there is a floor, and the caller has to be told where it is.
    # max(), not min(): these are negative dB, so the SHALLOWEST dip is the
    # least negative one.  Taking the deepest instead let a set whose worst dip
    # was 3.9 dB through unflagged, and it returned Qu 46845 for a true 780.
    shallow = max(20*np.log10(max(m.min(), 1e-9)) for _, m in traces)
    return dict(Qu=qu, Qe=qes, f0s=f0s, cal=c, k=kk, tilt=tl,
                rms=float(np.sqrt(np.mean(r.fun**2))), n=n,
                shallowest_db=float(shallow),
                too_loose=bool(shallow > -5.0),
                # Running to the rail is not a measurement.  It is what the fit
                # does when the dips carry too little shape to pin Qu at all.
                at_bound=bool(qu > 0.5*hi[0]))

# ---------------------------------------------------------------- law testing
def compare_laws(f, s11, s21, n, Qu, laws=None, seed=None, window=None):
    """Fit the same data under several coupling laws and say which wins -- or
    say that the data cannot tell.

    The second half is the point.  Ranking four residuals always produces a
    winner, and on a narrowband filter that winner is noise: the exponents only
    change the SKIRTS, so if the sweep does not reach far enough down the skirts
    before hitting the fixture's leakage floor, every law predicts nearly the
    same thing over the usable band and the ranking is meaningless.

    So this also runs a POWER CHECK: it takes the parameters fitted under
    `constant`, evaluates every law at those same parameters, and measures how
    far apart the laws actually are over the band being used.  A comparison is
    only worth reading when that separation is large compared with the residual.
    Returns a dict with per-law fits, the separations, and a `powered` verdict.
    """
    laws = laws or ["constant", "capacitive", "inductive", "helical_tapped"]
    if seed is None:
        f0 = f[int(np.argmax(np.abs(s21)))]
        seed = dict(f0s=[f0]*n, ks=[0.02]*(n-1), Qe1=60.0, Qen=60.0)

    fits = {}
    for name in laws:
        try:
            fits[name] = fit(f, s11, s21, n, Qu, seed=seed, window=window,
                             coupling=name)
        except Exception as e:                     # noqa: BLE001
            fits[name] = {"error": str(e)}

    ref = fits.get("constant")
    sep = {}
    if ref and "error" not in ref:
        fr = ref["f"]
        _, b0 = response(fr, ref["f0s"], ref["ks"], ref["Qe1"], ref["Qen"], Qu,
                         coupling="constant")
        for name in laws:
            _, b = response(fr, ref["f0s"], ref["ks"], ref["Qe1"], ref["Qen"], Qu,
                            coupling=name)
            sep[name] = float(np.abs(db(b) - db(b0)).max())

    ok = {k: v for k, v in fits.items() if "error" not in v}
    order = sorted(ok, key=lambda k: ok[k]["rms"])
    resid = ref["rms"] if ref and "error" not in ref else float("nan")
    best = order[0] if order else None
    # A law only counts as chosen if it beats constant-k by more than the
    # comparison can resolve, AND the laws are actually far enough apart to be
    # distinguished over this band.
    powered = bool(best and sep.get(best, 0.0) > 3.0 * resid)
    return dict(fits=fits, order=order, separation=sep, residual=resid,
                best=best if powered else None, powered=powered,
                need_separation=3.0 * resid)


# Retuning exponents, which are NOT the within-sweep exponents in
# COUPLING_LAWS, and confusing the two is a category error.
#
# COUPLING_LAWS says how k and Qe drift with frequency ACROSS one sweep at fixed
# hardware -- that is what shapes the skirts.  What follows is how k and Qe move
# when you RETUNE the filter, i.e. change the resonator capacitance with the
# tuning screw while the coupling element and the taps stay put.  Different
# derivative, different exponent, same underlying topology.
#
# For parallel resonators retuned by C (so C ~ f0^-2):
#
#   fixed coupling cap Cm   k = Cm/C            ~ f0^+2
#   fixed mutual M          k = M/L             ~ f0^0
#   fixed coupling ind. Lm  k = L/Lm            ~ f0^0
#   fixed tap ratio n       Qe = w0*C*Z0/n^2    ~ f0^-1
#   fixed series cap Cs     Qe = C/(w0 Cs^2 Z0) ~ f0^-3
#   fixed series ind. Ls    Qe = w0*C*(w0Ls)^2/Z0 ~ f0^+1
#
# Verified numerically against the netlist engine: a cap-coupled, cap-fed pair
# retunes as (+2.21, -3.26) and a mutual-coupled, tapped pair as (-0.01, -1.02),
# the small excesses in the first being the (w0*Cs*Z0)^2 term and the coupling
# cap's own loading of the node.
#
# name -> (kexp_retune, qexp_retune, implied within-sweep law)
RETUNE_LAWS = {
    "capacitive":     (+2.0, -3.0, "capacitive"),
    "helical_tapped": (0.0, -1.0, "helical_tapped"),
    "inductive":      (0.0, +1.0, "inductive"),
}


def nearest_topology(kexp, qexp):
    """Closest topology to a measured RETUNING exponent pair.

    Returns (name, (kexp, qexp), distance, implied_within_sweep_law).  The last
    is what to hand to `response(..., coupling=...)`: the retuning measurement
    identifies the hardware, and the hardware implies the skirt law.
    """
    best = min(RETUNE_LAWS.items(),
               key=lambda kv: (kv[1][0] - kexp) ** 2 + (kv[1][1] - qexp) ** 2)
    d = float(np.hypot(best[1][0] - kexp, best[1][1] - qexp))
    return best[0], (best[1][0], best[1][1]), d, best[1][2]


def measure_exponents(sweeps, n, Qu, seed=None):
    """Measure the coupling/port frequency exponents directly, by retuning.

    The wideband-skirt fit that `compare_laws` performs is the obvious way to
    get at these exponents and, on a narrowband filter, usually the wrong one:
    the laws differ only far down the skirts, which is exactly where a real
    fixture has nothing but leakage.  On a 2 m helical measured with a NanoVNA
    the separation is ~1 dB against a ~1.2 dB residual, and no amount of
    averaging fixes it -- the instrument's own residual crosstalk is about
    -86 dB and settling it that way needs roughly -100 dB.

    This is the other route, and it has 80 dB of headroom instead of 3.  Retune
    the whole filter to several centre frequencies and measure k and Qe at each,
    in the PASSBAND where the signal is large.  Then the exponents are just the
    slopes of log k and log Qe against log f0:

        k(f0)  ~ f0^kexp        Qe(f0) ~ f0^qexp

    Each individual sweep is fitted under `constant`, which is not an assumption
    being smuggled in: inside one narrow sweep every law agrees to well under
    0.1 dB, so "the local k" is well defined whatever the global law is.  The
    law only shows up in how that local k MOVES between tunings.

    It also separates the two exponents, which the skirt fit confounds:
    `helical_tapped` and `inductive` share kexp = -1 within a sweep and are told
    apart only by qexp; under retuning they differ in qexp by a factor of 100 in
    sensitivity (-1 against +1).

    IMPORTANT: the slopes returned here are RETUNING exponents and must be read
    against RETUNE_LAWS, not COUPLING_LAWS.  See the table above.

    RETUNE ONLY WITH THE TUNING SCREW.  If the aperture or the taps move between
    sweeps, the coupling element itself has changed and the slope measures
    nothing.  That is the whole protocol.

    `sweeps` is a list of (f, s11, s21).  Needs at least 2; 3 or more gives an
    uncertainty on the slope.
    """
    if len(sweeps) < 2:
        raise ValueError("need at least two tunings to measure a slope")
    rows = []
    for f, s11, s21 in sweeps:
        f = np.asarray(f, float)
        sd = seed or dict(f0s=[f[int(np.argmax(np.abs(s21)))]] * n,
                          ks=[0.02] * (n - 1), Qe1=60.0, Qen=60.0)
        r = fit(f, s11, s21, n, Qu, seed=sd)
        rows.append(dict(f0=float(np.mean(r["f0s"])), k=float(np.mean(r["ks"])),
                         qe=float(np.sqrt(r["Qe1"] * r["Qen"])), rms=r["rms"]))
    rows.sort(key=lambda d: d["f0"])
    x = np.log([d["f0"] for d in rows])
    out = dict(points=rows, span=float(np.exp(x.max() - x.min())))
    for key, name in (("k", "kexp"), ("qe", "qexp")):
        y = np.log([d[key] for d in rows])
        slope, icept = np.polyfit(x, y, 1)
        if len(rows) > 2:
            resid = y - (slope * x + icept)
            dof = len(rows) - 2
            sxx = np.sum((x - x.mean()) ** 2)
            se = float(np.sqrt(np.sum(resid ** 2) / dof / sxx)) if sxx > 0 else np.inf
        else:
            se = float("nan")
        out[name] = float(slope)
        out[name + "_se"] = se
    return out


