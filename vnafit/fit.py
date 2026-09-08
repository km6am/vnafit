"""Fitting a netlist to a measured trace.

Three decisions here are not stylistic, and each one is a mistake this project
has already made once.

**Log space.**  Every fitted parameter is a positive physical quantity, so it is
fitted as `ln(value)`.  Positivity comes for free without bounds, which removes
the runs-to-the-rail pathology; the Jacobian becomes scale-free, so a singular
value reads directly as a relative sensitivity; and the degenerate
impedance-scale direction becomes a straight line, which is what makes it
visible to an SVD instead of merely making the fit wander.

**A two-component noise model, propagated into dB.**  A flat dB residual weights
a -80 dB stopband point as heavily as the passband peak, which is badly wrong: a
NanoVNA-H4 bottoms out around -70 dB on fixture leakage, and a point down there
carries several dB of uncertainty while the peak carries a few hundredths.

The first attempt at this was a hard floor, auto-estimated from the trace.  It
was abandoned, and the reason is worth keeping: **you cannot tell a leakage floor
from a shallow skirt by looking at one trace.**  A quantile estimator picked the
deepest transmission ZERO -- real filter behaviour, and often the most
informative part of the sweep.  An out-of-band plateau estimator did better on a
100 MHz helical sweep and then threw away 40% of a narrower one, because there
the sweep simply never reached the floor.  Both were guessing at an INSTRUMENT
property from the data.

So the floor is a property of the measurement, stated once:

    sigma_lin   additive noise, in linear S units (the leakage/noise floor)
    sigma_rel   passband repeatability in dB (cal drift, cable flex)

and the per-point dB uncertainty follows from propagating them:

    sigma_dB(i) = sqrt( (8.686*sigma_lin/|S21_i|)^2 + sigma_rel^2 )

which grows automatically as the trace sinks toward the floor.  No point is
discarded and no threshold is guessed: a point at the floor simply carries ~9 dB
of error and contributes almost nothing.  It also makes chi-squared an actual
chi-squared, which is what lets the profile threshold in `identify.py` mean what
it says.

**The S11 weight is derived, not chosen.**  The older residual concatenated
`db(S21)` with `(|S11| - meas)*40`.  The 40 was a magic number.  Here each
channel is divided by its own noise sigma, so the two are weighted by how well
they are actually known.
"""
import numpy as np
from scipy.optimize import least_squares

from . import mna
from .coupled import db


def noise_sigma(a, b):
    """Trace noise from two repeat sweeps of the same DUT: std(a-b)/sqrt(2).

    This is the only honest source for sigma.  Taking it from a fit residual
    inflates it wherever the MODEL is wrong -- which is exactly the skirts --
    and every confidence interval downstream inherits that inflation.
    """
    a, b = np.asarray(a), np.asarray(b)
    return float(np.std(np.asarray(a) - np.asarray(b)) / np.sqrt(2.0))


# A NanoVNA-H4's transmission floor is fixture leakage, not receiver noise: the
# older tool's IF-bandwidth experiment moved it 3.1 dB where 8x narrower IF plus
# 8x averaging would have moved a noise-limited floor by ~18.  -70 dBc is a fair
# default for a decent cable set; measure your own with the DUT replaced by two
# terminated cables side by side.
DEFAULT_FLOOR_DBC = -70.0
DEFAULT_SIGMA_REL_DB = 0.05


def sigma_db_per_point(s21, sigma_lin, sigma_rel_db=DEFAULT_SIGMA_REL_DB):
    """dB uncertainty of each point, from the additive and relative terms.

    d(20*log10|S|)/d|S| = 8.686/|S|, so an additive error of sigma_lin becomes
    8.686*sigma_lin/|S| dB -- small at the peak, enormous at the floor.  The
    magnitude is clamped at sigma_lin because below that the phase is random and
    the dB value is meaningless anyway.
    """
    mag = np.maximum(np.abs(np.asarray(s21)), sigma_lin)
    return np.sqrt((8.6858896 * sigma_lin / mag) ** 2 + float(sigma_rel_db) ** 2)


def fit_band(f, d21, mult=10.0, min_span_frac=0.02):
    """A frequency window around the measured passband.

    Fitting a 2-pole lumped model across a 100 MHz span is not ambition, it is
    a category error: out there the response is leakage and, for a helical, the
    lumped model is heading toward a 3*lambda/4 resonance it does not have.
    Defaults to peak +/- `mult` times the measured 3 dB bandwidth.
    """
    f = np.asarray(f, float)
    d21 = np.asarray(d21, float)
    i = int(np.argmax(d21))
    inside = np.flatnonzero(d21 >= d21[i] - 3.0)
    bw = (f[inside[-1]] - f[inside[0]]) if len(inside) > 1 else 0.0
    half = max(mult * bw, min_span_frac * f[i])
    return (f[i] - half, f[i] + half)

class Problem:
    """A netlist, a measurement, and the free parameters between them."""

    def __init__(self, netlist, f, s11=None, s21=None, free=None,
                 sigma_db=DEFAULT_SIGMA_REL_DB, sigma_s11=0.01, floor_dbc=None,
                 w_ref=None, band=None):
        self.nl = netlist
        self.f = np.asarray(f, float)
        self.s21 = None if s21 is None else np.asarray(s21, complex)
        self.s11 = None if s11 is None else np.asarray(s11, complex)
        self.sigma_rel_db, self.sigma_s11 = float(sigma_db), float(sigma_s11)
        self.floor_dbc = float(DEFAULT_FLOOR_DBC if floor_dbc is None else floor_dbc)
        self.sigma_lin = 10 ** (self.floor_dbc / 20.0)
        self.w_ref = w_ref or 2 * np.pi * float(np.sqrt(self.f[0] * self.f[-1]))

        specs = {s.param: s for s in netlist.fits}
        self.names = list(free) if free else list(specs)
        if not self.names:
            raise ValueError(
                "nothing to fit: add `.fit <param> <seed> <lo> <hi>` lines to the "
                "netlist, or pass free=[...]")
        base = netlist.resolve_params()
        self.seed = np.array([specs[n].seed if n in specs else base[n]
                              for n in self.names], float)
        self.lo = np.array([specs[n].lo if n in specs else base[n] / 10
                            for n in self.names], float)
        self.hi = np.array([specs[n].hi if n in specs else base[n] * 10
                            for n in self.names], float)
        if np.any(self.seed <= 0):
            raise ValueError("fitted parameters must be positive (they are fitted "
                             "in log space)")

        d21 = db(self.s21) if self.s21 is not None else None
        keep = np.ones(len(self.f), bool)
        if band == "auto":
            band = fit_band(self.f, d21) if d21 is not None else None
        self.band = band
        if band is not None:
            keep &= (self.f >= band[0]) & (self.f <= band[1])
        self.keep = keep
        # What was actually USED, which is the requested window intersected
        # with the sweep -- not the window.  `fit_band` asks for peak +/- 10
        # bandwidths, so on a 10 MHz filter it asks for 200 MHz; against a
        # 125-180 MHz sweep that selects every point, and reporting the request
        # told the user their fit ran over 43-250 MHz of data they never swept.
        self.band_used = (float(self.f[keep].min()), float(self.f[keep].max())) \
            if keep.any() else None
        self.sigma_db = (sigma_db_per_point(self.s21, self.sigma_lin,
                                            self.sigma_rel_db)[keep]
                         if self.s21 is not None else None)
        if keep.sum() < 2 * len(self.names):
            raise ValueError(f"only {keep.sum()} usable points for "
                             f"{len(self.names)} parameters")

    # ------------------------------------------------------------- evaluation
    def model(self, theta):
        """theta is in LOG space."""
        over = dict(zip(self.names, np.exp(theta)))
        b = mna.build(self.nl, over, w_ref=self.w_ref)
        return b.solve(self.f)

    def residual(self, theta):
        S = self.model(theta)
        k = self.keep
        parts = []
        if self.s21 is not None:
            parts.append((db(S[:, 1, 0])[k] - db(self.s21)[k]) / self.sigma_db)
        if self.s11 is not None:
            parts.append((np.abs(S[:, 0, 0])[k] - np.abs(self.s11)[k]) / self.sigma_s11)
        return np.concatenate(parts)

    def rms_db(self, theta):
        S = self.model(theta)
        k = self.keep
        return float(np.sqrt(np.mean((db(S[:, 1, 0])[k] - db(self.s21)[k]) ** 2)))

    # ------------------------------------------------------------------- solve
    RESEED_CHI2_RED = 4.0        # above this, the first landing is suspect

    def _solve_from(self, t0, **kw):
        bounds = (np.log(self.lo), np.log(self.hi))
        t0 = np.clip(np.asarray(t0, float), bounds[0] + 1e-9, bounds[1] - 1e-9)
        return least_squares(self.residual, t0, bounds=bounds, x_scale=1.0,
                             max_nfev=kw.pop("max_nfev", 4000), **kw)

    def run(self, theta0=None, restarts=8, seed=0, **kw):
        """Fit, restarting from a scattered set of seeds if the first fit looks
        poor.

        Not optional polish.  This residual surface has local minima that a
        single descent from the declared seed falls into: on a synthetic 2-pole
        with the truth only ~1.5x away from the seed, the plain fit slid a
        coupling capacitor onto its lower bound and stopped, while a restart
        from a nearby point recovered every parameter to under 1%.  The declared
        seed is a hint, not a starting gun.

        Restarts are skipped entirely once a fit lands at reduced chi2 near 1,
        so the interactive path pays nothing for this.
        """
        best = self._solve_from(np.log(self.seed) if theta0 is None else theta0, **kw)
        if restarts and self._reduced(best) > self.RESEED_CHI2_RED:
            rng = np.random.default_rng(seed)
            lo, hi = np.log(self.lo), np.log(self.hi)
            mid = 0.5 * (lo + hi)
            starts = [mid] + [np.log(self.seed) + rng.normal(scale=0.7, size=len(lo))
                              for _ in range(restarts - 1)]
            for t0 in starts:
                cand = self._solve_from(t0, **kw)
                if cand.cost < best.cost:
                    best = cand
                if self._reduced(best) <= 1.5:
                    break
        return Result(self, best)

    @staticmethod
    def _reduced(res):
        dof = max(len(res.fun) - len(res.x), 1)
        return float(np.sum(res.fun ** 2) / dof)


class Result:
    def __init__(self, prob, res):
        self.prob, self.res = prob, res
        self.theta = res.x
        self.values = dict(zip(prob.names, np.exp(res.x)))
        self.rms = prob.rms_db(res.x) if prob.s21 is not None else float("nan")
        self.at_bound = np.asarray(res.active_mask) != 0
        # Reduced chi-squared: how many sigmas of residual are left over per
        # degree of freedom.  It should be about 1 when the model fits to within
        # the trace noise.  Much more than 1 means the residual is MODEL error,
        # not measurement noise -- and then every covariance-based error bar is
        # too small, because it assumes what is left over is noise.
        dof = max(len(res.fun) - len(res.x), 1)
        self.chi2 = float(np.sum(res.fun ** 2))
        self.chi2_red = self.chi2 / dof

    def __repr__(self):
        vals = ", ".join(f"{k}={v:.6g}" for k, v in self.values.items())
        return f"<fit rms={self.rms:.4f} dB  {vals}>"

    def report(self):
        lines = [f"fit rms {self.rms:.4f} dB over {int(self.prob.keep.sum())} points"]
        used = self.prob.band_used
        if used is not None:
            lines.append(f"  fitted over {used[0]/1e6:.3f}-{used[1]/1e6:.3f} MHz")
            b = self.prob.band
            if b is not None and (b[0] < self.prob.f[0] or b[1] > self.prob.f[-1]):
                lines.append(f"  (the window asked for "
                             f"{b[0]/1e6:.1f}-{b[1]/1e6:.1f} MHz, so the sweep "
                             f"is the limit here, not the window -- every point "
                             f"is in the fit, skirts included)")
        lines.append(f"  noise model: floor {self.prob.floor_dbc:.0f} dBc + "
                     f"{self.prob.sigma_rel_db:.3f} dB relative; a point at the "
                     f"floor carries ~{8.69:.1f} dB of error and barely counts")
        for i, n in enumerate(self.prob.names):
            flag = ("  AT BOUND -- this is the limit you set, not a measurement. "
                    "A parameter that slides to a rail is usually one the data "
                    "cannot see; pin it rather than widening the bound."
                    if self.at_bound[i] else "")
            lines.append(f"  {n:<12} {self.values[n]:.6g}{flag}")
        if self.at_bound.any():
            lines.append("  (a parameter at a bound makes the covariance "
                         "meaningless; identifiability is not reported)")
        if self.chi2_red > 4.0:
            lines.append(
                f"  reduced chi2 {self.chi2_red:.1f}: the residual is "
                f"{np.sqrt(self.chi2_red):.1f}x the assumed noise, so what is left "
                f"over is MODEL error, not noise.  Error bars below are scaled up "
                f"by that factor; they still assume the model shape is right.")
        return "\n".join(lines)
