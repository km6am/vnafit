"""Did the data determine that parameter, or did the seed?

Two tools, deliberately both, because they fail differently:

  `profile()`   scans one parameter and refits everything else.  Global along
                one axis, slow, and it is the one that tells the truth when the
                problem is nonlinear.

  `svd_report()`  the local quadratic picture from the Jacobian at the optimum.
                Cheap, and it names WHICH parameters trade against which --
                something a profile cannot do.

Run both.  They disagree exactly when the nonlinearity matters, and that
disagreement is itself the diagnostic.

The number that makes this usable is small and easy to miss:

    for m residual points, the 95% bound is dchi2 = 3.84, so
        rms(v) >= rms_min * sqrt(1 + 3.84/m)
    at m = 400 that is a rise of 0.5 PER CENT.

So "the profile looks flat" is not a test -- a 0.5% rise is invisible by eye and
is the entire confidence interval.  Everything here draws that line explicitly.

The motivating case is on the record: a Qu scan in the older tool gave an
identical 0.3807 dB rms from Qu = 800 to Qu = 8000, and a fitted Qu that appeared
to move when a shield was fitted turned out to be collinear with the coupling
constant.  A tool that had drawn this line would have said so by itself.
"""
import numpy as np

from .fit import Problem

CHI2_95 = 3.84          # one degree of freedom, 95%


def threshold_factor(m, dchi2=CHI2_95):
    """How much rms must rise, as a multiple, to leave the confidence region."""
    return float(np.sqrt(1.0 + dchi2 / max(m, 1)))


def profile(prob, name, result=None, span=None, npts=21, dchi2=None):
    """rms vs one parameter, everything else refitted at each point.

    Walks OUTWARD from the optimum in both directions, warm-starting each step
    from the previous one.  Cold-starting every node produces a ragged curve for
    optimiser reasons, and that raggedness reads as structure when it is nothing
    of the kind.

    `dchi2` defaults to 3.84 scaled by the reduced chi-squared of the fit, for
    the same reason `svd_report` scales sigma: on real bench data the residual is
    several times the trace noise because the MODEL is imperfect, and an
    unscaled threshold then reports an interval several times too tight.  Both
    tools must apply the same correction or comparing them is meaningless -- and
    comparing them is the point of having both.

    `span` is in natural-log units: ln(10) is one decade each way.  Left as None
    it is chosen from the local Jacobian, which matters more than it sounds: a
    fixed decade-wide grid on a parameter known to 2% puts the entire confidence
    interval inside one grid cell, and the profile then reports an interval of
    zero width -- precise-looking and meaningless.  The span is set to a few
    standard deviations so the crossing is actually resolved.
    """
    if not isinstance(prob, Problem):
        raise TypeError("profile() needs a fit.Problem")
    if name not in prob.names:
        raise ValueError(f"{name!r} is not a free parameter ({prob.names})")
    res = result or prob.run()
    if dchi2 is None:
        dchi2 = CHI2_95 * max(res.chi2_red, 1.0)
    idx = prob.names.index(name)
    others = [i for i in range(len(prob.names)) if i != idx]

    if span is None:
        span = np.log(3.0)
        try:                                  # a local sd is only a hint here
            sd = svd_report(prob, res).sd[idx]
            if np.isfinite(sd) and sd > 0:
                span = float(np.clip(8.0 * sd, np.log(1.05), np.log(30.0)))
        except Exception:
            pass

    grid = np.linspace(res.theta[idx] - span, res.theta[idx] + span, npts)
    mid = int(np.argmin(np.abs(grid - res.theta[idx])))
    # chi2 is the criterion; the dB rms is carried alongside only so the report
    # can speak in dB.  Thresholding on the dB rms instead would be thresholding
    # a DIFFERENT quantity from the one the fit minimises -- with S11 in the
    # residual the two are not proportional, and the interval comes out wrong.
    chi2 = np.full(npts, np.nan)
    rms = np.full(npts, np.nan)
    chi2[mid] = float(np.sum(prob.residual(res.theta) ** 2))
    rms[mid] = res.rms

    from scipy.optimize import least_squares
    lo, hi = np.log(prob.lo), np.log(prob.hi)

    for direction in (range(mid + 1, npts), range(mid - 1, -1, -1)):
        warm = res.theta[others].copy()
        for j in direction:
            def r(p, v=grid[j]):
                th = res.theta.copy()
                th[others] = p
                th[idx] = v
                return prob.residual(th)
            out = least_squares(r, np.clip(warm, lo[others] + 1e-9, hi[others] - 1e-9),
                                bounds=(lo[others], hi[others]), x_scale=1.0,
                                max_nfev=2000)
            warm = out.x
            th = res.theta.copy()
            th[others] = out.x
            th[idx] = grid[j]
            chi2[j] = float(np.sum(out.fun ** 2))
            rms[j] = prob.rms_db(th)

    return Profile(name, np.exp(grid), chi2, rms, float(np.exp(res.theta[idx])),
                   int(prob.keep.sum()), dchi2)


class Profile:
    def __init__(self, name, values, chi2, rms, best, m, dchi2):
        self.name, self.values = name, values
        self.chi2, self.rms = chi2, rms
        self.best, self.m, self.dchi2 = best, m, dchi2
        self.chi2_min = float(np.nanmin(chi2))
        self.limit = self.chi2_min + dchi2          # the actual criterion
        self.rms_min = float(np.nanmin(rms))
        # The same bound expressed as a rise in dB rms, purely so the report can
        # say how invisible it is.
        self.factor = threshold_factor(m, dchi2)
        self.rms_limit = self.rms_min * self.factor

    def interval(self):
        """(lo, hi) at the 95% crossing, linearly interpolated between grid
        points, or None on a side where the profile never crosses.

        Interpolating is not a nicety.  The interval is often much narrower than
        one grid cell, and returning the nearest grid values instead reports a
        width of zero -- which reads as certainty rather than as "the grid was
        too coarse to tell".
        """
        v, r = np.asarray(self.values, float), np.asarray(self.chi2, float)
        ok = np.isfinite(r)
        if ok.sum() < 2:
            return (None, None)
        i0 = int(np.nanargmin(r))

        def walk(step):
            j = i0
            while 0 <= j + step < len(r):
                if not np.isfinite(r[j + step]):
                    return None
                if r[j + step] > self.limit:
                    lo_r, hi_r = r[j], r[j + step]
                    if hi_r == lo_r:
                        return float(v[j + step])
                    t = (self.limit - lo_r) / (hi_r - lo_r)
                    lv, hv = np.log(v[j]), np.log(v[j + step])
                    return float(np.exp(lv + t * (hv - lv)))
                j += step
            return None                        # never left the band

        return (walk(-1), walk(+1))

    def verdict(self):
        """A word, not a number, because the number is what misleads.

        Reported as a multiplicative spread, which is the natural unit in log
        space: "within 6%" or "anywhere over a decade".
        """
        lo, hi = self.interval()
        if lo is None or hi is None:
            return "NOT MEASURED", float("inf")
        spread = float(np.sqrt(hi / lo))
        if spread < 1.1:
            return "identified", spread
        if spread < 2.0:
            return "weakly identified", spread
        return "NOT MEASURED", spread

    def report(self):
        word, spread = self.verdict()
        lo, hi = self.interval()
        out = [f"profile of {self.name}: best {self.best:.6g}, {word}"]
        out.append(f"  95% bound is dchi2={self.dchi2:.2f} -- over m={self.m} points "
                   f"that is a rise of only {100*(self.factor-1):.2f}% in rms, "
                   f"{self.rms_min:.5f} -> {self.rms_limit:.5f} dB.")
        out.append(f"  (invisible by eye, which is why 'the profile looks flat' "
                   f"is not a test)")
        if lo is None or hi is None:
            out.append(f"  the profile never leaves that band inside the scanned "
                       f"range ({self.values[0]:.4g} to {self.values[-1]:.4g}).")
            out.append(f"  Report this parameter as an assumption, not a result.")
        else:
            out.append(f"  95% interval {lo:.6g} to {hi:.6g}  (x/ {spread:.2f})")
        return "\n".join(out)

    def __repr__(self):
        return f"<Profile {self.name} {self.verdict()[0]}>"


def svd_report(prob, result=None, sigma=None):
    """Local identifiability from the Jacobian at the optimum.

    Residuals are already divided by their sigma in `Problem.residual`, so the
    Jacobian is in units of sigma and the covariance is (J^T J)^-1 directly.

    In LOG space `sqrt(diag(Cov))` IS the fractional error on the parameter,
    which is what makes the thresholds below interpretable without units.

    `sigma` defaults to sqrt(reduced chi2) rather than 1.  That is the standard
    correction, and it matters here: on real bench data the residual is usually
    several times the trace noise because the MODEL is imperfect, and leaving
    sigma at 1 then reports error bars several times too small.  It is a patch,
    not a cure -- it widens the interval but still assumes the model shape is
    right -- so the report says when it has been applied.
    """
    res = result or prob.run()
    scaled = sigma is None
    if scaled:
        sigma = float(np.sqrt(max(res.chi2_red, 1.0)))
    if res.at_bound.any():
        stuck = [n for n, b in zip(prob.names, res.at_bound) if b]
        raise ValueError(
            f"parameter(s) at a bound ({', '.join(stuck)}); the covariance is not "
            "defined there.  Widen the bounds or drop the parameter -- do not "
            "read an uncertainty off this fit.")
    J = np.asarray(res.res.jac, float)
    U, s, Vt = np.linalg.svd(J, full_matrices=False)
    s_safe = np.where(s > 0, s, np.inf)
    Cov = (Vt.T * (sigma / s_safe) ** 2) @ Vt
    sd = np.sqrt(np.clip(np.diag(Cov), 0, None))
    denom = np.outer(sd, sd)
    corr = np.divide(Cov, denom, out=np.zeros_like(Cov), where=denom > 0)
    return SVDReport(prob.names, s, sd, corr, Vt[-1],
                     sigma if scaled else None, res.chi2_red)


class SVDReport:
    SD_UNMEASURED = 0.30        # +-35% -- effectively unknown
    COND_DEGENERATE = 1e6       # past this at least one direction is at noise
    CORR_TRADES = 0.95

    def __init__(self, names, sv, sd, corr, v_min, sigma_scale=None, chi2_red=None):
        self.names, self.sv, self.sd, self.corr, self.v_min = names, sv, sd, corr, v_min
        self.sigma_scale, self.chi2_red = sigma_scale, chi2_red
        self.cond = float(sv[0] / sv[-1]) if sv[-1] > 0 else float("inf")
        self.p_eff = int((sd < self.SD_UNMEASURED).sum())

    def trades(self):
        out = []
        n = len(self.names)
        for i in range(n):
            for j in range(i + 1, n):
                if abs(self.corr[i, j]) > self.CORR_TRADES:
                    out.append((self.names[i], self.names[j], float(self.corr[i, j])))
        return out

    def report(self):
        out = [f"{len(self.names)} parameters, {self.p_eff} determined "
               f"(condition number {self.cond:.3g})"]
        if self.sigma_scale and self.sigma_scale > 1.05:
            out.append(f"  error bars widened {self.sigma_scale:.1f}x because the "
                       f"residual exceeds the assumed noise (reduced chi2 "
                       f"{self.chi2_red:.1f}) -- that excess is model error")
        for n, sd in zip(self.names, self.sd):
            mark = "  NOT MEASURED" if sd >= self.SD_UNMEASURED else ""
            pct = 100 * sd
            # A tightly-determined parameter must not print as "+/- 0.0%", which
            # reads as a bug rather than as a small number.
            txt = f"{pct:6.1f}%" if pct >= 0.1 else f"{pct:6.3f}%"
            out.append(f"  {n:<12} +/- {txt}{mark}")
        if self.cond > self.COND_DEGENERATE:
            out.append(f"  condition number {self.cond:.3g} -- at least one direction "
                       f"is at the noise level; treat it as degenerate.")
        for a, b, c in self.trades():
            out.append(f"  {a} and {b} trade against each other (corr {c:+.3f}); "
                       f"quote the combination, not either one alone.")
        soft = ", ".join(f"{n}^{v:+.2f}" for n, v in zip(self.names, self.v_min)
                         if abs(v) > 0.15)
        out.append(f"  softest direction: {soft or '(none dominant)'}")
        return "\n".join(out)

    def __repr__(self):
        return f"<SVDReport p_eff={self.p_eff}/{len(self.names)} cond={self.cond:.3g}>"
