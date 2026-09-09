"""Coupling-matrix parameters -> a netlist that realises them.

This is the bridge between the two models, and the answer to "can the helical
model be ported into the netlist approach?".  It can, exactly, but only once you
say WHICH physical filter you mean -- and that is the whole point.

The normalised coupling matrix

    A_ii = 1/Qu + j*(f/f0i - f0i/f)   (+1/Qe on the ends)
    A_i,i+1 = -j*k_i

is the Schur complement of a real network onto its resonator subspace, frozen at
f0.  Several different circuits share the same (f0, k, Qe, Qu) at the centre and
disagree in the skirts, because the coupling ELEMENT's own frequency dependence
is what the coupling COEFFICIENT threw away.  Both filters here are
parallel-resonator (J-inverter) types:

  magnetic()     mutual M between the coils, ports tapped on the coils.
                 The helical.   ->  k ~ 1/f, Qe ~ constant   (-1, 0)

  capacitive()   series coupling caps, series input caps into 50 ohm.
                 The TinyFilter. ->  k ~ f,  Qe ~ 1/f^2      (+1, -2)

They are not just two flavours of the same thing.  They differ in a way that
decides what can be measured at all:

  A TAP is an ideal autotransformer, so every impedance in `magnetic()` scales
  together and the network is EXACTLY invariant under scaling L up and C down at
  fixed (f0, k, Qe, Qu).  Verified to 1e-13 dB over +/-60 MHz.  The absolute L
  of a helical resonator is therefore NOT measurable from S-parameters, and a
  fit that reports one is reporting its own seed.

  A SERIES INPUT CAP works against the fixed 50 ohm, which does not scale.  That
  breaks the invariance, and in `capacitive()` a 4x change in L moves |S21| by
  ~12 dB in-band.  There, L is strongly measurable.

So the netlist buys real new physics -- the tap ratio, the coupling cap -- but
only where the port coupling is modelled as an ELEMENT rather than as an
abstract Qe.  `identify.py` is what checks which case you are in, per fit,
rather than trusting this docstring.
"""
import numpy as np

from .netlist import parse

C_LIGHT = 299792458.0


def _tank(f0, L):
    """Total node capacitance and angular resonance for a parallel L||C tank."""
    w0 = 2 * np.pi * f0
    return 1.0 / (w0 * w0 * L), w0


def tap_ratio(f0, L, Qe, z0=50.0):
    """Turns ratio n for an ideal autotransformer tap giving external Q `Qe`.

    The port's z0 appears across the tank as z0/n^2, and a parallel tank loaded
    by R has Q = R*w0*C, so n = sqrt(w0*C*z0/Qe).
    """
    C, w0 = _tank(f0, L)
    return float(np.sqrt(w0 * C * z0 / Qe))


def series_cap_for_qe(f0, L, Qe, z0=50.0):
    """Series capacitor giving external Q `Qe`, and the shunt C it adds.

    A series Cs into z0 presents Y = jwCs/(1 + jwCs*z0), whose real part is the
    port conductance and whose imaginary part is an extra shunt capacitance that
    DETUNES the end resonator -- and does so frequency-dependently, which is a
    term the coupling-matrix model has no representation for at all.

        Ge = w^2 Cs^2 z0 / (1 + (w Cs z0)^2)      -> Qe = w0*C_total/Ge
        Cs_eff = Cs / (1 + (w Cs z0)^2)

    Returns (Cs, Cs_eff).  Note the (w Cs z0)^2 term is NOT small: at Cs = 4 pF
    and 146 MHz it is ~4%, and the "Qe ~ 1/f^2" law is only the limit where it
    is neglected.
    """
    C, w0 = _tank(f0, L)
    ge = w0 * C / Qe
    x = ge * z0
    if x >= 1.0:
        raise ValueError(
            f"Qe={Qe:g} is too low to reach through a series capacitor with "
            f"L={L:g} H at {f0/1e6:.3f} MHz (it would need a negative "
            f"capacitance).  Use a larger L, or tap the coil instead.")
    u = np.sqrt(x / (1.0 - x))               # u = w0*Cs*z0
    Cs = u / (w0 * z0)
    return float(Cs), float(Cs / (1.0 + u * u))


def _fmt(x):
    """Enough digits that a round trip through the parser is lossless."""
    return f"{x:.12g}"


def magnetic(f0s, ks, Qe1, Qen, Qu=None, L=330e-9, z0=50.0, title=None,
             parametric=True):
    """Tapped, magnetically coupled parallel resonators -- the helical.

    `L` is a GAUGE, not a measurement: any L reproduces the same S-parameters
    exactly, provided C, the tap and Qu move with it.  It is an input here so
    the netlist reads like the physical object, not because the number is
    recoverable.  See the module docstring.

    `parametric` (the default) writes the netlist in terms of `.param f01..f0n,
    k1.., qe1, qen, qu` and derives every element from them.  Without it the
    numbers are baked into the elements, and a netlist with no parameters is
    one nothing downstream can do anything with: the identifiability card
    refuses it, the fitter has nothing to turn loose, and the tuning readout
    has no resonator to name.  Baked output stays available because it is
    easier to read when all you want is the circuit.
    """
    n = len(f0s)
    if len(ks) != n - 1:
        raise ValueError(f"{n} resonators need {n-1} couplings, got {len(ks)}")
    lines = [title or f"{n}-pole magnetically coupled, tapped ports (synthesised)"]

    if parametric:
        p = [f"lref={_fmt(L)}"]
        p += [f"f0{i}={_fmt(f0)}" for i, f0 in enumerate(f0s, 1)]
        p += [f"k{i}={_fmt(k)}" for i, k in enumerate(ks, 1)]
        p += [f"qe1={_fmt(Qe1)}", f"qen={_fmt(Qen)}"]
        if Qu:
            p.append(f"qu={_fmt(Qu)}")
        for chunk in [p[i:i + 4] for i in range(0, len(p), 4)]:
            lines.append(".param " + "  ".join(chunk))
        LV, ff = "{lref}", (lambda i: f"f0{i}")
        cap = lambda i: "{1/(4*pi*pi*%s*%s*lref)}" % (ff(i), ff(i))
        qq = "  Q={qu}" if Qu else ""
        kv = lambda i: "{k%d}" % i
        tap = lambda i, q: "{sqrt(%s/(2*pi*%s*lref*%s))}" % (_fmt(z0), ff(i), q)
    else:
        lines.append(f".param lref={_fmt(L)}")
        LV = _fmt(L)
        cap = lambda i: _fmt(_tank(f0s[i - 1], L)[0])
        qq = f"  Q={_fmt(Qu)}" if Qu else ""
        kv = lambda i: _fmt(ks[i - 1])
        tap = lambda i, q: _fmt(tap_ratio(f0s[i - 1], L,
                                          Qe1 if i == 1 else Qen, z0))

    lines.append(f".port 1 p1 0 Z0={_fmt(z0)}")
    lines.append(f".port 2 p2 0 Z0={_fmt(z0)}")

    for i in range(1, n + 1):
        lines.append(f"L{i} n{i} 0 {LV}{qq}")
        lines.append(f"C{i} n{i} 0 {cap(i)}")

    # k = M/sqrt(Li*Lj) is exactly the netlist's K coefficient, so the mapping
    # is the identity here -- for MAGNETIC coupling between parallel resonators.
    for i in range(1, n):
        lines.append(f"K{i} L{i} L{i+1} {kv(i)}")

    for port, idx, q in ((1, 1, "qe1"), (2, n, "qen")):
        lines.append(f"X{port} p{port} 0 n{idx} 0 n={tap(idx, q)}")

    lines.append(_ac(f0s))
    return parse("\n".join(lines))


def capacitive(f0s, ks, Qe1, Qen, Qu=None, L=30e-9, z0=50.0, title=None):
    """Series-cap coupled parallel resonators, series-cap fed -- the TinyFilter.

    Unlike `magnetic`, `L` here is a real parameter: the series input caps work
    against the fixed z0, which does not scale with L, so the impedance level is
    identifiable.
    """
    n = len(f0s)
    if len(ks) != n - 1:
        raise ValueError(f"{n} resonators need {n-1} couplings, got {len(ks)}")
    Ls = [L] * n
    Cs_tot = [_tank(f0, Li)[0] for f0, Li in zip(f0s, Ls)]

    # Coupling caps: k_ij = Cm/sqrt(Ci*Cj) on the TOTAL node capacitances.
    Cm = [k * np.sqrt(Cs_tot[i] * Cs_tot[i + 1]) for i, k in enumerate(ks)]
    Cin, Cin_eff = series_cap_for_qe(f0s[0], Ls[0], Qe1, z0)
    Cout, Cout_eff = series_cap_for_qe(f0s[-1], Ls[-1], Qen, z0)

    # Every series capacitor contributes to the node it lands on, so the
    # physical trimmer is the remainder.  Getting this wrong detunes the end
    # resonators by several percent, which reads as a k error.
    loading = [0.0] * n
    for i, cm in enumerate(Cm):
        loading[i] += cm
        loading[i + 1] += cm
    loading[0] += Cin_eff
    loading[-1] += Cout_eff
    Cphys = [tot - load for tot, load in zip(Cs_tot, loading)]
    for i, c in enumerate(Cphys, 1):
        if c <= 0:
            raise ValueError(
                f"resonator {i}: the coupling and port capacitors already exceed "
                f"the total node capacitance ({Cs_tot[i-1]*1e12:.2f} pF).  L is "
                f"too small, or k/Qe too aggressive, for this topology.")

    lines = [title or f"{n}-pole capacitively coupled (synthesised)"]
    lines.append(f".port 1 p1 0 Z0={_fmt(z0)}")
    lines.append(f".port 2 p2 0 Z0={_fmt(z0)}")
    for i, (Li, c) in enumerate(zip(Ls, Cphys), 1):
        q = f" Q={_fmt(Qu)}" if Qu else ""
        lines.append(f"L{i} n{i} 0 {_fmt(Li)}{q}")
        lines.append(f"C{i} n{i} 0 {_fmt(c)}")
    for i, cm in enumerate(Cm, 1):
        lines.append(f"Cm{i} n{i} n{i+1} {_fmt(cm)}")
    lines.append(f"Cin p1 n1 {_fmt(Cin)}")
    lines.append(f"Cout p2 n{n} {_fmt(Cout)}")
    lines.append(_ac(f0s))
    return parse("\n".join(lines))


def _ac(f0s, span_frac=0.4, points=401):
    f0 = float(np.mean(f0s))
    return f".ac lin {points} {_fmt(f0*(1-span_frac))} {_fmt(f0*(1+span_frac))}"


# The exponent pair each topology PREDICTS, for checking against
# coupled.COUPLING_LAWS.  These are derivations, not fits.
PREDICTED_LAW = {
    "magnetic": (-1.0, 0.0),        # k ~ 1/f (mutual, parallel), Qe flat (tap)
    "capacitive": (+1.0, -2.0),     # k ~ f (series Cm), Qe ~ 1/f^2 (series Cs)
}

TOPOLOGIES = {"magnetic": magnetic, "capacitive": capacitive}
