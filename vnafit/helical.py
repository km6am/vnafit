"""The helical-filter bench, on the ordinary machinery.

A helical filter is a parallel L||C resonator per section, magnetically coupled
through an aperture, fed by a tap on the end coils.  Nothing about that needs
its own solver, its own plotting or its own calibration -- so this module is
only the part that IS specific: turning "four sections, 146 MHz, 2 MHz wide"
into a netlist.  Everything after that is `mna`, `card`, `fit`, `cal` and the
ordinary tuning window.

`design()` is deliberately thin.  `coupled.targets` already does the Chebyshev
synthesis and `synth.magnetic` already writes the topology; this picks the
sweep span, declares which parameters may be fitted, and refuses to declare the
one that cannot.

**`lref` is never fitted.**  With tapped ports every impedance in the circuit
scales together, so raising L and lowering C at fixed f0/k/Qe/Qu changes |S21|
by about 1e-11 dB.  It is an exact gauge freedom, and a fit that turns it loose
slides along that flat direction until it hits a bound and then reports the
bound.  See `docs/identifiability.md`.
"""
import numpy as np

from . import coupled, synth

MAX_SECTIONS = 4

# The bench sequence.  Each stage is the SAME filter with everything but the
# thing being set stripped out, so the same solver, card, fitter and window
# work on all of them -- there is no separate "alignment mode" anywhere in this
# codebase, only a smaller netlist.
#
#   align    one resonator, both probes slack.  Loaded Q ~ Qu, so this is where
#            f0 and the unloaded Q come from.
#   couple   one resonator, the real tap on port 1 and a slack probe on port 2.
#            The singly-loaded 3 dB width gives Qe.
#   gap      two resonators, both probes slack.  The peaks split by k*f0.
#   measure  the whole filter, terminated properly.
#
# A probe is slack relative to the UNLOADED Q, not relative to the real tap.
# The first version of this made it 20x Qe, reasoning that 20x is obviously
# loose -- but the quantity that matters is the probe's Q against Qu, and
# 20 x 75 = 1500 against Qu = 670 means the two probes carry 45% of the total
# loading.  The measured width then came out 0.408 MHz where f0/Qu is 0.218:
# nearly double, on the stage whose entire purpose is reading Qu.
#
# So the probe is specified by what it does: PROBE_Q x Qu, which adds
# 1/PROBE_Q of the resonator's own loss per probe.  At 50 that is 2% each.
STAGES = ("align", "couple", "gap", "measure")
PROBE_Q = 50.0


def design(order, f0=146e6, bw=2e6, ripple_db=0.1, qu=670, L=330e-9,
           f0s=None, span=6.0, points=401, stage="measure", section=1):
    """A helical filter of `order` sections, as a fittable netlist.

    `bw` is the RIPPLE bandwidth for a Chebyshev design and the 3 dB bandwidth
    for Butterworth (`ripple_db <= 0`), which is what `coupled.targets` means by
    it -- getting that wrong is the mistake that cost this project a week on the
    TinyFilter.

    `f0s` overrides the per-resonator frequencies, so a deliberately detuned
    starting point can be built; by default they are all `f0`.  `span` is how
    many 3 dB bandwidths either side of centre the `.ac` line covers.
    """
    order = int(order)
    if not 1 <= order <= MAX_SECTIONS:
        raise ValueError(f"between 1 and {MAX_SECTIONS} sections, not {order}")
    t = coupled.targets(order, f0, bw, ripple_db)
    f0s = list(f0s) if f0s is not None else [f0] * order
    if len(f0s) != order:
        raise ValueError(f"{order} sections need {order} frequencies")

    if stage not in STAGES:
        raise ValueError(f"stage must be one of {', '.join(STAGES)}")
    if stage != "measure":
        return _stage(stage, section, t, f0s, qu, L, span, points)

    nl = synth.magnetic(f0s, t["k"], t["Qe1"], t["Qen"], Qu=qu, L=L,
                        title=f"{order}-section helical, "
                              f"{f0/1e6:.3f} MHz, {bw/1e6:.3f} MHz "
                              f"{'Chebyshev ' + str(ripple_db) + ' dB' if ripple_db > 0 else 'Butterworth'}")
    half = max(span * t["bw3"], 2 * bw)
    nl.ac = ("lin", int(points), max(f0 - half, 1e5), f0 + half)

    # What may be turned loose, and what may not.
    fits = []
    for i in range(1, order + 1):
        fits.append((f"f0{i}", f0s[i - 1], f0 - 3 * t["bw3"], f0 + 3 * t["bw3"]))
    for i, k in enumerate(t["k"], 1):
        fits.append((f"k{i}", k, k / 8, k * 8))
    # Both taps are real even for a single section, where they land on the same
    # resonator -- and the analysis says both are separately measurable, so
    # skipping one was simply wrong.
    fits.append(("qe1", t["Qe1"], t["Qe1"] / 8, t["Qe1"] * 8))
    fits.append(("qen", t["Qen"], t["Qen"] / 8, t["Qen"] * 8))
    fits.append(("qu", qu, qu / 10, qu * 10))
    from .netlist import FitSpec
    for name, seed, lo, hi in fits:
        nl.fits.append(FitSpec(name, seed, lo, hi))
    nl.targets = t
    return nl


def _stage(stage, section, t, f0s, qu, L, span, points):
    """One bench step as a netlist, with only the parameters it can settle."""
    from .netlist import FitSpec
    n = t["n"]
    i = max(1, min(int(section), n if stage != "gap" else max(n - 1, 1)))
    weak1 = weakn = PROBE_Q * qu

    if stage == "gap":
        if n < 2:
            raise ValueError("a gap needs two resonators; this design has one")
        used, ks = f0s[i - 1:i + 1], [t["k"][i - 1]]
        qe1, qen = weak1, weakn
        title = (f"gap {i}-{i+1}: two resonators, probes slack -- the peaks "
                 f"split by k*f0 = {t['split'][i-1]/1e6:.4f} MHz")
        fits = [(f"f0{j}", f0s[i - 2 + j], t["f0"] - 3 * t["bw3"],
                 t["f0"] + 3 * t["bw3"]) for j in (1, 2)]
        fits.append(("k1", t["k"][i - 1], t["k"][i - 1] / 8, t["k"][i - 1] * 8))
    else:
        used, ks = [f0s[i - 1]], []
        if stage == "align":
            qe1, qen = weak1, weakn
            title = (f"align section {i}: one resonator, both probes slack -- "
                     f"f0 and unloaded Q "
                     f"(the two probes add {200/PROBE_Q:.0f}% to the width)")
            fits = [(f"f01", f0s[i - 1], t["f0"] - 3 * t["bw3"],
                     t["f0"] + 3 * t["bw3"]),
                    ("qu", qu, qu / 10, qu * 10)]
        else:                                   # couple
            qe1, qen = (t["Qe1"] if i == 1 else t["Qen"]), weakn
            want = t["f0"] * (1 / qu + 1 / qe1)
            title = (f"couple section {i}: real tap on port 1, probe slack on "
                     f"port 2 -- loaded 3 dB width {want/1e6:.4f} MHz")
            fits = [("qe1", qe1, qe1 / 8, qe1 * 8),
                    (f"f01", f0s[i - 1], t["f0"] - 3 * t["bw3"],
                     t["f0"] + 3 * t["bw3"])]

    nl = synth.magnetic(used, ks, qe1, qen, Qu=qu, L=L, title=title)
    half = max(span * t["bw3"], 2 * t["fbw"] * t["f0"])
    nl.ac = ("lin", int(points), max(t["f0"] - half, 1e5), t["f0"] + half)
    for name, seed, lo, hi in fits:
        nl.fits.append(FitSpec(name, seed, lo, hi))
    nl.targets = t
    nl.stage = (stage, i)
    return nl


def _qu_of(nl):
    p = nl.resolve_params()
    return float(p.get("qu", 1.0))


def stage_hint(nl):
    """What to read off the screen at this stage, in the bench's own words."""
    stage, i = getattr(nl, "stage", ("measure", 0))
    t = nl.targets
    if stage == "align":
        qu = _qu_of(nl)
        return (f"Set section {i} to {t['f0']/1e6:.4f} MHz.  With both probes "
                f"slack the 3 dB width is {t['f0']/qu*(1+2/PROBE_Q)/1e3:.1f} kHz "
                f"for Qu = {qu:.0f} -- that includes {200/PROBE_Q:.0f}% added "
                f"by the probes themselves.")
    if stage == "couple":
        qe = t["Qe1"] if i == 1 else t["Qen"]
        qu = _qu_of(nl)
        # NOT f0/Qe.  What the analyser shows is the LOADED width, and the
        # resonator's own loss is in there too: 1/QL = 1/Qu + 1/Qe.  With
        # Qu/Qe about 9 that is an 11% difference, and aiming at f0/Qe makes
        # you over-couple by that much -- measured, 2.153 MHz against the
        # 1.939 the naive figure asks for.
        want = t["f0"] * (1 / qu + 1 / qe)
        return (f"Set the tap on section {i} until the 3 dB width is "
                f"{want/1e6:.4f} MHz.  That is Qe = {qe:.2f} loaded by "
                f"Qu = {qu:.0f}; the tap alone would give "
                f"{t['f0']/qe/1e6:.4f} MHz, which is the number to aim at only "
                f"if Qu is very much larger than Qe.")
    if stage == "gap":
        return (f"Set the aperture between {i} and {i+1} until the two peaks "
                f"are {t['split'][i-1]/1e6:.4f} MHz apart.  That is k = "
                f"{t['k'][i-1]:.5f}.")
    return (f"The whole filter: {t['bw3']/1e6:.4f} MHz at 3 dB, return loss "
            f"{t['rl']:.1f} dB." if t.get("rl") else "The whole filter.")


def summary(t):
    """The design in the numbers a bench procedure is written in."""
    out = [f"{t['n']} sections, {t['f0']/1e6:.4f} MHz",
           f"  ripple bandwidth {t['fbw']*t['f0']/1e6:.4f} MHz, "
           f"3 dB {t['bw3']/1e6:.4f} MHz"]
    if t.get("rl") is not None:
        out.append(f"  passband return loss {t['rl']:.2f} dB")
    out.append("  k    " + "  ".join(f"{k:.5f}" for k in t["k"]))
    out.append("  split" + "  ".join(f" {s/1e6:.4f}" for s in t["split"])
               + "  MHz between the peaks of a two-resonator pair")
    out.append(f"  Qe   {t['Qe1']:.2f} / {t['Qen']:.2f}")
    out.append(f"  singly-loaded 3 dB width {t['sl_bw1']/1e6:.4f} / "
               f"{t['sl_bwn']/1e6:.4f} MHz")
    return "\n".join(out)
