"""Headless entry point.

    vnafit show     model.net                       what the model does
    vnafit compare  model.net measured.s2p          model vs measurement
    vnafit fit      model.net measured.s2p          fit, then say what was measured
    vnafit profile  model.net measured.s2p -p cm    scan one parameter
    vnafit sweep    out.s2p --start .. --stop ..    capture from the NanoVNA

Every command that reports a fitted number also reports whether the data
determined it.  That is not a flag you have to remember to pass.
"""
import argparse
import os
import sys

import numpy as np

from . import fit as F
from . import identify, mna, touchstone
from .coupled import db
from .netlist import NetlistError, load as _load_vnafit


def load_netlist(path, quiet=False):
    """Load a vnafit netlist OR an ordinary SPICE one, whichever it is.

    A raw LTspice export has no ports and no parameters, so on its own it can
    be simulated and nothing else.  Both are supplied here -- ports inferred
    from the source and load resistors, element values promoted to parameters
    -- and what was inferred is printed, because a silent guess about which
    node is a port models a different network.
    """
    from . import spice as SP
    nl, report = SP.load_any(path)
    if report is not None and not quiet:
        txt = report.text()
        if txt:
            print(f"{os.path.basename(path)} is a SPICE netlist:\n{txt}")
    return nl


def _grid(nl, args):
    if args.start and args.stop:
        return np.linspace(args.start, args.stop, args.points)
    if nl.ac:
        kind, n, a, b = nl.ac
        n = args.points or n
        return np.logspace(np.log10(a), np.log10(b), n) if kind == "dec" \
            else np.linspace(a, b, n)
    raise SystemExit("no frequency range: give --start/--stop or a .ac line")


# The AM broadcast band and the HF ham bands a BCI filter has to keep.  Used
# only for reporting and shading; nothing depends on them numerically.
AM_BAND = (540e3, 1710e3)
HAM_HF = [("160m", 1.8e6, 2.0e6), ("80m", 3.5e6, 4.0e6), ("40m", 7.0e6, 7.3e6),
          ("30m", 10.1e6, 10.15e6), ("20m", 14.0e6, 14.35e6),
          ("17m", 18.068e6, 18.168e6), ("15m", 21.0e6, 21.45e6),
          ("12m", 24.89e6, 24.99e6), ("10m", 28.0e6, 29.7e6)]


def _at_table(f, S, freqs):
    """Attenuation and match at named frequencies.

    A peak-and-bandwidth summary is meaningless for a high-pass, which is what
    a BCI filter is; what you actually want to know is how much is left at
    1 MHz and how little is lost at 7 MHz.
    """
    out = ["    freq        |S21|      |S11|    VSWR"]
    for fr in freqs:
        i = int(np.argmin(np.abs(f - fr)))
        s21, s11 = S[i, 1, 0], S[i, 0, 0]
        m = min(abs(s11), 0.999999)
        vswr = (1 + m) / (1 - m)
        out.append(f"  {f[i]/1e6:8.3f} MHz {db(s21):+8.2f}  {db(s11):+8.2f}  {vswr:6.2f}")
    return out


def _trace_summary(f, s21, s11=None):
    d = db(s21)
    i = int(np.argmax(d))
    out = [f"  peak {d[i]:+7.3f} dB at {f[i]/1e6:8.4f} MHz"]
    for lvl in (3.0, 20.0):
        inside = np.flatnonzero(d >= d[i] - lvl)
        if len(inside) > 1 and inside[0] > 0 and inside[-1] < len(f) - 1:
            bw = f[inside[-1]] - f[inside[0]]
            out.append(f"  {lvl:>4.0f} dB BW {bw/1e6:8.4f} MHz")
    if s11 is not None:
        out.append(f"  best return loss {-20*np.log10(np.abs(s11).min()):6.2f} dB")
    return out


def _load_pair(args):
    nl = load_netlist(args.netlist)
    t = touchstone.load(args.measured)
    if t.nports < 2:
        raise SystemExit(f"{args.measured} is 1-port; a filter fit needs S21")
    f = t.f
    if args.start:
        f = f[f >= args.start]
    if args.stop:
        f = f[f <= args.stop]
    m = np.isin(t.f, f)
    return nl, t, t.f[m], t.s11[m], t.s21[m]


def cmd_show(args):
    nl = load_netlist(args.netlist)
    f = _grid(nl, args)
    b = mna.build(nl)
    S = b.solve(f)
    print(f"{nl.title or os.path.basename(args.netlist)}")
    print(f"  {len(nl.elements)} elements, {len(nl.ports)} ports, MNA dim {b.dim}")
    print(f"  {len(f)} points, {f[0]/1e6:.4f}-{f[-1]/1e6:.4f} MHz")
    if args.at:
        for line in _at_table(f, S, [float(x) for x in args.at.split(",")]):
            print(line)
    elif args.highpass:
        d = db(S[:, 1, 0])
        i3 = np.flatnonzero(d >= d.max() - 3.0)
        if len(i3):
            print(f"  3 dB corner {f[i3[0]]/1e6:8.4f} MHz")
        print("  attenuation in the AM broadcast band "
              f"({AM_BAND[0]/1e3:.0f}-{AM_BAND[1]/1e3:.0f} kHz):")
        m = (f >= AM_BAND[0]) & (f <= AM_BAND[1])
        if m.any():
            print(f"    worst {d[m].max():+7.2f} dB, best {d[m].min():+7.2f} dB")
        print("  insertion loss on the HF bands:")
        for name, lo, hi in HAM_HF:
            k = (f >= lo) & (f <= hi)
            if k.any():
                print(f"    {name:>5} {d[k].min():+7.3f} to {d[k].max():+7.3f} dB")
    else:
        for line in _trace_summary(f, S[:, 1, 0], S[:, 0, 0]):
            print(line)
    if args.out:
        touchstone.save(args.out, f, S, comments=[f"vnafit model {nl.title}"])
        print(f"  wrote {args.out}")


def cmd_compare(args):
    nl, t, f, s11, s21 = _load_pair(args)
    S = mna.build(nl).solve(f)
    print(f"measured  {os.path.basename(args.measured)}")
    for line in _trace_summary(f, s21, s11):
        print(line)
    print(f"model     {nl.title or os.path.basename(args.netlist)}")
    for line in _trace_summary(f, S[:, 1, 0], S[:, 0, 0]):
        print(line)
    d = db(S[:, 1, 0]) - db(s21)
    k = db(s21) > F.DEFAULT_FLOOR_DBC + 6.0
    print(f"residual  {np.sqrt(np.mean(d[k]**2)):.3f} dB rms over {k.sum()} points "
          f"more than 6 dB above the assumed {F.DEFAULT_FLOOR_DBC:.0f} dBc floor")
    # Passband and skirts separately: every model agrees in the passband, so a
    # single number hides where the disagreement actually is.
    pk = f[np.argmax(db(s21))]
    near = k & (np.abs(f - pk) < 0.02 * pk)
    far = k & ~near
    if near.any():
        print(f"          passband {np.sqrt(np.mean(d[near]**2)):.3f} dB rms")
    if far.any():
        print(f"          skirts   {np.sqrt(np.mean(d[far]**2)):.3f} dB rms")


def _problem(args):
    nl, t, f, s11, s21 = _load_pair(args)
    band = None if (args.start or args.stop or args.full_span) else "auto"
    return F.Problem(nl, f, None if args.no_s11 else s11, s21,
                     sigma_db=args.sigma, sigma_s11=args.sigma_s11,
                     floor_dbc=args.floor, band=band)


def cmd_fit(args):
    p = _problem(args)
    r = p.run()
    print(r.report())
    print()
    if r.at_bound.any():
        return
    try:
        print(identify.svd_report(p, r).report())
    except ValueError as e:
        print(f"identifiability not available: {e}")
        return
    if args.profile_all:
        print()
        for name in p.names:
            print(identify.profile(p, name, r, npts=args.npts).report())
            print()


def cmd_profile(args):
    p = _problem(args)
    r = p.run()
    print(r.report())
    print()
    names = args.param or p.names
    for name in names:
        pr = identify.profile(p, name, r, npts=args.npts)
        print(pr.report())
        if args.plot:
            # Scale to the curve's own range, and mark the 95% line, so the
            # shape is readable whether the parameter is sharp or flat.
            rng = max(np.nanmax(pr.chi2) - pr.chi2_min, 1e-12)
            for v, c in zip(pr.values, pr.chi2):
                bar = "#" * int(round(56 * (c - pr.chi2_min) / rng))
                mark = "in " if c <= pr.limit else "out"
                print(f"    {v:12.6g} {mark} {bar}")
            print(f"    {'':12} 95% line at chi2 = {pr.limit:.1f} "
                  f"(min {pr.chi2_min:.1f})")
        print()


def cmd_plot(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9.5, 6.6), sharex=True,
                                 gridspec_kw=dict(height_ratios=[2, 1]))
    for path in args.netlist:
        nl = load_netlist(path)
        f = _grid(nl, args)
        S = mna.build(nl).solve(f)
        lab = nl.title or os.path.basename(path)
        a1.semilogx(f / 1e6, db(S[:, 1, 0]), lw=1.8, label=lab)
        a2.semilogx(f / 1e6, db(S[:, 0, 0]), lw=1.4, label=lab)
    if args.measured:
        t = touchstone.load(args.measured)
        a1.semilogx(t.f / 1e6, db(t.s21), lw=2.2, color="#1f6fd0",
                    label=f"measured {os.path.basename(args.measured)}")
        a2.semilogx(t.f / 1e6, db(t.s11), lw=1.6, color="#e06010")

    a1.axvspan(AM_BAND[0] / 1e6, AM_BAND[1] / 1e6, color="#c0442e", alpha=.10)
    a1.annotate("AM broadcast", (np.sqrt(AM_BAND[0] * AM_BAND[1]) / 1e6, 4),
                ha="center", fontsize=8, color="#c0442e")
    for name, lo, hi in HAM_HF:
        for ax in (a1, a2):
            ax.axvspan(lo / 1e6, hi / 1e6, color="#2e8b57", alpha=.13)
    a1.set_ylabel("|S21|  dB")
    a1.set_ylim(args.ymin, 6)
    a2.set_ylabel("|S11|  dB")
    a2.set_ylim(-45, 2)
    a2.set_xlabel("MHz")
    for ax in (a1, a2):
        ax.grid(which="both", alpha=.22)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    a1.legend(loc="lower right", fontsize=8, framealpha=.92)
    a1.set_title(args.title or "", fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"wrote {args.out}")


def cmd_laws(args):
    from . import coupled as M
    t = touchstone.load(args.measured)
    if t.nports < 2:
        raise SystemExit(f"{args.measured} is 1-port; this needs S21")
    d = db(t.s21)
    floor = (args.floor if args.floor is not None
             else float(np.median(np.r_[d[:30], d[-30:]])))
    k = d > floor + args.margin
    f, s11, s21 = t.f[k], t.s11[k], t.s21[k]
    if k.sum() < 30:
        raise SystemExit(f"only {k.sum()} points survive the {floor:.0f} dB floor")

    res = M.compare_laws(f, s11, s21, args.poles, args.qu)
    print(f"{os.path.basename(args.measured)}: {k.sum()} points, "
          f"{f[0]/1e6:.1f}-{f[-1]/1e6:.1f} MHz (x{f[-1]/f[0]:.2f}), "
          f"floor {floor:.0f} dB, Qu pinned {args.qu:g}")
    for name in res["order"]:
        r = res["fits"][name]
        kx, qx = M.COUPLING_LAWS[name]
        print(f"  {name:<15} (k~f^{kx:+.0f}, Qe~f^{qx:+.0f})  rms {r['rms']:7.3f} dB"
              f"   k={r['ks'][0]:.5f}  Qe={r['Qe1']:.0f}/{r['Qen']:.0f}")
    print()
    print(f"  power check -- residual under constant-k is {res['residual']:.2f} dB,")
    print(f"  so a law must separate from it by more than "
          f"{res['need_separation']:.2f} dB to be distinguishable:")
    for name in res["order"]:
        if name == "constant":
            continue
        s_ = res["separation"].get(name, float("nan"))
        verdict = ("DISCRIMINATING" if s_ > res["need_separation"]
                   else "underpowered")
        print(f"    {name:<15} separation {s_:6.2f} dB   {verdict}")
    print()
    if res["powered"]:
        print(f"  VERDICT: {res['best']} -- the data can tell, and it does.")
    else:
        print("  VERDICT: this sweep CANNOT distinguish the laws.  The ranking")
        print("  above is noise; do not read physics into it.  The exponents only")
        print("  change the skirts, so you need to see further down them -- which")
        print("  means a lower fixture leakage floor, not more averaging.")


def cmd_exponents(args):
    from . import coupled as M
    sweeps, labels = [], []
    for path in args.measured:
        t = touchstone.load(path)
        if t.nports < 2:
            raise SystemExit(f"{path} is 1-port; this needs S21")
        sweeps.append((t.f, t.s11, t.s21))
        labels.append(os.path.basename(path))
    res = M.measure_exponents(sweeps, args.poles, args.qu)

    print(f"{len(sweeps)} tunings, Qu pinned {args.qu:g}")
    print("   f0 (MHz)      k        Qe     fit rms   file")
    for row, lab in zip(res["points"], labels):
        print(f"  {row['f0']/1e6:9.4f}  {row['k']:.5f}  {row['qe']:7.1f}   "
              f"{row['rms']:6.3f}   {lab}")
    print(f"\n  tuning range x{res['span']:.3f}")
    print("  (these are RETUNING slopes, not the within-sweep exponents)")
    for name, sym in (("kexp", "k"), ("qexp", "Qe")):
        se = res[name + "_se"]
        pm = f" +/- {se:.2f}" if np.isfinite(se) else " (no uncertainty: needs 3+ tunings)"
        print(f"  {sym} ~ f0^{res[name]:+.2f}{pm}")
    nm, pair, dist, law = M.nearest_topology(res["kexp"], res["qexp"])
    print(f"\n  nearest topology: {nm}  (expects {pair[0]:+.0f}, {pair[1]:+.0f}; "
          f"distance {dist:.2f})")
    print(f"  -> which implies the within-sweep law `{law}` "
          f"{M.COUPLING_LAWS[law]} for the skirts")
    if dist > 0.8:
        print("  but that distance is large; none of the known topologies fits well.")
    print("\n  Reminder: this only means anything if you retuned with the TUNING")
    print("  SCREW alone.  Moving the aperture or the taps changes the coupling")
    print("  element itself, and then the slope measures your hands.")
    if res["span"] < 1.10:
        print("  WARNING: the tunings span less than 10%, so a f^+-1 law only moves")
        print("  k by ~10% -- comparable to how well k can be measured.  Retune")
        print("  further apart before reading anything into these slopes.")


def cmd_track(args):
    from . import track as TR
    from .vna import NanoVNA
    from .acq import FileSource
    nl = load_netlist(args.netlist) if args.netlist else None
    dev = FileSource(touchstone.load(args.replay)) if args.replay else NanoVNA(args.port)
    try:
        if args.ifbw is not None and hasattr(dev, "set_bandwidth"):
            dev.set_bandwidth(args.ifbw)
        print("device %s" % getattr(dev, "_info", dev))
        print("%.4f-%.4f MHz, %d points, %.0f s, %d-pole fit, Qu pinned %g, %s resonator"
              % (args.start / 1e6, args.stop / 1e6, args.points, args.seconds,
                 args.poles, args.qu, args.resonator))
        if nl is not None:
            print("model: %s" % (nl.title or os.path.basename(args.netlist)))
        print("turn something -- every sweep is logged to %s/track.csv\n" % args.out)
        TR.run(dev, dict(start=args.start, stop=args.stop, points=args.points,
                         segments=args.segments),
               args.seconds, args.poles, args.qu, netlist=nl,
               resonator=args.resonator, out_dir=args.out, every=args.every)
    finally:
        dev.close()


def cmd_schematic(args):
    from . import schematic as SCH
    nl = load_netlist(args.netlist)
    lay = SCH.save(nl, args.out, title=args.title or None)
    print(f"wrote {args.out}")
    print(f"  spine: {' - '.join(lay.spine)}")
    print(f"  {len(lay.series)} series, {len(lay.shunt)} shunt, "
          f"{len(lay.bridges)} bridging, {len(lay.couplings)} coupling")
    if lay.leftover:
        print("  NOT placed in the ladder (drawn as a note, never omitted): "
              + ", ".join(e.name for e in lay.leftover))


def _port_spec(txt):
    if not txt:
        return None
    node, _, z = txt.partition(":")
    return (node, float(z) if z else 50.0)


def cmd_import(args):
    from . import spice as SP
    nl, rep = SP.load_spice(args.spicefile, port1=_port_spec(args.port1),
                            port2=_port_spec(args.port2))
    print(f"{args.spicefile}: {len(nl.elements)} elements")
    print(rep.text())
    text = "\n".join([nl.title or "imported from SPICE"]
                      + [f".port {p.index} {p.pos} {p.neg} Z0={p.z0:g}"
                         for p in nl.ports]
                      + [_reemit(e) for e in nl.elements]
                      + ([f".ac {nl.ac[0]} {nl.ac[1]} {nl.ac[2]:.10g} "
                          f"{nl.ac[3]:.10g}"] if nl.ac else []))
    if args.out:
        open(args.out, "w").write(text + "\n")
        print(f"  wrote {args.out}")
    else:
        print("\n" + text)


def _reemit(e):
    from .spice import _fmt
    if e.kind in ("R", "L", "C"):
        s = f"{e.name} {e.nodes[0]} {e.nodes[1]} {_fmt(e.args['value'])}"
        if "q" in e.args:
            s += f" Q={_fmt(e.args['q'])}"
        return s
    if e.kind == "K":
        return f"{e.name} {e.args['a']} {e.args['b']} {_fmt(e.args['k'])}"
    if e.kind == "XFMR":
        return f"{e.name} {' '.join(e.nodes)} n={_fmt(e.args['n'])}"
    return f"{e.name} {' '.join(e.nodes)} " + " ".join(
        f"{k}={_fmt(v)}" for k, v in e.args.items())


def cmd_export(args):
    from . import spice as SP
    nl = load_netlist(args.netlist)
    text = SP.to_spice(nl, f_ref=args.f_ref, xfmr=args.xfmr,
                       inline=args.literal)
    if args.out:
        open(args.out, "w").write(text)
        print(f"wrote {args.out} -- open it in LTspice, ngspice or EasyEDA")
    else:
        print(text)


def cmd_card(args):
    from . import card as CARD
    nl = load_netlist(args.netlist)
    st, notes = CARD.save(nl, args.out, sigma_db=args.sigma,
                          reduce=not args.raw_params)
    print(st.report(args.sigma))
    for n in notes:
        print("  " + n)
    fit = [n for n in st.names if st.verdict[n][0] == "determined"]
    print("\nsuggested .fit block:")
    for n in fit:
        v = st.p0[n]
        print(f"  .fit {n} {v:.6g} {v/5:.6g} {v*5:.6g}")
    print(f"\nwrote {args.out}")


def cmd_gui(args):
    """Open the window.  Imported lazily so the CLI does not need tkinter."""
    from . import gui
    argv = [a for a in (args.netlist, args.capture) if a]
    gui.main(argv)
    return 0


def cmd_sweep(args):
    from .vna import NanoVNA
    dev = NanoVNA(args.port)
    try:
        print(f"device {dev._info}")
        import contextlib
        with (dev.uncorrected() if args.no_cal else contextlib.nullcontext()):
            if args.no_cal:
                print("instrument correction OFF for this sweep "
                      "(restored when the sweep finishes)")
            f, s11, s21 = dev.scan(args.start, args.stop, args.points)
        S = np.zeros((len(f), 2, 2), complex)
        S[:, 0, 0], S[:, 1, 0] = s11, s21
        # The correction state goes IN THE FILE.  A capture whose calibration is
        # unknown six months later cannot be reasoned about, and this project
        # has already lost an afternoon to exactly that question.
        touchstone.save(args.out, f, S, comments=[
            f"vnafit sweep", f"device {dev.portname}",
            f"span {args.start/1e6:.4f}-{args.stop/1e6:.4f} MHz points {len(f)}",
            f"instrument correction: {'OFF (raw)' if args.no_cal else 'as configured on the instrument'}",
            "S12 and S22 are NOT measured by this instrument and are written as zero"])
        print(f"wrote {args.out}")
        for line in _trace_summary(f, s21, s11):
            print(line)
    finally:
        dev.close()


def main(argv=None):
    ap = argparse.ArgumentParser(prog="vnafit", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def freqs(q):
        q.add_argument("--start", type=float, help="Hz")
        q.add_argument("--stop", type=float, help="Hz")
        q.add_argument("--points", type=int, default=0)

    q = sub.add_parser("show", help="evaluate a netlist")
    q.add_argument("netlist")
    q.add_argument("--out", help="write the model response as a Touchstone file")
    q.add_argument("--at", help="comma-separated frequencies in Hz to tabulate")
    q.add_argument("--highpass", action="store_true",
                   help="summarise as a high-pass: corner, AM-band rejection, "
                        "and insertion loss per ham band")
    freqs(q)
    q.set_defaults(fn=cmd_show)

    q = sub.add_parser("plot", help="plot one or more netlists, optionally over "
                                    "a measurement")
    q.add_argument("netlist", nargs="+")
    q.add_argument("--measured")
    q.add_argument("--out", default="response.png")
    q.add_argument("--title", default="")
    q.add_argument("--ymin", type=float, default=-90.0)
    freqs(q)
    q.set_defaults(fn=cmd_plot)

    q = sub.add_parser("compare", help="netlist against a measurement")
    q.add_argument("netlist")
    q.add_argument("measured")
    freqs(q)
    q.set_defaults(fn=cmd_compare)

    def fitargs(q):
        q.add_argument("netlist")
        q.add_argument("measured")
        q.add_argument("--sigma", type=float, default=0.05,
                       help="passband repeatability in dB; MEASURE it from two "
                            "repeat sweeps rather than trusting this default")
        q.add_argument("--sigma-s11", type=float, default=0.01)
        q.add_argument("--floor", type=float, default=None,
                       help="fixture leakage floor in dBc (default -70); measure "
                            "it with the DUT replaced by two terminated cables")
        q.add_argument("--no-s11", action="store_true")
        q.add_argument("--full-span", action="store_true",
                       help="fit the whole file instead of a window around the "
                            "passband; usually a mistake for a lumped model")
        q.add_argument("--npts", type=int, default=17)
        freqs(q)

    q = sub.add_parser("fit", help="fit .fit parameters and report identifiability")
    fitargs(q)
    q.add_argument("--profile-all", action="store_true",
                   help="also profile every free parameter (slower, more honest)")
    q.set_defaults(fn=cmd_fit)

    q = sub.add_parser("profile", help="scan a parameter, refitting the rest")
    fitargs(q)
    q.add_argument("-p", "--param", action="append")
    q.add_argument("--plot", action="store_true", help="ASCII profile curve")
    q.set_defaults(fn=cmd_profile)

    q = sub.add_parser("laws", help="which coupling law does a measurement "
                                    "support -- if any")
    q.add_argument("measured")
    q.add_argument("--poles", type=int, default=2)
    q.add_argument("--qu", type=float, required=True,
                   help="unloaded Q, PINNED; measure it in single-resonator mode")
    q.add_argument("--floor", type=float, default=None)
    q.add_argument("--margin", type=float, default=8.0,
                   help="dB above the floor to keep")
    q.set_defaults(fn=cmd_laws)

    q = sub.add_parser("exponents", help="measure the k and Qe frequency "
                                         "exponents by retuning, not by skirt-fitting")
    q.add_argument("measured", nargs="+", help="one sweep per tuning")
    q.add_argument("--poles", type=int, default=2)
    q.add_argument("--qu", type=float, required=True)
    q.set_defaults(fn=cmd_exponents)

    q = sub.add_parser("import", help="read an ordinary SPICE netlist "
                                      "(LTspice, EasyEDA, KiCad, ngspice)")
    q.add_argument("spicefile")
    q.add_argument("--out", help="write a vnafit netlist here")
    q.add_argument("--port1", help="node[:Z0] -- overrides the inference")
    q.add_argument("--port2", help="node[:Z0]")
    q.set_defaults(fn=cmd_import)

    q = sub.add_parser("export", help="write an ordinary SPICE netlist")
    q.add_argument("netlist")
    q.add_argument("--out")
    q.add_argument("--f-ref", type=float, default=None,
                   help="frequency at which Q is turned into a resistance")
    q.add_argument("--xfmr", default="refuse", choices=("refuse", "approximate"),
                   help="SPICE has no ideal transformer; refuse to export one "
                        "(default) or write coupled inductors instead")
    q.add_argument("--literal", action="store_true",
                   help="write the numbers instead of .param references, which "
                        "is what a schematic tool actually produces")
    q.set_defaults(fn=cmd_export)

    q = sub.add_parser("schematic", help="draw the netlist as a schematic")
    q.add_argument("netlist")
    q.add_argument("--out", default="schematic.png")
    q.add_argument("--title", default="")
    q.set_defaults(fn=cmd_schematic)

    q = sub.add_parser("card", help="what a sweep could determine about a "
                                    "netlist, before you measure anything")
    q.add_argument("netlist")
    q.add_argument("--out", default="card.png")
    q.add_argument("--sigma", type=float, default=0.05,
                   help="trace noise in dB; measure it from two repeat sweeps")
    q.add_argument("--raw-params", action="store_true",
                   help="do not collapse degenerate parameters into their "
                        "invariant (shows the raw .param set instead)")
    q.set_defaults(fn=cmd_card)

    q = sub.add_parser("track", help="live tuning readout: sweep, fit, print, repeat")
    q.add_argument("--netlist", help="optional model to score against")
    q.add_argument("--start", type=float, required=True)
    q.add_argument("--stop", type=float, required=True)
    q.add_argument("--points", type=int, default=401)
    q.add_argument("--segments", type=int, default=1)
    q.add_argument("--seconds", type=float, default=120.0)
    q.add_argument("--every", type=float, default=1.0)
    q.add_argument("--poles", type=int, default=2)
    q.add_argument("--qu", type=float, default=900.0)
    q.add_argument("--resonator", default="lumped", choices=("lumped", "quarter"))
    q.add_argument("--ifbw", type=int, default=None)
    q.add_argument("--out", default="tracklog")
    q.add_argument("--port")
    q.add_argument("--replay", help="drive from a .s2p instead of hardware")
    q.set_defaults(fn=cmd_track)

    q = sub.add_parser("gui", help="open the live tuning window")
    q.add_argument("netlist", nargs="?",
                   help="netlist or SPICE netlist to load")
    q.add_argument("capture", nargs="?",
                   help="a .s2p to replay instead of an instrument")
    q.set_defaults(fn=cmd_gui)

    q = sub.add_parser("sweep", help="capture from a NanoVNA-H4")
    q.add_argument("out")
    q.add_argument("--start", type=float, required=True)
    q.add_argument("--stop", type=float, required=True)
    q.add_argument("--points", type=int, default=401)
    q.add_argument("--port")
    q.add_argument("--no-cal", action="store_true",
                   help="sweep with the instrument's own correction OFF; it is "
                        "restored afterwards, and the .s2p records which it was")
    q.set_defaults(fn=cmd_sweep)

    args = ap.parse_args(argv)
    try:
        return args.fn(args) or 0
    except (NetlistError, ValueError, OSError) as e:
        print(f"vnafit: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
