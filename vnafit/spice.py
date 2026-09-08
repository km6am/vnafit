"""Interchange with ordinary SPICE netlists.

The point of this module is one structural mismatch. A vnafit netlist declares
**ports** -- `.port 1 in 0 Z0=50` -- because S-parameters are defined between
reference impedances. An ordinary SPICE netlist has no such concept. It has a
voltage source and a load resistor, and the port impedance is implicit in
whatever resistance happens to sit in series with them.

So importing is not a syntax problem, it is an inference problem: given

    V1 in 0 AC 1
    Rs in n1 50
    ... filter ...
    RL n2 0 50

the ports are at `n1` and `n2`, both 50 ohm, and `V1`, `Rs`, `RL` are fixture,
not circuit. Get that wrong and the model is of a different network.

The inference is deliberately conservative and always reported. It is better to
say "I think port 2 is RL at node n2, 50 ohm -- override with --port2 if not"
than to guess silently and hand back an authoritative-looking wrong answer.

Exporting is the easy direction: ports become a source with its series
resistance and a load resistor, which is what every SPICE tool expects.

WHAT PRODUCES NETLISTS THIS CAN READ
  LTspice          File > View SPICE Netlist, then save.  Free, desktop, the
                   de-facto standard format.
  EasyEDA          File > Export NetList > Spice.  Free, browser.
  QUCS-S / KiCad   both export SPICE from their schematic editors.
  SPICE-Online     browser, edits a live netlist directly.
  CircuitLab       browser, exports SPICE, but paid.
  Falstad CircuitJS -- popular, but it has its OWN save format and does not
                   export SPICE at all.  Redraw it elsewhere.
"""
import re

from .netlist import NetlistError, Netlist, Port, parse

# LTspice writes the real micro sign, and a few tools use Ohm/ohm suffixes.
_SUBS = {"µ": "u", "μ": "u", "Ω": ""}

# Directives that are meaningful to a simulator but not to a linear AC solver.
# Dropped with a note rather than silently, so an import is never quietly
# missing half of what the file said.
_IGNORABLE = {".backanno", ".end", ".probe", ".options", ".option", ".save",
              ".op", ".tran", ".dc", ".noise", ".four", ".temp", ".width",
              ".plot", ".print", ".meas", ".measure", ".step", ".global"}
_UNSUPPORTED = {".subckt": "subcircuits", ".include": "included files",
                ".inc": "included files", ".lib": "libraries",
                ".model": "device models"}


class ImportReport:
    """What the importer decided, and what it threw away."""

    def __init__(self):
        self.ports = []
        self.dropped = []
        self.fixture = []
        self.notes = []

    def text(self):
        out = []
        for idx, node, z0, why in self.ports:
            out.append(f"  port {idx}: node {node!r}, Z0 {z0:g} ohm  ({why})")
        if self.fixture:
            out.append("  treated as fixture, not circuit: "
                       + ", ".join(self.fixture))
        if self.dropped:
            out.append("  ignored (meaningless to a linear AC solver): "
                       + ", ".join(sorted(set(self.dropped))))
        out.extend("  " + n for n in self.notes)
        return "\n".join(out)


def _clean(text):
    for k, v in _SUBS.items():
        text = text.replace(k, v)
    return text


def _rows(text):
    """Logical lines, with continuations joined and comments stripped."""
    out = []
    for i, raw in enumerate(text.splitlines(), 1):
        s = raw.split(";", 1)[0]
        if s.lstrip().startswith("*"):
            s = ""
        s = s.strip()
        if not s:
            continue
        if s.startswith("+") and out:
            out[-1] = (out[-1][0], out[-1][1] + " " + s[1:].strip())
        else:
            out.append((i, s))
    return out


def _looks_like_element(s):
    """A line that names a known element kind and has enough fields."""
    toks = s.split()
    return (len(toks) >= 3 and toks[0][:1].upper() in "RLCKTXVIEFGHS"
            and not toks[0].startswith("."))


def _is_ac_source(toks):
    """`V1 a b AC 1`, `V1 a b DC 0 AC 1`, `V1 a b 1 AC 1`, or a bare `V1 a b 1`."""
    return any(t.upper() == "AC" for t in toks[3:]) or len(toks) >= 4


def sniff_ports(rows, port1=None, port2=None, report=None):
    """Work out where the ports are in a source-and-load SPICE netlist.

    Returns (ports, fixture_names).  `ports` is [(index, node, z0)].

    The rules, in order, and each one is reported rather than assumed:
      1. An explicit override always wins.
      2. Port 1 is the far side of the resistor in series with the AC source.
         With no series resistor the port is the source node itself and the
         impedance is unknown, so it defaults to 50 and says so.
      3. Port 2 is a resistor to ground that is not the source's, preferring one
         whose value matches the source resistance, then the largest remaining.
    """
    report = report or ImportReport()
    src = None
    res = []          # (name, a, b, value)
    for _ln, s in rows:
        toks = s.split()
        head = toks[0]
        k = head[0].upper()
        if k == "V" and len(toks) >= 4 and _is_ac_source(toks):
            if src is None:
                src = (head, toks[1], toks[2])
        elif k == "R" and len(toks) >= 4:
            try:
                from .units import value as _v
                res.append((head, toks[1], toks[2], _v(toks[3])))
            except ValueError:
                pass

    fixture, ports = [], {}

    if port1:
        ports[1] = (port1[0], port1[1], "given on the command line")
    elif src is not None:
        sname, sp, sn = src
        fixture.append(sname)
        series = [r for r in res if sp in (r[1], r[2])
                  and not (r[1] in ("0", "gnd") or r[2] in ("0", "gnd"))]
        if len(series) == 1:
            rname, a, b, val = series[0]
            far = b if a == sp else a
            ports[1] = (far, val, f"far side of {rname}, the source resistance")
            fixture.append(rname)
        else:
            ports[1] = (sp, 50.0,
                        f"{sname} drives {sp} with no single series resistor, so "
                        f"the port impedance is a guess -- 50 ohm assumed")

    if port2:
        ports[2] = (port2[0], port2[1], "given on the command line")
    else:
        used = set(fixture)
        grounded = [r for r in res if r[0] not in used
                    and (r[1] in ("0", "gnd") or r[2] in ("0", "gnd"))]
        if grounded:
            want = ports.get(1, (None, None))[1]
            match = [r for r in grounded if want and abs(r[3] - want) < 1e-9]
            pick = (match or sorted(grounded, key=lambda r: -r[3]))[0]
            rname, a, b, val = pick
            node = b if a in ("0", "gnd") else a
            why = ("matches the source resistance" if match
                   else "the largest resistor to ground")
            ports[2] = (node, val, f"{rname} to ground, {why}")
            fixture.append(rname)

    if 1 not in ports or 2 not in ports:
        raise NetlistError(
            "could not work out where the ports are.  A SPICE netlist has no "
            "port concept, so vnafit looks for an AC voltage source with a "
            "series resistor (port 1) and a resistor to ground (port 2).  "
            "Neither was found unambiguously -- name them explicitly, e.g. "
            "--port1 in:50 --port2 out:50.")

    for idx in (1, 2):
        node, z0, why = ports[idx]
        report.ports.append((idx, node, z0, why))
    report.fixture = fixture
    return [(idx, ports[idx][0], ports[idx][1]) for idx in (1, 2)], fixture


def import_spice(text, port1=None, port2=None, title=None):
    """Read an ordinary SPICE netlist.  Returns (Netlist, ImportReport)."""
    text = _clean(text)
    rows = _rows(text)
    if not rows:
        raise NetlistError("empty netlist")

    # SPICE's rule is that the FIRST LINE OF THE FILE is the title, whatever it
    # looks like -- and every real exporter writes it as a "*" comment, which
    # _rows() has already stripped.  Applying the rule to the first surviving
    # logical line instead ate the voltage source out of an LTspice export and
    # the port inference then found nothing.  So: take the title from the raw
    # first line, and only consume a logical line as a title when it plainly is
    # not an element or a directive.
    raw_first = text.lstrip().split("\n", 1)[0].strip()
    got_title = title or ""
    body = rows
    if raw_first.startswith("*"):
        got_title = got_title or raw_first.lstrip("* ").strip()
    elif rows and not rows[0][1].startswith(".") and not _looks_like_element(rows[0][1]):
        got_title = got_title or rows[0][1]
        body = rows[1:]

    report = ImportReport()
    ports, fixture = sniff_ports(body, port1, port2, report)
    drop = {n.lower() for n in fixture}

    keep = []
    for _ln, s in body:
        toks = s.split()
        head = toks[0]
        low = head.lower()
        if low.startswith("."):
            if low in _IGNORABLE:
                report.dropped.append(low)
                continue
            if low in _UNSUPPORTED:
                raise NetlistError(
                    f"{low} is not supported: {_UNSUPPORTED[low]} would have to "
                    f"be flattened first.  Expand it in the tool that wrote this "
                    f"file, or delete it if it is unused.")
            if low in (".ac", ".param"):
                keep.append(s)
                continue
            if low in (".control", ".endc"):
                report.dropped.append(low)
                continue
            report.dropped.append(low)
            continue
        if low in drop:
            continue
        k = head[0].upper()
        if k in "RLCKTX":
            keep.append(s)
        elif k in "VI":
            report.fixture.append(head)          # a second source: fixture too
        else:
            report.notes.append(
                f"element {head!r} is not a linear passive part and was dropped; "
                f"this solver models R, L, C, K, transmission lines and ideal "
                f"transformers only")
    lines = [got_title or "imported from SPICE"]
    for idx, node, z0 in ports:
        lines.append(f".port {idx} {node} 0 Z0={z0:g}")
    lines.extend(keep)
    nl = parse("\n".join(lines))
    return nl, report


def promote_literals(nl):
    """Give every literal element value a `.param` of its own.

    An LTspice netlist has no parameters -- it has numbers -- so there is
    nothing for the identifiability analysis to reason about and nothing to
    fit.  Promoting each value to a parameter named after its element makes the
    whole circuit fittable without anyone hand-editing the file.

    This deliberately does NOT try to guess which elements are the same part.
    A hand-written netlist says `Cin` drives both end capacitors and `l` drives
    all three inductors, and that grouping is real information -- but it is the
    author's knowledge, not the netlist's, and inventing it here would be
    asserting a symmetry the file never claimed.  The consequence is more
    parameters than degrees of freedom, which is exactly the situation the
    structural analysis exists to report: it will find the degeneracies and say
    so.  Returns the names it created.
    """
    from .units import Expr
    made = []
    for e in nl.elements:
        for key, val in list(e.args.items()):
            if isinstance(val, Expr):
                continue
            try:
                v = float(val)
            except (TypeError, ValueError):
                continue
            name = e.name.lower() if key == "value" else f"{e.name}_{key}".lower()
            if name in nl.params:                  # never shadow a real .param
                name = f"{name}_{key}".lower()
            if name in nl.params:
                continue
            nl.params[name] = v
            e.args[key] = Expr(name)
            made.append(name)
    return made


def looks_like_spice(text):
    """A vnafit netlist declares `.port`; an ordinary SPICE netlist cannot."""
    return not any(ln.strip().lower().startswith(".port")
                   for ln in text.splitlines())


def load_any(path, promote="auto", group="auto", **kw):
    """Load either flavour of netlist.

    `promote`: True always, False never, "auto" only when the file arrived
    with no parameters at all -- which is every plain SPICE netlist, and no
    hand-written one.

    `group`: tie elements the `roles` heuristic says are one part -- matched end
    capacitors, a set of coupling capacitors, the resonator inductors and their
    loss.  "auto" means only on a file that was just promoted, i.e. never over
    an author's own choices.  Every group made is reported.
    """
    with open(path, "r", errors="replace") as fh:
        text = fh.read()
    if looks_like_spice(text):
        nl, report = import_spice(text, **kw)
    else:
        nl, report = parse(text), None
    promoted = False
    if promote is True or (promote == "auto" and not nl.params):
        made = promote_literals(nl)
        promoted = bool(made)
        if report is not None and made:
            report.notes.append(
                f"{len(made)} element values promoted to parameters so they can "
                f"be analysed and fitted")
    if group is True or (group == "auto" and promoted):
        from . import roles
        try:
            groups = roles.group(nl)
        except Exception:                                    # noqa: BLE001
            groups = []                  # a topology the planner cannot lay out
        if report is not None:
            for name, els, why in groups:
                report.notes.append(
                    f"{name} drives {', '.join(e.name for e in els)}: {why}")
            if groups:
                report.notes.append(
                    "those are a HEURISTIC about how filters are built, applied "
                    "only where the values were already equal -- pass "
                    "group=False to keep one parameter per element")
    return nl, report


def load_spice(path, **kw):
    with open(path, "r", errors="replace") as fh:
        return import_spice(fh.read(), **kw)


# ----------------------------------------------------------------- exporting
Q_SPAN_LIMIT = 4.0      # widest stop/start ratio for which constant-R is fair


def to_spice(nl, style="ngspice", ac=None, params=None, f_ref=None,
             xfmr="refuse", inline=False):
    """Write an ordinary SPICE netlist that LTspice or ngspice will run.

    The ports become what SPICE expects: an AC source behind the port-1
    resistance, and a resistor to ground at port 2.  Anyone opening the result
    gets the same network with the same reference impedances, and `V(port2)` is
    proportional to S21 -- exactly, `S21 = 2*V(p2out)/V(src)` for equal port
    impedances, which is written into the file as a comment so the reader does
    not have to rederive it.

    `Q=` has no SPICE equivalent.  Where the sweep is narrow enough for it to
    mean anything, each lossy element is exported as an ideal one plus a series
    resistance `R = w_ref*L/Q` -- a CONSTANT resistance standing in for a
    constant Q, exact at f_ref and drifting either side.

    Where the sweep is wide it is NOT exported, and that is a measured decision
    rather than laziness.  On the BCI example, declared over 300 kHz to 60 MHz,
    dropping the loss changed the round trip by 0.13 dB while the constant-R
    stand-in changed it by 0.22 dB -- the approximation was worse than the
    omission, because no single R matches a constant Q across 200:1.  So it is
    only used below `Q_SPAN_LIMIT`, and the file says which happened.

    An ideal transformer has no SPICE primitive at all and is refused rather
    than exported as a comment: a comment silently disconnects the circuit, and
    an export that quietly changes the network is worse than one that stops.
    """
    params = params if params is not None else nl.resolve_params()
    p1, p2 = nl.ports[0], nl.ports[1]
    out = [f"* {nl.title or 'vnafit netlist'}",
           "* Exported by vnafit.  Ports are modelled the SPICE way:",
           f"*   V1 drives port 1 through Rsrc = {p1.z0:g} ohm",
           f"*   Rload = {p2.z0:g} ohm terminates port 2",
           "* With equal port impedances, S21 = 2*V(%s) and S11 = 2*V(%s) - 1"
           % (p2.pos, p1.pos)]

    if not inline:
        for name, raw in nl.params.items():
            out.append(f".param {name}={params[name]:.10g}")

    out.append(f"V1 vsrc {p1.neg} AC 1")
    out.append(f"Rsrc vsrc {p1.pos} {p1.z0:g}")
    out.append(f"Rload {p2.pos} {p2.neg} {p2.z0:g}")

    xf = [e for e in nl.elements if e.kind == "XFMR"]
    if xf and xfmr == "refuse":
        raise NetlistError(
            "this netlist contains ideal transformer(s) "
            + ", ".join(e.name for e in xf)
            + ", which SPICE has no primitive for.  Exporting them as comments "
              "would silently disconnect the circuit, so the export stops here.  "
              "Either replace them with coupled inductors by hand, or pass "
              "xfmr='approximate' / --xfmr approximate to have them written as "
              "two coupled inductors with K=0.9999 -- which is an approximation, "
              "not the same network.")

    a_, b_ = (nl.ac[2], nl.ac[3]) if nl.ac else (1e6, 1e8)
    span = max(b_ / a_, 1.0)
    if f_ref is None:
        f_ref = float((a_ * b_) ** 0.5)
    lossy = [e for e in nl.elements if "q" in e.args or "tand" in e.args]
    export_q = bool(lossy) and span <= Q_SPAN_LIMIT
    if lossy and export_q:
        out.append(f"* Q exported as a series resistance at f_ref = "
                   f"{f_ref/1e6:.4f} MHz.  That is a constant R standing in for")
        out.append("* a constant Q: exact at f_ref, drifting either side of it.")
    elif lossy:
        out.append(f"* NOTE: {len(lossy)} element(s) carry Q= in the vnafit "
                   f"netlist.  The loss is NOT exported, because this sweep")
        out.append(f"* spans {span:.0f}:1 and no single resistance stands in for "
                   f"a constant Q over that range -- the approximation would be")
        out.append("* worse than the omission.  The exported network is lossless.")
    for e in nl.elements:
        out.extend(_element_lines(e, style, params, f_ref if export_q else None,
                                  xfmr, inline))

    if ac is None and nl.ac:
        ac = nl.ac
    if ac:
        kind, n, a, b = ac
        out.append(f".ac {kind} {int(n)} {a:.10g} {b:.10g}")
    out.append(".end")
    return "\n".join(out) + "\n"


def _peak_frequency(nl):
    """Where the network passes best -- the sensible place to pin a loss model.

    The geometric centre of the declared sweep is the obvious choice and is a
    poor one: on a high-pass declared over 300 kHz to 60 MHz it lands at 4.2 MHz,
    right on the corner, and the constant-R stand-in for constant-Q is then wrong
    across the whole passband.  Solving on a coarse grid and taking the peak puts
    the approximation where the loss actually shows.
    """
    from . import mna
    import numpy as _np
    a_, b_ = (nl.ac[2], nl.ac[3]) if nl.ac else (1e6, 1e8)
    try:
        f = _np.logspace(_np.log10(a_), _np.log10(b_), 200)
        S = mna.build(nl).solve(f)
        return float(f[int(_np.argmax(_np.abs(S[:, 1, 0])))])
    except Exception:                                        # noqa: BLE001
        return float((a_ * b_) ** 0.5)


def _fmt(v, params=None):
    """A value as SPICE wants it -- as `{expr}`, or evaluated.

    `params` given means INLINE: write the number, not the reference.  That is
    what a tool other than vnafit actually produces, so it is what an import
    has to be tested against.
    """
    if hasattr(v, "src"):
        if params is None:
            return "{" + v.src + "}"
        from .units import resolve
        return f"{resolve(v, params):.10g}"
    return f"{float(v):.10g}"


def _element_lines(e, style, params, f_ref, xfmr="refuse", inline=False):
    from .units import resolve
    pv = params if inline else None
    if e.kind in ("R", "L", "C"):
        a, b = e.nodes[0], e.nodes[1]
        rq = None
        if f_ref and e.kind in ("L", "C") and ("q" in e.args or "tand" in e.args):
            try:
                q = (resolve(e.args["q"], params) if "q" in e.args
                     else 1.0 / resolve(e.args["tand"], params))
                v = resolve(e.args["value"], params)
                w = 2 * 3.141592653589793 * f_ref
                rq = (w * v / q) if e.kind == "L" else 1.0 / (w * v * q)
            except Exception:                                # noqa: BLE001
                rq = None
        if rq:
            mid = f"{e.name.lower()}_q"
            return [f"{e.name} {a} {mid} {_fmt(e.args['value'], pv)}",
                    f"R{e.name} {mid} {b} {rq:.6g}  ; vnafit Q="
                    f"{_fmt(e.args.get('q', 0), pv)} at f_ref"]
        return [f"{e.name} {a} {b} {_fmt(e.args['value'], pv)}"]
    if e.kind == "K":
        return [f"{e.name} {e.args['a']} {e.args['b']} {_fmt(e.args['k'], pv)}"]
    if e.kind == "TLIN":
        z0 = _fmt(e.args["z0"], pv)
        if "td" in e.args:
            return [f"{e.name} {' '.join(e.nodes)} Z0={z0} "
                    f"TD={_fmt(e.args['td'], pv)}"]
        return [f"{e.name} {' '.join(e.nodes)} Z0={z0} "
                f"LEN={_fmt(e.args['len'], pv)}"]
    if e.kind == "XFMR":
        # SPICE has no ideal transformer primitive.  The portable spelling is a
        # pair of coupled inductors with k -> 1, which is what most tools use.
        n = resolve(e.args["n"], params)
        # Two coupled inductors with k -> 1 behave as an n:1 transformer when
        # their reactance is large against everything around them.  "Large" is
        # a judgement, so it is written into the file with its reference.
        lsec = 50.0 * 40.0 / (2 * 3.141592653589793 * (f_ref or 1e8))
        lpri = lsec * n * n
        p1, n1, p2, n2 = e.nodes
        return [f"* {e.name}: ideal 1:{n:.6g} transformer approximated by two",
                f"* coupled inductors (reactance ~40x50 ohm at "
                f"{(f_ref or 1e8)/1e6:.3f} MHz).  NOT the same network.",
                f"L{e.name}a {p1} {n1} {lpri:.6g}",
                f"L{e.name}b {p2} {n2} {lsec:.6g}",
                f"K{e.name} L{e.name}a L{e.name}b 0.9999"]
    return [f"* {e.name}: no SPICE equivalent"]
