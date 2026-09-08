"""SPICE-subset netlist parsing.

Supported lines (case-insensitive, SPICE order of fields):

    * comment                     ; also a comment
    .param f0=145.4Meg L=330n     parameters, may reference earlier ones
    .port 1 in 0 Z0=50            port n, + node, - node   (Z0 optional, default 50)
    .fit L 100n 50n 800n          free parameter: name, seed, lo, hi
    .ac lin 401 100Meg 200Meg     default sweep (the caller may override)

    R1 in  out 50                 resistor
    C1 out 0   {1/(4*pi*pi*f0*f0*L)}
    L1 out 0   330n Q=200         inductor, optional unloaded Q at f0
    K1 L1 L2   0.021              mutual coupling between two named inductors
    T1 p1 n1 p2 n2 Z0=100 TD=1.7n     transmission line (or LEN=..., VF=...)
    X1 a b  n=3.2                 ideal transformer, V(a)=n*V(b)

    + continuation of the previous line

Element values may be a number with a SPICE suffix (`330n`, `1Meg`, `4.7k`), a
bare parameter name, or a braced expression `{...}` over the parameters.

Deliberately NOT supported: nonlinear devices, sources other than the ports,
time-domain analysis.  This engine is a linear AC solver for passive RF
networks; anything else belongs in ngspice.
"""
import os
import re

from .units import Expr, numeric, resolve, value

GND = {"0", "gnd", "GND", "ground"}


class NetlistError(ValueError):
    """A netlist the parser can name a problem in, with a line number."""

    def __init__(self, msg, lineno=None, text=None):
        self.lineno, self.text = lineno, text
        where = f"line {lineno}: " if lineno else ""
        extra = f"\n    {text}" if text else ""
        super().__init__(f"{where}{msg}{extra}")


class Element:
    __slots__ = ("kind", "name", "nodes", "args", "lineno")

    def __init__(self, kind, name, nodes, args, lineno=0):
        self.kind, self.name, self.nodes = kind, name, tuple(nodes)
        self.args, self.lineno = args, lineno

    def __repr__(self):
        return f"<{self.kind} {self.name} {self.nodes} {self.args}>"


class Port:
    __slots__ = ("index", "pos", "neg", "z0")

    def __init__(self, index, pos, neg, z0):
        self.index, self.pos, self.neg, self.z0 = index, pos, neg, z0

    def __repr__(self):
        return f"<Port {self.index} {self.pos}/{self.neg} Z0={self.z0}>"


class FitSpec:
    """One `.fit` line: a parameter that may be turned loose, and its bounds.

    Declaring a parameter here does NOT mean it is identifiable -- see
    `identify.py`.  It means the user is asserting the netlist has structure
    that could constrain it.
    """

    __slots__ = ("param", "seed", "lo", "hi")

    def __init__(self, param, seed, lo, hi):
        self.param, self.seed, self.lo, self.hi = param, seed, lo, hi

    def __repr__(self):
        return f"<fit {self.param} {self.seed} [{self.lo}, {self.hi}]>"


class Netlist:
    def __init__(self):
        self.elements = []
        self.ports = []
        self.params = {}        # name -> float | Expr, in declaration order
        self.fits = []
        self.ac = None          # (kind, n, start, stop) or None
        self.title = ""
        self.source = ""

    # ------------------------------------------------------------ parameters
    def resolve_params(self, overrides=None):
        """Evaluate .param in declaration order, with optional overrides.

        Overrides win over the file, so a fitter can drive the netlist without
        rewriting it.  Order matters: a later .param may reference an earlier
        one, and an override of the earlier one propagates.
        """
        out = {}
        over = dict(overrides or {})
        for name, raw in self.params.items():
            if name in over:
                out[name] = float(over[name])
                continue
            try:
                out[name] = resolve(raw, out)
            except ValueError as e:
                raise NetlistError(f".param {name}: {e}") from None
        # An override for something that was never declared is almost always a
        # typo in a fit spec; saying so beats silently ignoring it.
        unknown = set(over) - set(self.params)
        if unknown:
            raise NetlistError(f"override(s) for undeclared .param: "
                               f"{', '.join(sorted(unknown))}")
        return out

    def __repr__(self):
        return (f"<Netlist {self.title!r} {len(self.elements)} elements, "
                f"{len(self.ports)} ports>")


# --------------------------------------------------------------------- parse
_KV = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)=(.*)$")


def _split(line):
    """Tokenise, honouring braces so `{1/(2*pi)}` survives as one token."""
    toks, cur, depth = [], "", 0
    for ch in line:
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        if depth == 0 and ch in " \t,":
            if cur:
                toks.append(cur)
                cur = ""
            continue
        cur += ch
    if cur:
        toks.append(cur)
    if depth:
        raise NetlistError("unbalanced { } in line")
    return toks


def _kwargs(toks):
    """Pull `key=value` tokens out, returning (positional, keyword)."""
    pos, kw = [], {}
    for t in toks:
        m = _KV.match(t)
        if m:
            kw[m.group(1).lower()] = m.group(2)
        else:
            pos.append(t)
    return pos, kw


def _logical_lines(text):
    """Strip comments, join `+` continuations, yield (lineno, text)."""
    out = []
    for i, raw in enumerate(text.splitlines(), 1):
        s = raw.split(";", 1)[0]
        if s.lstrip().startswith("*"):
            s = ""
        s = s.strip()
        if not s:
            continue
        if s.startswith("+"):
            if not out:
                raise NetlistError("continuation '+' with nothing to continue", i, raw)
            out[-1] = (out[-1][0], out[-1][1] + " " + s[1:].strip())
        else:
            out.append((i, s))
    return out


def parse(text, source=""):
    """Parse netlist text into a `Netlist`."""
    nl = Netlist()
    nl.source = source
    lines = _logical_lines(text)
    if not lines:
        raise NetlistError("empty netlist")

    # SPICE convention, followed exactly: the first line is the TITLE.  The only
    # exception is a leading "." directive, so that a netlist fragment written
    # without a title still works.
    #
    # An earlier version tried to be clever and only took the first line as a
    # title when it did not "look like an element".  That heuristic failed on
    # the first real-world file it met: the title "K9DP QRP BCI filter" was read
    # as a K (mutual coupling) element, because it starts with K and has three
    # or more fields, and the parser then complained that "QRP" is not an
    # inductor.  Guessing at the author's intent is worse than following the
    # convention everyone already knows.
    first = lines[0][1]
    if not first.startswith("."):
        nl.title = first
        lines = lines[1:]

    for lineno, s in lines:
        try:
            _parse_line(nl, s, lineno)
        except NetlistError:
            raise
        except ValueError as e:
            raise NetlistError(str(e), lineno, s) from None
    _validate(nl)
    return nl


def _parse_line(nl, s, lineno):
    toks = _split(s)
    head = toks[0]
    low = head.lower()

    if low.startswith("."):
        return _parse_directive(nl, low, toks[1:], lineno, s)

    kind = head[0].upper()
    pos, kw = _kwargs(toks[1:])

    if kind in "RLC":
        if len(pos) < 3:
            raise NetlistError(f"{head}: expected `{head} n+ n- value`", lineno, s)
        args = {"value": numeric(pos[2])}
        if "q" in kw:
            args["q"] = numeric(kw["q"])
        if "tand" in kw:
            args["tand"] = numeric(kw["tand"])
        nl.elements.append(Element(kind, head, (pos[0], pos[1]), args, lineno))

    elif kind == "K":
        if len(pos) < 3:
            raise NetlistError(f"{head}: expected `{head} Lx Ly k`", lineno, s)
        nl.elements.append(Element("K", head, (), {
            "a": pos[0], "b": pos[1], "k": numeric(pos[2])}, lineno))

    elif kind == "T":
        if len(pos) < 4:
            raise NetlistError(f"{head}: expected `{head} p1 n1 p2 n2 Z0=.. TD=..`",
                               lineno, s)
        args = {"z0": numeric(kw.get("z0", 50.0))}
        if "td" in kw:
            args["td"] = numeric(kw["td"])
        elif "len" in kw:
            args["len"] = numeric(kw["len"])
            args["vf"] = numeric(kw.get("vf", 1.0))
        else:
            raise NetlistError(f"{head}: needs TD= or LEN=", lineno, s)
        for opt in ("loss", "q"):
            if opt in kw:
                args[opt] = numeric(kw[opt])
        nl.elements.append(Element("TLIN", head, tuple(pos[:4]), args, lineno))

    elif kind == "X":
        if len(pos) < 4:
            raise NetlistError(f"{head}: expected `{head} p1 n1 p2 n2 n=ratio`",
                               lineno, s)
        if "n" not in kw:
            raise NetlistError(f"{head}: ideal transformer needs n=", lineno, s)
        nl.elements.append(Element("XFMR", head, tuple(pos[:4]),
                                   {"n": numeric(kw["n"])}, lineno))
    else:
        raise NetlistError(f"unsupported element {head!r}", lineno, s)


def _parse_directive(nl, low, rest, lineno, s):
    if low == ".param":
        _, kw = _kwargs(rest)
        if not kw:
            raise NetlistError(".param: expected name=value", lineno, s)
        for k, v in kw.items():
            nl.params[k] = numeric(v)

    elif low == ".port":
        pos, kw = _kwargs(rest)
        if len(pos) < 3:
            raise NetlistError(".port: expected `.port index n+ n- [Z0=50]`",
                               lineno, s)
        nl.ports.append(Port(int(value(pos[0])), pos[1], pos[2],
                             float(value(kw.get("z0", 50.0)))))

    elif low == ".fit":
        pos, _ = _kwargs(rest)
        if len(pos) != 4:
            raise NetlistError(".fit: expected `.fit param seed lo hi`", lineno, s)
        nl.fits.append(FitSpec(pos[0].lower(), value(pos[1]),
                               value(pos[2]), value(pos[3])))

    elif low == ".ac":
        pos, _ = _kwargs(rest)
        if len(pos) != 4:
            raise NetlistError(".ac: expected `.ac lin|dec n start stop`", lineno, s)
        nl.ac = (pos[0].lower(), int(value(pos[1])), value(pos[2]), value(pos[3]))

    elif low in (".end", ".ends", ".title"):
        if low == ".title" and rest:
            nl.title = " ".join(rest)
    else:
        raise NetlistError(f"unsupported directive {low!r}", lineno, s)


def _validate(nl):
    if not nl.ports:
        raise NetlistError("no .port lines -- nothing to compute S-parameters between")
    idx = [p.index for p in nl.ports]
    if sorted(idx) != list(range(1, len(idx) + 1)):
        raise NetlistError(f"ports must be numbered 1..N with no gaps, got {sorted(idx)}")
    nl.ports.sort(key=lambda p: p.index)

    names = {e.name.lower() for e in nl.elements}
    if len(names) != len(nl.elements):
        seen, dup = set(), set()
        for e in nl.elements:
            (dup if e.name.lower() in seen else seen).add(e.name.lower())
        raise NetlistError(f"duplicate element name(s): {', '.join(sorted(dup))}")

    ind = {e.name.lower() for e in nl.elements if e.kind == "L"}
    for e in nl.elements:
        if e.kind == "K":
            for side in ("a", "b"):
                if e.args[side].lower() not in ind:
                    raise NetlistError(f"{e.name}: {e.args[side]!r} is not an inductor",
                                       e.lineno)
            if e.args["a"].lower() == e.args["b"].lower():
                raise NetlistError(f"{e.name}: an inductor cannot couple to itself",
                                   e.lineno)

    for f in nl.fits:
        if f.param not in nl.params:
            raise NetlistError(f".fit {f.param}: no such .param")
        if not (f.lo < f.hi):
            raise NetlistError(f".fit {f.param}: lo must be below hi")


def load(path):
    with open(path, "r") as fh:
        return parse(fh.read(), source=os.path.abspath(path))
