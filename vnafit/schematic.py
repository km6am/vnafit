"""Draw a netlist as a schematic.

Automatic schematic layout is, in general, a hard research problem: a netlist
records connectivity and says nothing about position, so a general placer has to
invent an aesthetic.  Tools that try it on arbitrary circuits mostly produce
something worse than the netlist.

This does not attempt the general problem.  The circuits this tool exists for are
almost all **ladders**, and a ladder has an obvious drawing that everyone already
agrees on:

    a SPINE of series elements running left to right from port 1 to port 2,
    RUNGS of shunt elements hanging from spine nodes down to ground,
    and the occasional COUPLING or bridging element drawn as an arc.

So the layout is: find the spine by walking the ungrounded graph from port 1 to
port 2, hang everything that touches ground off the node it shares with the
spine, and arc anything left over.  That covers every filter anyone is likely to
type in here, and the heuristics are stated rather than tuned:

  1. The spine is the LONGEST simple path from port 1 to port 2 through
     elements that do not touch ground.  Longest, not shortest, and the
     difference matters: a filter's signal path goes through EVERY resonator,
     while a bridging cross-coupling is a shortcut across them.  Taking the
     shortest path draws a 3-pole cross-coupled filter as a 2-pole one with the
     middle resonator dumped in the leftover row.  For circuits this size the
     search is trivially cheap and it is capped anyway.
  2. Transformers and transmission lines bridge their two "+" nodes even though
     both windings touch ground, so they count as spine elements.
  3. Anything with one node on the spine and one at ground is a rung.
  4. Anything joining two non-adjacent spine nodes is a bridge, arced ABOVE the
     spine, which is where filter designers draw cross-couplings.
  5. `K` couplings also connect the ladder, because two magnetically coupled
     resonators sit side by side on the signal path even though no wire joins
     them.  Without that, a tapped helical pair has no ungrounded route from
     port to port at all and the whole circuit lands in the leftover row.  A
     spine link made of a coupling is drawn as a dashed arc rather than a
     series symbol, since there is no component in line there.
  6. Whatever is left is drawn in a leftover row with explicit net labels rather
     than being hidden.  A schematic that silently omits a component is worse
     than no schematic.

Rendered with matplotlib so the same code serves the GUI, a PNG, and a PDF.
"""
import math
from collections import deque

import numpy as np

from .netlist import GND
from .units import resolve

# Element value formatting: the unit each prefix belongs to.
UNIT = {"R": "Ω", "L": "H", "C": "F", "TLIN": "s", "XFMR": ""}
PREFIX = [(1e9, "G"), (1e6, "M"), (1e3, "k"), (1.0, ""), (1e-3, "m"),
          (1e-6, "µ"), (1e-9, "n"), (1e-12, "p"), (1e-15, "f")]


def eng(v, unit=""):
    """1.2e-11 -> '12 pF'.  Plain engineering notation, three significant
    figures, because a schematic is read at a glance."""
    if v == 0:
        return "0" + unit
    if not np.isfinite(v):
        return "?" + unit
    for scale, pre in PREFIX:
        if abs(v) >= scale * 0.999:
            x = v / scale
            txt = ("%.0f" % x) if abs(x) >= 100 else ("%.3g" % x)
            return f"{txt} {pre}{unit}".strip()
    return f"{v:.3g}{unit}"


class Layout:
    """Where every element goes.  Separated from the drawing so the placement
    can be tested without rendering anything."""

    gaps = frozenset()
    boxes = None          # element name -> (x0, y0, x1, y1) once drawn
    ylim = None           # the extent actually drawn, for callers to reuse
    grounds = None        # [(x, y)] every ground symbol placed
    shapes = None         # element name -> a non-rectangular outline, when a
                          # box would be wrong: ("arc", a, b, rad)

    def __init__(self, spine, series, shunt, bridges, couplings, leftover):
        self.spine = spine            # node names, left to right
        self.series = series          # [(element, i)] between spine[i], spine[i+1]
        self.shunt = shunt            # [(element, i)] every element in a leg
        self.legs = []                # [(i, [elements])] one leg to ground,
                                      # which may be a CHAIN (L then its ESR)
        self.bridges = bridges        # [(element, i, j)] spine[i] <-> spine[j]
        self.couplings = couplings    # [(K element, name_a, name_b)]
        self.leftover = leftover      # elements that did not fit the ladder

    def __repr__(self):
        return (f"<Layout spine={self.spine} series={len(self.series)} "
                f"shunt={len(self.shunt)} bridges={len(self.bridges)} "
                f"leftover={len(self.leftover)}>")


def _endpoints(e):
    """The two nodes an element spans for layout purposes.

    A transformer or a transmission line has four nodes and both of its ports
    touch ground, so connectivity alone would call it two shunt elements.  It is
    drawn as a bridge between its two "+" nodes instead, which is what it does
    to the signal.
    """
    if e.kind in ("XFMR", "TLIN"):
        return e.nodes[0], e.nodes[2]
    if e.kind == "K":
        return None
    return e.nodes[0], e.nodes[1]


def plan(nl):
    """Netlist -> Layout."""
    if len(nl.ports) < 2:
        raise ValueError("a schematic needs two ports to run a spine between")
    p1, p2 = nl.ports[0].pos, nl.ports[1].pos

    elems = [e for e in nl.elements if e.kind != "K"]
    ks = [e for e in nl.elements if e.kind == "K"]

    # Graph of ungrounded connections -- that is what the spine runs along.
    adj = {}

    def link(a, b, e):
        if a in GND or b in GND or a == b:
            return
        adj.setdefault(a, []).append((b, e))
        adj.setdefault(b, []).append((a, e))

    for e in elems:
        a, b = _endpoints(e)
        link(a, b, e)

    # Magnetic couplings connect the ladder too: coupled resonators are drawn
    # side by side even though no wire joins them.
    by_name_all = {e.name.lower(): e for e in nl.elements}
    for k in ks:
        ea = by_name_all.get(k.args["a"].lower())
        eb = by_name_all.get(k.args["b"].lower())
        if ea is None or eb is None:
            continue
        ha, hb = _hot(ea), _hot(eb)
        if ha and hb:
            link(ha, hb, k)

    spine = _longest_path(adj, p1, p2)
    if spine is None:
        # No ungrounded route: a shunt-only network, or the ports share a node.
        spine = [p1] if p1 == p2 else [p1, p2]

    pos = {n: i for i, n in enumerate(spine)}
    used, series, shunt, bridges, leftover = set(), [], [], [], []
    # A spine link made of a K coupling carries no in-line component.
    gaps = set()
    for i in range(len(spine) - 1):
        for nxt, e in adj.get(spine[i], []):
            if nxt == spine[i + 1] and e.kind == "K":
                gaps.add(i)

    for e in elems:
        a, b = _endpoints(e)
        if a in pos and b in pos:
            i, j = pos[a], pos[b]
            if abs(i - j) == 1:
                series.append((e, min(i, j)))
            else:
                bridges.append((e, min(i, j), max(i, j)))
            used.add(e.name)
        elif (a in pos and b in GND) or (b in pos and a in GND):
            shunt.append((e, pos[a] if a in pos else pos[b]))
            used.add(e.name)
    # A shunt LEG may be a chain, not one element: LTspice writes a lossy
    # inductor as `L1 n1 l1_q` + `RL1 l1_q 0`, two elements in series to ground
    # through a node of degree two.  Matching only single elements dropped every
    # inductor and every loss resistor of an imported netlist into `leftover`,
    # where nothing draws them -- six of thirteen components missing from the
    # picture, and no outline for the parameters driving them.
    deg = {}
    for e in elems:
        if e.name in used:
            continue
        for nd in _endpoints(e):
            deg.setdefault(nd, []).append(e)
    legs = []
    for e, i in shunt:
        legs.append((i, [e]))
    for start in list(pos):
        for e0 in list(deg.get(start, ())):
            if e0.name in used:
                continue
            chain, node, cur = [], start, e0
            while cur is not None and cur.name not in used:
                a, b = _endpoints(cur)
                nxt = b if a == node else a
                chain.append(cur)
                used.add(cur.name)
                if nxt in GND:
                    break
                # only continue through a node that goes nowhere else
                others = [x for x in deg.get(nxt, ()) if x.name not in used]
                if nxt in pos or len(others) != 1:
                    for c in chain:                  # not a leg after all
                        used.discard(c.name)
                    chain = []
                    break
                node, cur = nxt, others[0]
            if chain:
                legs.append((pos[start], chain))
    for e in elems:
        if e.name not in used:
            leftover.append(e)

    by_name = {e.name.lower(): e for e in nl.elements}
    couplings = [(k, by_name.get(k.args["a"].lower()), by_name.get(k.args["b"].lower()))
                 for k in ks]
    lay = Layout(spine, series, shunt, bridges, couplings, leftover)
    lay.gaps = gaps
    lay.legs = legs
    lay.shunt = [(e, i) for i, ch in legs for e in ch]
    return lay


def _longest_path(adj, src, dst, max_nodes=40, max_steps=200000):
    """Longest SIMPLE path from src to dst, by capped depth-first search.

    Exponential in the worst case, which is why it is capped -- but a filter
    netlist is a near-linear graph with a handful of nodes, so the search
    terminates immediately in practice.  If the cap is hit the shortest path is
    returned instead: a slightly wrong drawing beats a hang.
    """
    if src == dst:
        return [src]
    if len(adj) > max_nodes:
        return _shortest_path(adj, src, dst)
    best, steps = None, 0

    def walk(path, seen):
        nonlocal best, steps
        steps += 1
        if steps > max_steps:
            return False
        node = path[-1]
        if node == dst:
            if best is None or len(path) > len(best):
                best = list(path)
            return True
        for nxt, _e in adj.get(node, []):
            if nxt in seen:
                continue
            seen.add(nxt)
            path.append(nxt)
            if not walk(path, seen):
                path.pop()
                seen.discard(nxt)
                return False
            path.pop()
            seen.discard(nxt)
        return True

    ok = walk([src], {src})
    if best is None or not ok:
        return _shortest_path(adj, src, dst) if best is None else best
    return best


def _hot(e):
    """The non-ground node of a two-terminal element, for coupling layout."""
    for n in e.nodes[:2]:
        if n not in GND:
            return n
    return None


def _shortest_path(adj, src, dst):
    if src == dst:
        return [src]
    q, seen = deque([[src]]), {src}
    while q:
        path = q.popleft()
        for nxt, _e in adj.get(path[-1], []):
            if nxt in seen:
                continue
            if nxt == dst:
                return path + [nxt]
            seen.add(nxt)
            q.append(path + [nxt])
    return None


# ------------------------------------------------------------------- symbols
def _lead(ax, p, q, **kw):
    ax.plot([p[0], q[0]], [p[1], q[1]], color=kw.pop("color", "k"),
            lw=kw.pop("lw", 1.3), solid_capstyle="round", zorder=3, **kw)


def _frame(p, q, body=0.42):
    """Unit vectors and the two body ends, for a symbol drawn between p and q."""
    p, q = np.asarray(p, float), np.asarray(q, float)
    d = q - p
    L = np.hypot(*d)
    u = d / L
    n = np.array([-u[1], u[0]])
    a = p + u * (L - body * L) / 2
    b = q - u * (L - body * L) / 2
    return p, q, u, n, a, b, np.hypot(*(b - a))


def sym_resistor(ax, p, q, zig=6, h=0.085):
    p, q, u, n, a, b, L = _frame(p, q)
    _lead(ax, p, a)
    _lead(ax, b, q)
    xs = [a + u * (L * i / zig) + n * (h if i % 2 else -h) * (i not in (0, zig))
          for i in range(zig + 1)]
    ax.plot([v[0] for v in xs], [v[1] for v in xs], color="k", lw=1.3, zorder=3)


def sym_capacitor(ax, p, q, gap=0.055, plate=0.13):
    p, q, u, n, a, b, L = _frame(p, q, body=0.0)
    mid = (p + q) / 2
    for s in (-1, 1):
        c = mid + u * gap * s
        ax.plot([c[0] - n[0] * plate, c[0] + n[0] * plate],
                [c[1] - n[1] * plate, c[1] + n[1] * plate],
                color="k", lw=1.7, zorder=3)
    _lead(ax, p, mid - u * gap)
    _lead(ax, mid + u * gap, q)


def sym_inductor(ax, p, q, bumps=4, h=0.09):
    p, q, u, n, a, b, L = _frame(p, q)
    _lead(ax, p, a)
    _lead(ax, b, q)
    t = np.linspace(0, bumps * np.pi, 220)
    arc = np.abs(np.sin(t)) * h
    along = np.linspace(0, L, len(t))
    pts = a[None, :] + u[None, :] * along[:, None] + n[None, :] * arc[:, None]
    ax.plot(pts[:, 0], pts[:, 1], color="k", lw=1.3, zorder=3)


def sym_tline(ax, p, q, h=0.10):
    p, q, u, n, a, b, L = _frame(p, q, body=0.55)
    _lead(ax, p, a)
    _lead(ax, b, q)
    corners = [a + n * h, b + n * h, b - n * h, a - n * h]
    ax.add_patch(__import__("matplotlib.patches", fromlist=["Polygon"]).Polygon(
        corners, closed=True, fill=True, facecolor="white", edgecolor="k",
        lw=1.3, zorder=3))


def sym_ground(ax, p, w=0.13):
    x, y = p
    _lead(ax, (x, y), (x, y - 0.10))
    for i, s in enumerate((1.0, 0.62, 0.28)):
        ax.plot([x - w * s, x + w * s], [y - 0.10 - i * 0.055] * 2,
                color="k", lw=1.4, zorder=3)


def sym_port(ax, p, label, z0, side=+1):
    x, y = p
    ax.add_patch(__import__("matplotlib.patches", fromlist=["Circle"]).Circle(
        (x, y), 0.055, fill=True, facecolor="white", edgecolor="k", lw=1.3, zorder=4))
    ax.text(x + 0.10 * side, y + 0.20, label, ha="center", va="bottom",
            fontsize=9, fontweight="bold")
    ax.text(x + 0.10 * side, y + 0.06, f"{z0:g}Ω", ha="center", va="bottom",
            fontsize=7.5, color="#5f6873")


def sym_xfmr(ax, p, q, hh=0.13):
    """A transformer, as a compact boxed glyph: two windings across a core.

    Two earlier attempts are worth recording because both looked wrong for the
    same reason -- they tried to draw a physical transformer inline in a ladder
    spine, where there is no room for one.  Coils laid end to end read as a
    single long inductor with something odd in the middle; upright windings with
    ground stubs collided with the shunt legs on either side.

    A box says "this is a two-port block, not a coil", the glyph inside says
    which block, and the turns ratio is already printed above it.  It also
    distinguishes cleanly from a transmission line, which is a plain box.
    """
    from matplotlib.patches import FancyBboxPatch
    p, q, u, n, a, b, L = _frame(p, q, body=0.44)
    y = p[1]
    x0, x1 = a[0], b[0]
    _lead(ax, p, a)
    _lead(ax, b, q)
    ax.add_patch(FancyBboxPatch((x0, y - hh), x1 - x0, 2 * hh,
                                boxstyle="round,pad=0.01,rounding_size=0.03",
                                fill=True, facecolor="white", edgecolor="k",
                                lw=1.3, zorder=3))
    w = x1 - x0
    for xc, sgn in ((x0 + 0.30 * w, -1.0), (x1 - 0.30 * w, +1.0)):
        t = np.linspace(0, 2 * np.pi, 90)
        ys = y - hh * 0.62 + (t / (2 * np.pi)) * 2 * hh * 0.62
        ax.plot(xc + sgn * 0.030 * np.abs(np.sin(t)), ys,
                color="k", lw=1.1, zorder=4)
    for off in (-0.017, 0.017):
        xc = (x0 + x1) / 2 + off
        ax.plot([xc, xc], [y - hh * 0.66, y + hh * 0.66], color="k", lw=1.0,
                zorder=4)


SYMBOL = {"R": sym_resistor, "C": sym_capacitor, "L": sym_inductor,
          "TLIN": sym_tline, "XFMR": sym_xfmr}


def _label(e, params):
    """Name plus the EVALUATED value.

    I first had this print the expression instead, reasoning that
    `{1/(4*pi*pi*f0*f0*L)}` says "tuned to f0" where `3.63 pF` does not.  Drawing
    it settled the argument: at schematic type size the expression is an
    unreadable smear that collides with its neighbour, and a schematic's job is
    to answer "what is this circuit" at a glance.  The netlist is right there
    for intent.
    """
    kind = e.kind
    if kind == "XFMR":
        try:
            return e.name, f"1:{resolve(e.args['n'], params):.3g}", ""
        except Exception:                                    # noqa: BLE001
            return e.name, "1:n", ""
    if kind == "TLIN":
        try:
            z0 = resolve(e.args["z0"], params)
            td = resolve(e.args.get("td", 0.0), params) if "td" in e.args else None
            return e.name, f"{z0:g}Ω" + (f" {eng(td,'s')}" if td else ""), ""
        except Exception:                                    # noqa: BLE001
            return e.name, "", ""
    try:
        txt = eng(resolve(e.args.get("value"), params), UNIT.get(kind, ""))
    except Exception:                                        # noqa: BLE001
        txt = "?"
    extra = ""
    q = e.args.get("q")
    if q is not None:
        try:
            extra = f"Q={resolve(q, params):g}"
        except Exception:                                    # noqa: BLE001
            pass
    return e.name, txt, extra


def draw(nl, ax=None, title=None, params=None):
    """Render `nl` onto a matplotlib axis.  Returns (fig, ax, layout)."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch

    lay = plan(nl)
    params = params if params is not None else nl.resolve_params()
    n = max(len(lay.spine) - 1, 1)
    # A boxed block (transformer, transmission line) is 0.44 of the segment
    # wide against a capacitor's 0.11, so at dx=1.6 its outline reached into
    # the shunt leg at the next node.  Widen the whole ladder rather than
    # shrink the glyph, which would have to stop enclosing itself.
    dx = 1.9 if any(e.kind in ("XFMR", "TLIN") for e, _i in lay.series) else 1.6
    y0 = 0.0
    xs = {i: 0.9 + i * dx for i in range(len(lay.spine))}

    if ax is None:
        w = max(6.0, 1.6 + dx * n + 1.8)
        h = 3.6 + (0.42 if lay.leftover else 0) + (0.4 if lay.couplings else 0)
        fig, ax = plt.subplots(figsize=(w, h))
    else:
        fig = ax.figure

    # Where every element ended up on the page.  Recorded rather than
    # recomputed by callers, because anything that wants to annotate the drawing
    # -- highlighting which elements a fitted parameter touches, say -- needs the
    # same geometry the drawing used, and a second copy of it would drift.
    boxes = {}
    shapes = {}
    grounds = []          # every ground symbol drawn, so it can be asserted on

    # spine wire
    _lead(ax, (xs[0], y0), (xs[len(lay.spine) - 1], y0), lw=1.3)

    # An elliptic section puts a capacitor ACROSS a series inductor: two
    # elements between the same pair of nodes.  Drawn on the spine they land on
    # top of each other, symbols and labels both -- "340 nH" and "33 pF"
    # overprinted into an unreadable smear.  The second and later arms of a slot
    # are lifted onto a branch above the spine, which is where a textbook draws
    # them anyway.
    slot_n, slot_seen = {}, {}
    for e, i in lay.series:
        slot_n[i] = slot_n.get(i, 0) + 1
    PARALLEL_H = 0.95

    for e, i in lay.series:
        k = slot_seen.get(i, 0)
        slot_seen[i] = k + 1
        yb = y0 + PARALLEL_H * k
        if k:
            _lead(ax, (xs[i], y0), (xs[i], yb), lw=1.2)
            _lead(ax, (xs[i + 1], y0), (xs[i + 1], yb), lw=1.2)
            _lead(ax, (xs[i], yb), (xs[i + 1], yb), lw=1.2)
        p, q = (xs[i], yb), (xs[i + 1], yb)
        SYMBOL.get(e.kind, sym_resistor)(ax, p, q)
        name, val, extra = _label(e, params)
        mid = ((p[0] + q[0]) / 2, yb)
        # The SYMBOL, not the whole node-to-node segment.  An outline drawn on
        # the segment reaches halfway to the neighbouring node and appears to
        # claim the shunt leg hanging off it: `cin`'s outline round Cin looked
        # like it enclosed L1 and C1 as well.
        # A capacitor's plates are a tenth of the segment; a resistor or
        # inductor body is 0.42 of it.  One width for both left the capacitor
        # outlines touching the shunt legs either side of them.
        half = {"C": 0.10, "XFMR": 0.235, "TLIN": 0.235}.get(
            e.kind, 0.215) * (q[0] - p[0])
        boxes[e.name] = (mid[0] - half, yb - 0.16, mid[0] + half, yb + 0.16)
        ax.text(mid[0], mid[1] + 0.24, name, ha="center", fontsize=8.5, fontweight="bold")
        ax.text(mid[0], mid[1] + 0.40, val, ha="center", fontsize=7.5, color="#333")
        if extra:
            ax.text(mid[0], mid[1] + 0.54, extra, ha="center", fontsize=7,
                    color="#5f6873")

    # Shunt legs are fanned out sideways when several share a node.  A parallel
    # L||C resonator is the commonest thing in this whole tool, and drawing both
    # on the same vertical line stacks the symbols and their labels on top of
    # each other -- which is what the first version did.
    #
    # A LEG may hold more than one element.  LTspice writes a lossy inductor as
    # an inductor to a private node and a resistor from there to ground, so an
    # imported netlist has two-element legs everywhere; drawing only the first
    # element of each left six of thirteen components off the page.
    from collections import defaultdict
    # 0.90 keeps a single-element leg exactly where it was before
    # chains existed, so nothing that already looked right moved.
    SEG, GAP = 0.90, 0.30
    at_node = defaultdict(list)
    for idx, chain in lay.legs:
        at_node[idx].append(chain)

    def _depth(k):
        return 0.12 + k * SEG + (k - 1) * GAP

    deepest = max((len(c) for _i, c in lay.legs), default=1)
    bot_y = y0 - _depth(deepest)          # all grounds on one line
    label_y = bot_y - 0.34

    shunt_xy = {}
    for i, group in at_node.items():
        m = len(group)
        span = 0.62 * (m - 1)
        for j, chain in enumerate(group):
            x = xs[i] - span / 2 + 0.62 * j
            if m > 1:                       # tie the fanned legs back to the node
                _lead(ax, (xs[i], y0), (x, y0), lw=1.2)
            # CENTRED in the run, so a one-element leg is a symbol with wire
            # above and below rather than a symbol with a very long wire under
            # it.  Stretching the outline down to the common ground line to
            # cover that wire made a plain shunt capacitor's box as tall as the
            # whole leg -- taller than the two-element leg beside it, and it
            # looked like it enclosed its neighbour.
            k = len(chain)
            run = (y0 - 0.12) - bot_y
            body = k * SEG + (k - 1) * GAP
            top = y0 - 0.12 - max(0.0, (run - body) / 2)
            _lead(ax, (x, y0), (x, top), lw=1.2)
            for e in chain:
                bot = top - SEG
                SYMBOL.get(e.kind, sym_resistor)(ax, (x, top), (x, bot))
                boxes[e.name] = (x - 0.17, bot - 0.05, x + 0.17, top + 0.05)
                shunt_xy[e.name.lower()] = (x, (top + bot) / 2)
                top = bot - GAP
                if e is not chain[-1]:
                    _lead(ax, (x, bot), (x, top), lw=1.2)
            _lead(ax, (x, bot), (x, bot_y), lw=1.2)
            sym_ground(ax, (x, bot_y))
            grounds.append((x, bot_y))
            # Staggered between neighbouring legs at one node.  Two centred
            # value labels 0.62 apart are wider than the gap between them, so
            # "34.5 pF" and "30 nH" ran into each other; dropping every other
            # column separates them without spreading the ladder out.
            for k, e in enumerate(chain):
                name, val, extra = _label(e, params)
                ty = label_y - k * 0.46 - (0.30 if j % 2 else 0.0)
                ax.text(x, ty - 0.16, name, ha="center", fontsize=8.5,
                        fontweight="bold")
                ax.text(x, ty - 0.32, val, ha="center", fontsize=7, color="#333")
                if extra:
                    ax.text(x, ty - 0.46, extra, ha="center", fontsize=6.8,
                            color="#5f6873")
        ax.plot([xs[i]], [y0], marker="o", ms=3.4, color="k", zorder=5)

    # bridges arc above; cross-couplings are drawn where designers draw them
    for e, i, j in lay.bridges:
        a, b = (xs[i], y0 + 0.08), (xs[j], y0 + 0.08)
        boxes[e.name] = (xs[i], y0, xs[j], y0 + 0.55 + 0.12 * (j - i))
        rad = -(0.30 + 0.06 * (j - i))     # negative arcs UPWARD going left->right
        shapes[e.name] = ("arc", a, b, rad)
        ax.add_patch(FancyArrowPatch(a, b, connectionstyle=f"arc3,rad={rad}",
                                     arrowstyle="-", lw=1.2, color="#7a3fa0",
                                     zorder=2))
        name, val, _x = _label(e, params)
        ax.text((a[0] + b[0]) / 2, y0 + 0.55 + 0.12 * (j - i),
                f"{name}  {val}", ha="center", fontsize=7.5, color="#7a3fa0")

    # K couplings: dashed, below, between the two inductors
    place = dict(shunt_xy)
    for e, i in lay.series:
        place.setdefault(e.name.lower(), ((xs[i] + xs[i + 1]) / 2, y0 - 0.10))
    for k, ea, eb in lay.couplings:
        if ea is None or eb is None:
            continue
        a, b = place.get(ea.name.lower()), place.get(eb.name.lower())
        if a is None or b is None:
            continue
        if a[0] > b[0]:
            a, b = b, a
        # A rectangle spanning the two coupled inductors necessarily contains
        # whatever sits between them -- on a helical that is the whole of C1 --
        # and its edges run across both coils.  The coupling IS the arc, so
        # outline the arc.
        boxes[k.name] = (a[0], min(a[1], b[1]) - 0.34, b[0], max(a[1], b[1]))
        shapes[k.name] = ("arc", a, b, 0.22)
        ax.add_patch(FancyArrowPatch(a, b, connectionstyle="arc3,rad=0.22",
                                     arrowstyle="-", lw=1.2, ls="--",
                                     color="#c0442e", zorder=2))
        try:
            kv = "%.4g" % resolve(k.args["k"], params)
        except Exception:                                    # noqa: BLE001
            kv = "?"
        ax.text((a[0] + b[0]) / 2, min(a[1], b[1]) - 0.30,
                f"{k.name}  k={kv}", ha="center", va="center", fontsize=7.5,
                color="#c0442e",
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none"))

    # A port is a PAIR of terminals, and the drawing showed one.  P1 and P2 sat
    # on the spine as bare circles with nothing to say what the 50 ohm was
    # measured against -- the return, which is ground in every ordinary
    # measurement, was simply absent from a schematic full of ground symbols.
    for idx, (x, port) in enumerate(((xs[0], nl.ports[0]),
                                     (xs[len(lay.spine) - 1], nl.ports[1]))):
        sym_port(ax, (x, y0), f"P{port.index}", port.z0, side=-1 if idx else 1)
        if port.neg in GND:
            _lead(ax, (x, y0 - 0.055), (x, y0 - 0.50), lw=1.2)
            sym_ground(ax, (x, y0 - 0.50))
            grounds.append((x, y0 - 0.50))
        else:
            # A differential port: name the node rather than imply ground.
            ax.plot([x], [y0 - 0.50], marker="o", ms=5.5, mfc="white",
                    mec="k", mew=1.3, zorder=4)
            _lead(ax, (x, y0 - 0.055), (x, y0 - 0.50), lw=1.2)
            ax.text(x + 0.10, y0 - 0.62, port.neg, fontsize=7.5,
                    color="#5f6873", ha="left", va="top")

    if lay.leftover:
        # Never silently omit a component.  Anything the ladder could not place
        # is listed with its nets, so the drawing stays honest.
        txt = "not drawn in the ladder: " + ", ".join(
            f"{e.name}({'-'.join(e.nodes)})" for e in lay.leftover)
        ax.text(0.9, y0 - 1.80, txt, fontsize=7.5, color="#c0442e", ha="left")

    ax.set_xlim(0.1, xs[len(lay.spine) - 1] + 0.9)
    # Derived from what was actually drawn, not from a pair of constants.  With
    # chained legs the labels run as deep as the chain is long, and a fixed
    # -2.15 left them outside the axes -- which put anything a caller placed
    # "below the axes", such as the colour key, on top of them.
    fan = any(len(v) > 1 for v in at_node.values())
    lowest = label_y - 0.46 * max(len(c) for _i, c in lay.legs) - 0.48 \
        if lay.legs else y0 - 1.0
    if lay.leftover:
        lowest = min(lowest, y0 - 1.98)
    tallest = max(slot_n.values(), default=1) - 1
    ax.set_ylim(min(lowest, y0 - (1.9 if fan else 1.5)),
                y0 + PARALLEL_H * tallest + (1.45 if lay.bridges else 1.05))
    lay.ylim = ax.get_ylim()
    ax.set_aspect("equal")
    ax.axis("off")
    if title or nl.title:
        ax.set_title(title or nl.title, fontsize=10, loc="left")
    lay.boxes = boxes
    lay.shapes = shapes
    lay.grounds = grounds
    return fig, ax, lay


def save(nl, path, **kw):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax, lay = draw(nl, **kw)
    fig.tight_layout()
    fig.savefig(path, dpi=kw.pop("dpi", 150), bbox_inches="tight")
    plt.close(fig)
    return lay
