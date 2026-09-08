"""The model card: a schematic with the fittable parameters drawn on it.

Three panels, and the top one is the point.  Each parameter OUTLINES the
elements it acts on, coloured to match its curve in the sensitivity panel
below.  Earlier versions wrote the parameter names as small coloured text under
each element; five parameters over four elements became an unreadable smear,
and it never showed the one thing worth seeing, which is a parameter's EXTENT
-- `k` touches one element, `f0` touches six.

**One box per ELEMENT, not one box per parameter.**  Drawing a single rectangle
round the bounding box of everything a parameter touches is wrong whenever
those elements are not adjacent, and on a ladder filter they almost never are:
`cin` drives the two end capacitors, so its bounding box swallowed all nine
elements in between and claimed nine that it does not act on.  Every parameter
of a symmetric filter is like this.  Stacking seven such rectangles, each
padded further out than the last, produced overlapping frames larger than the
schematic with labels colliding on top of it.  Tight outlines drawn once per
element are exact by construction, and nesting is then only as deep as the
number of parameters sharing ONE element -- two, in practice.

The names live in a key under the drawing rather than on it.  There is no room
beside a shunt leg (adjacent legs are 0.62 apart and a box is 0.34 wide) and
the space above a series element already holds its name, value and Q.

Elements whose value is held -- pinned gauges, and anything the netlist writes
as a literal -- get a dashed grey outline, so "no box" never has to be read as
"forgot to draw one".

The card is drawn in the coordinates that are actually determined.  Listing `al`
and `turns` as separate rows while also reporting that only `al*turns^2` is
determined contradicts itself, so the degenerate pair is collapsed into the
netlist's own name for the invariant before anything is drawn.
"""
import numpy as np

from . import schematic, structure

BOX_PAD, BOX_STEP = 0.05, 0.06
PINNED_COLOR = "#9aa3ad"


def param_colors(names):
    """One colour per parameter, shared between its box and its curve.

    Colouring boxes by verdict instead made every fittable one green, left the
    label as the only distinguisher, and wasted the chance to tie the two panels
    together.
    """
    import matplotlib.pyplot as plt
    cm = plt.cm.tab10(np.linspace(0, 1, 10))
    return {n: cm[i % 10] for i, n in enumerate(names)}


def _outline(ax, lay, e, depth, color, ls="-", lw=1.5, alpha=0.95):
    """Outline one element, in whatever shape actually fits it.

    A coupling is an arc between two inductors; the smallest rectangle holding
    it also holds everything drawn between them, and its edges run across both
    coils.  So an arc is outlined as an arc -- a wide soft stroke laid under the
    element's own line -- and only the box-shaped elements get boxes.
    """
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    shape = (lay.shapes or {}).get(e)
    if shape and shape[0] == "arc":
        _kind, a, b, rad = shape
        r = FancyArrowPatch(a, b, connectionstyle=f"arc3,rad={rad}",
                            arrowstyle="-", lw=4.5 + 2.5 * depth, ls="-",
                            color=color, alpha=0.35 * alpha, zorder=1,
                            capstyle="round")
        ax.add_patch(r)
        return r
    x0, y0, x1, y1 = lay.boxes[e]
    pad = BOX_PAD + BOX_STEP * depth
    r = FancyBboxPatch((x0 - pad, y0 - pad), (x1 - x0) + 2 * pad,
                       (y1 - y0) + 2 * pad,
                       boxstyle="round,pad=0.01,rounding_size=0.07",
                       fill=False, lw=lw, ls=ls, ec=color, alpha=alpha,
                       zorder=4)
    ax.add_patch(r)
    return r


def annotate(ax, st, lay, colors, only=None, pinned=(), dim=()):
    """Outline every element each parameter drives.

    Returns [(param, [elements], [rects], None)] so a caller can make them
    clickable -- one rect per element now, which is also what click-to-select
    wants: a hit test against the element you actually pointed at.

    `dim` names parameters to draw dashed and faint rather than solid: pinned,
    but still there.  Dropping their outline entirely, which is what the first
    interactive version did, left nothing on the drawing to click to get them
    back -- and made a click look as though it had deleted something.
    """
    boxes = lay.boxes or {}
    names = list(only if only is not None else st.names)
    dim = set(dim)
    per_el = {}
    for pn in names:
        for e in sorted(st.epar):
            if pn in st.epar[e] and e in boxes:
                per_el.setdefault(e, []).append(pn)

    drawn = []
    for pn in names:
        els = sorted(e for e, ps in per_el.items() if pn in ps)
        if not els:
            continue
        faint = pn in dim
        rects = [_outline(ax, lay, e, per_el[e].index(pn), colors[pn],
                          ls=((0, (3, 2)) if faint else "-"),
                          lw=(1.1 if faint else 1.5),
                          alpha=(0.45 if faint else 0.95))
                 for e in els]
        drawn.append((pn, els, rects, None))

    for e in sorted(pinned):
        if e in boxes:
            r = _outline(ax, lay, e, len(per_el.get(e, [])), PINNED_COLOR,
                         ls=(0, (2, 2)), lw=1.1, alpha=0.85)
            # Returned with param None so a caller can hit-test a HELD element
            # too.  Clicking one has to do something -- "this is pinned, and
            # here is why" -- rather than nothing.
            drawn.append((None, [e], [r], None))
    return drawn


def _key(ax, drawn, colors, pinned):
    """The colour key, under the drawing and out of its way."""
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=colors[pn], lw=2.6,
                ls=rects[0].get_linestyle() if rects else "-",
                alpha=rects[0].get_alpha() if rects else 1.0,
                label=f"{pn} \u2192 {','.join(els)}")
         for pn, els, rects, _t in drawn if pn is not None]
    if pinned:
        h.append(Line2D([], [], color=PINNED_COLOR, lw=1.6, ls=(0, (2, 2)),
                        label=f"held: {','.join(sorted(pinned))}"))
    if not h:
        return
    ax.legend(handles=h, loc="upper left", bbox_to_anchor=(0, -0.01),
              ncol=min(4, max(1, (len(h) + 1) // 2)), fontsize=7.6,
              frameon=False, handlelength=1.3, columnspacing=1.6,
              labelspacing=0.35, borderpad=0.0)


def draw(nl, f=None, sigma_db=0.05, fig=None, reduce=True):
    """Render the card.  Returns (fig, structure, notes, drawn_boxes)."""
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    st, notes = (structure.analyse_reduced(nl, f) if reduce
                 else (structure.analyse(nl, f), []))
    colors = param_colors(st.names)

    fig = fig or plt.figure(figsize=(11.5, 11.4))
    gs = GridSpec(3, 1, height_ratios=[3.6, 2.4, 2.6], hspace=0.66)

    ax0 = fig.add_subplot(gs[0])
    _f, _a, lay = schematic.draw(nl, ax=ax0, title=nl.title or "")
    fit = [n for n in st.names if st.verdict[n][0] == "determined"]
    # An element with a value but no outline is being HELD -- either a literal
    # in the netlist or a parameter pinned to break a gauge.  Say which, rather
    # than leaving a bare element to be read as an oversight.
    held = {e for e in (lay.boxes or {})
            if not any(p in st.epar.get(e, ()) for p in fit)}
    drawn = annotate(ax0, st, lay, colors, only=fit, pinned=held)
    _key(ax0, drawn, colors, held)
    lo0, hi0 = lay.ylim or ax0.get_ylim()
    ax0.set_ylim(lo0, max(hi0, 1.25))
    x0, x1 = ax0.get_xlim()
    ax0.set_xlim(x0 - 0.35, x1 + 0.35)
    ax0.text(0.0, 1.0, "outlined = free to fit, one outline per element; "
             "colour matches the curve below",
             transform=ax0.transAxes, fontsize=7.5, color="#5f6873",
             ha="left", va="top")

    ax1 = fig.add_subplot(gs[1])
    nf = len(st.f)
    for n in st.names:
        ax1.loglog(st.f / 1e6, np.maximum(st.sensitivity(n), 1e-9),
                   lw=2.0, color=colors[n], label=n)
    ax1.axhline(sigma_db, color="#c0442e", lw=1.0, ls="--")
    ax1.text(st.f[2] / 1e6, sigma_db * 1.25,
             f"trace noise {sigma_db:g} dB — a parameter is only determined "
             f"where its curve is above this",
             fontsize=7.5, color="#c0442e", va="bottom")
    ax1.set_xlabel("MHz")
    ax1.set_ylabel("|dS21| in dB per e-fold")
    ax1.set_ylim(max(1e-4, sigma_db / 50), None)
    ax1.set_title("Where each parameter acts  (parallel curves would be "
                  "degenerate; those have already been collapsed)",
                  fontsize=9.5, loc="left")
    ax1.grid(which="both", alpha=.22)
    ax1.legend(fontsize=8, ncol=4, loc="upper left", bbox_to_anchor=(0, -0.22))
    for sp in ("top", "right"):
        ax1.spines[sp].set_visible(False)

    ax2 = fig.add_subplot(gs[2])
    ax2.axis("off")
    body = st.report(sigma_db)
    if notes:
        # Wrapped, because a gauge note naming four parameters and explaining
        # why the choice is arbitrary runs to 250 characters and simply ran off
        # the right edge of the figure.
        import textwrap
        for n in notes:
            body += "\n" + "\n".join(textwrap.wrap(
                n, 96, initial_indent="  ", subsequent_indent="      "))
    body += "\n\n  suggested:  " + "  ".join(f".fit {n} ..." for n in fit)
    ax2.text(0.0, 1.0, body, va="top", ha="left", fontsize=8.6,
             family="DejaVu Sans Mono", transform=ax2.transAxes, linespacing=1.5)
    return fig, st, notes, drawn


def save(nl, path, **kw):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, st, notes, _d = draw(nl, **kw)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return st, notes
