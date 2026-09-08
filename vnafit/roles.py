"""What each element is FOR, and which ones are therefore the same part.

An imported netlist has one parameter per element, because a SPICE file says
nothing about which components are one part.  That costs a great deal: the
LTspice export of the TinyFilter comes back with 13 parameters of which 2 are
measurable, against 7 of 7 for the hand-written version whose only difference is
that it says `cin` drives BOTH end capacitors and `l` drives all three
inductors.

No syntactic rule recovers that.  Merging elements with equal values was tried
and is wrong in the worst possible way: on this filter it merges `c1` and `c3`,
the two end TRIMMERS, and those are exactly the parts that differ once a real
board is built -- measured, 1044 kHz of spread across the three resonators of a
unit whose caps are nominally equal.  Merging by port-swapping symmetry makes
the same mistake with more machinery, and merging by correlation cannot
discriminate at all: 25 pairs of the imported netlist sit at |r| = 1.0000
because the soft impedance-scale direction dominates the covariance.

What does work is the element's ROLE in the ladder, which the layout already
computes:

  end capacitor      series, one side on a port node    fixed part, matched pair
  coupling element   series, between internal nodes     fixed part, matched set
  resonator inductor in a leg to ground beside a cap    one part number
  loss resistor      in the same leg as its inductor    one construction
  resonator capacitor in a leg to ground beside an L    ADJUSTABLE - never tied

The last line is the whole point.  Everything a designer buys as a matched set
gets tied; the thing they turn with a screwdriver does not, so a detuned
resonator stays visible -- which is what the tool is for.

This is a HEURISTIC about how filters are built, not a fact about the netlist.
It is only ever applied to values that are already equal, every group it makes
is reported with its reason, and `--no-group` turns it off.
"""
from . import schematic
from .schematic import GND, _endpoints

# Roles whose members are bought as a matched set, and the parameter name to
# give them.  A role missing from here is never tied.
#
# The single rule behind the list: an element that SETS A RESONATOR'S FREQUENCY
# and is the adjustable half of it never appears.  Everything else that a
# designer orders as one line on a BOM does.
GROUPABLE = {
    # bandpass, parallel resonators (TinyFilter, helical)
    "end": "cin", "coupling": "cm", "coupling_l": "lk",
    "resonator_l": "l", "loss_r": "qr",
    # bandpass, series resonators (crystal ladder, mesh)
    "mesh_l": "lm", "mesh_c": "cs", "shunt_coupling": "cc",
    # lowpass / highpass ladders -- symmetric by construction, no trimmers
    "series_l": "ls", "shunt_c": "cp", "series_c": "cser", "shunt_l": "lp",
    # elliptic: the capacitor across a series inductor
    "bridge_c": "cb", "bridge_l": "lb",
    # traps and stubs
    "trap_l": "lt", "stub": "zt", "tap": "n",
}

WHY = {
    "end": "series capacitors onto the ports -- a matched pair in every "
           "design; if they differ, that is a build fault, not tuning",
    "coupling": "coupling elements between resonators -- fixed parts chosen "
                "together",
    "coupling_l": "coupling inductors between resonators -- fixed parts chosen "
                  "together",
    "resonator_l": "resonator inductors -- one part number, wound the same way",
    "loss_r": "loss resistors on those inductors -- one construction, so one Q",
    "mesh_l": "series-resonator inductors -- one part number",
    "mesh_c": "series-resonator capacitors -- the fixed half of a mesh "
              "resonator (a crystal ladder's motional arm)",
    "shunt_coupling": "shunt coupling capacitors between mesh resonators",
    "series_l": "series arms of a ladder -- a Chebyshev or elliptic prototype "
                "is symmetric, so equal arms are the same part",
    "shunt_c": "shunt arms of a ladder -- symmetric by construction",
    "series_c": "series arms of a highpass ladder -- symmetric by construction",
    "shunt_l": "shunt arms of a highpass ladder -- symmetric by construction",
    "bridge_c": "capacitors across the series arms -- an elliptic prototype "
                "places its transmission zeros in symmetric pairs",
    "bridge_l": "inductors across the shunt arms",
    "trap_l": "trap inductors -- one part number",
    "stub": "transmission-line resonators -- cut from one piece of line",
    "tap": "the taps at each end -- wound the same way",
}

# Never tied, and each for the same reason: it is the half of a resonator that
# someone adjusts, so tying it hides exactly the fault the tool exists to find.
NEVER = ("resonator_c", "mesh_trim_c", "trap_c")


def _series_shape(nl, lay):
    """Read the series arms: mesh resonators, and elliptic bridges.

    Two shapes that both come back from the layout as ordinary series elements
    and are not:

    A MESH resonator (a crystal ladder) is an inductor and a capacitor in
    series along the spine.  The node between them belongs to nothing else, so
    the pair is one resonator even though it is drawn as two arms.  Looking for
    it OFF the spine failed -- the planner routes the spine straight through
    that node, because it is on the only path from port to port.

    An ELLIPTIC section is a capacitor ACROSS a series inductor -- the same two
    nodes, so the layout files it as a second series arm in the same slot, not
    as a bridge.  It is what places the transmission zero.
    """
    ports = {p.pos for p in nl.ports} | {p.neg for p in nl.ports}
    at_slot = {}
    for e, i in lay.series:
        at_slot.setdefault(i, []).append(e)
    legs_at = {}
    for i, chain in lay.legs:
        legs_at.setdefault(i, []).extend(chain)

    out = {}
    # parallel arms in one slot: L with a C across it
    for i, els in at_slot.items():
        if len(els) == 2 and {e.kind for e in els} == {"L", "C"}:
            for e in els:
                out[e.name] = "series_l" if e.kind == "L" else "bridge_c"
    # a bare node joining two series arms
    for i in sorted(at_slot):
        a = at_slot.get(i)
        b = at_slot.get(i + 1)
        if not a or not b or len(a) != 1 or len(b) != 1:
            continue
        node = lay.spine[i + 1]
        if node in ports or legs_at.get(i + 1):
            continue                       # the node goes somewhere else too
        ea, eb = a[0], b[0]
        if {ea.kind, eb.kind} == {"L", "C"}:
            out[ea.name] = "mesh_l" if ea.kind == "L" else "mesh_c"
            out[eb.name] = "mesh_l" if eb.kind == "L" else "mesh_c"
    return out


def classify(nl, lay=None):
    """element name -> role, from where it sits in the ladder.

    Covers the shapes an RF filter actually comes in: parallel-resonator
    bandpass (capacitively or magnetically coupled), series-resonator / mesh
    bandpass, lowpass and highpass ladders, elliptic sections with a capacitor
    across a series arm, series-resonant traps to ground, and stub resonators.
    """
    lay = lay or schematic.plan(nl)
    ports = {p.pos for p in nl.ports} | {p.neg for p in nl.ports}
    spine = lay.spine
    role = {}

    # --- what sits in each leg to ground -----------------------------------
    legs_at = {}
    for i, chain in lay.legs:
        legs_at.setdefault(i, []).append(chain)

    for i, chains in legs_at.items():
        kinds = [{e.kind for e in ch} for ch in chains]
        # PARALLEL resonator: an L and a C in DIFFERENT legs at one node.
        # SERIES-resonant trap: an L and a C in the SAME leg.  The first pass
        # tested `L in here and C in here` over the union of the node's legs
        # and so read a trap as a resonator.
        parallel = any("L" in k for k in kinds) and any("C" in k for k in kinds) \
            and not any({"L", "C"} <= k for k in kinds)
        for ch in chains:
            k = {e.kind for e in ch}
            trap = {"L", "C"} <= k
            for e in ch:
                if e.kind == "R":
                    role[e.name] = "loss_r" if len(ch) > 1 else "shunt_r"
                elif e.kind == "L":
                    role[e.name] = ("trap_l" if trap else
                                    "resonator_l" if parallel else "shunt_l")
                elif e.kind == "C":
                    role[e.name] = ("trap_c" if trap else
                                    "resonator_c" if parallel else "shunt_c")
                elif e.kind == "TLIN":
                    role[e.name] = "stub"
                else:
                    role[e.name] = "other"

    # --- series arms --------------------------------------------------------
    mesh = _series_shape(nl, lay)

    has_res = any(v in ("resonator_l", "resonator_c") for v in role.values())
    has_mesh = "mesh_l" in mesh.values()
    for e, i in lay.series:
        if e.name in mesh:
            role[e.name] = mesh[e.name]
            continue
        a, b = spine[i], spine[i + 1]
        onport = a in ports or b in ports
        if e.kind == "C":
            role[e.name] = ("end" if (onport and (has_res or has_mesh)) else
                            "coupling" if has_res else
                            "series_c")
        elif e.kind == "L":
            role[e.name] = "coupling_l" if has_res else "series_l"
        elif e.kind == "XFMR":
            role[e.name] = "tap"
        elif e.kind == "TLIN":
            role[e.name] = "line"
        else:
            role[e.name] = "other"

    # --- bridges: an element across a series arm (an elliptic section) ------
    for e, _i, _j in lay.bridges:
        role[e.name] = {"C": "bridge_c", "L": "bridge_l"}.get(e.kind, "bridge")

    # --- shunt capacitors between mesh resonators are the coupling ---------
    if has_mesh:
        for n, r in list(role.items()):
            if r == "shunt_c":
                role[n] = "shunt_coupling"

    for e in lay.leftover:
        role.setdefault(e.name, "other")
    for e in nl.elements:
        role.setdefault(e.name, "other")
    return role


def _value(nl, e):
    from .units import resolve
    v = e.args.get("value")
    return None if v is None else resolve(v, nl.resolve_params())


def suggest(nl, lay=None, rel=1e-6):
    """Groups of elements worth driving from one parameter.

    Returns [(name, [elements], why)].  Only elements that share a role AND
    already hold the same value: the heuristic decides which roles may be tied,
    never which values are equal.
    """
    lay = lay or schematic.plan(nl)
    role = classify(nl, lay)
    buckets = {}
    for e in nl.elements:
        r = role.get(e.name)
        if r not in GROUPABLE:
            continue
        v = _value(nl, e)
        if v is None or v <= 0:
            continue
        # NOT rounded.  Rounding the key to 1e-15 turned 8.877e-13 into
        # 8.88e-13, and the tolerance check below then compared a rounded key
        # against an unrounded value and failed by three orders of magnitude --
        # so picofarad parts never grouped while nanohenry ones did.
        key = (r, e.kind, float(v))
        placed = False
        for (kr, kk, kv) in list(buckets):
            if kr == r and kk == e.kind and abs(kv - v) <= rel * max(kv, v):
                buckets[(kr, kk, kv)].append(e)
                placed = True
                break
        if not placed:
            buckets[key] = [e]

    out, seen = [], {}
    for (r, _kind, _v), els in buckets.items():
        if len(els) < 2:
            continue
        seen[r] = seen.get(r, 0) + 1
        base = GROUPABLE[r]
        name = base if seen[r] == 1 else f"{base}{seen[r]}"
        out.append((name, sorted(els, key=lambda e: e.name), WHY[r]))
    return sorted(out, key=lambda g: g[0])


def apply(nl, groups):
    """Rewrite the netlist so each group is driven by one parameter."""
    from .units import Expr
    made = []
    for name, els, _why in groups:
        # Retire the members' own parameters FIRST.  Checking for a collision
        # before doing so meant a group of `Cin`/`Cout` could not be called
        # `cin`, because element Cin's promoted parameter was still there, and
        # it came out as `cin_`.
        value = _value(nl, els[0])
        for e in els:
            prev = e.args["value"]
            if isinstance(prev, Expr) and len(prev.names) == 1:
                nl.params.pop(next(iter(prev.names)), None)
        while name in nl.params or name in made:
            name += "_"
        nl.params[name] = value
        for e in els:
            e.args["value"] = Expr(name)
        made.append(name)
    return made


def group(nl, lay=None):
    """suggest + apply.  Returns the groups made, for reporting."""
    groups = suggest(nl, lay)
    apply(nl, groups)
    return groups
