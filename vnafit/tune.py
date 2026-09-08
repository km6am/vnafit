"""Tuning advice that does not depend on where the fit parked the impedance.

The fitter can walk a long way along the impedance scale -- L up, every C down,
by the same factor -- because the response barely changes along it.  A readout
in picofarads is then worthless: a live fit came back asking for a 32 pF trimmer
to be changed to 301 pF, which was not a tuning error at all, only a scale.

Every quantity here is invariant along that direction, and that is checkable
rather than asserted:

    resonator f0 = 1/2pi sqrt(L*C)          L*C is unchanged
    coupling  k  = Cm / Ctot                a ratio of capacitances
    Ctot / Ctrim                            a ratio of capacitances
    unloaded  Q  = wL / R                   R tracks L under constant Q

so the fractional change a trimmer needs -- ((f/f_target)^2 - 1) * Ctot/Ctrim --
is a product of two invariants, and comes out the same whether the fit believes
in 30 nH or 3.6 nH.  Verified in tests/test_tune.py by scaling a netlist and
checking the advice does not move.

The absolute capacitance is still printed, taken from the TARGET's scale, which
is the one you can trust: it came off the schematic.
"""
import numpy as np

from . import roles, schematic
from .units import resolve


def _v(e, params, key="value"):
    return resolve(e.args[key], params)


def resonators(nl, params, lay=None, role=None):
    """Every parallel L||C node: its trimmer, its total C, and its frequency.

    The total includes what the couplings and end capacitors add to the node --
    getting that subtraction wrong is what detunes the ends by several percent
    and then reads as a coupling error.  An end capacitor works into the port
    impedance, so it loads the node by Cs/(1+(w*Cs*Z0)^2), which depends on the
    frequency it is helping to set; two passes settle it to well under a kHz.
    """
    lay = lay or schematic.plan(nl)
    role = role or roles.classify(nl, lay)
    ports = {p.pos for p in nl.ports} | {p.neg for p in nl.ports}
    z0 = nl.ports[0].z0 if nl.ports else 50.0

    legs = {}
    for i, chain in lay.legs:
        legs.setdefault(i, []).extend(chain)

    out = []
    for i, els in sorted(legs.items()):
        ind = [e for e in els if role.get(e.name) == "resonator_l"]
        trim = [e for e in els if role.get(e.name) == "resonator_c"]
        if not ind or not trim:
            continue
        L = _v(ind[0], params)
        ctrim = sum(_v(c, params) for c in trim)
        neighbours = [(e, j) for e, j in lay.series if i in (j, j + 1)]
        w = 1.0 / np.sqrt(L * ctrim)
        for _pass in range(3):
            ctot = ctrim
            for e, j in neighbours:
                if e.kind != "C":
                    continue
                cv = _v(e, params)
                other = lay.spine[j + 1 if i == j else j]
                ctot += (cv / (1.0 + (w * cv * z0) ** 2) if other in ports
                         else cv)
            w = 1.0 / np.sqrt(L * ctot)
        loss = [e for e in els if role.get(e.name) == "loss_r"]
        q = (w * L / _v(loss[0], params)) if loss else None
        if q is None and "q" in ind[0].args:
            q = resolve(ind[0].args["q"], params)
        out.append(dict(node=i, f0=w / (2 * np.pi), L=L, ctot=ctot,
                        ctrim=ctrim, trim=[c.name for c in trim], q=q))
    return out


def couplings(nl, params, res, lay=None, role=None):
    """k between adjacent resonators: Cm / sqrt(Ctot_i * Ctot_j)."""
    lay = lay or schematic.plan(nl)
    role = role or roles.classify(nl, lay)
    at = {r["node"]: r for r in res}
    out = []
    for e, j in lay.series:
        if role.get(e.name) != "coupling" or e.kind != "C":
            continue
        a, b = at.get(j), at.get(j + 1)
        if not (a and b):
            continue
        cm = _v(e, params)
        out.append((e.name, cm / np.sqrt(a["ctot"] * b["ctot"])))
    return out


def advice(nl, fitted, target):
    """What to turn, in numbers the impedance scale cannot move.

    Returns a list of lines.  A resonator's line says where it is, where it
    should be, and by what FRACTION its trimmer has to change to get there --
    all invariant.  The absolute picofarads come from the target.
    """
    lay = schematic.plan(nl)
    role = roles.classify(nl, lay)
    rf = resonators(nl, fitted, lay, role)
    rt = resonators(nl, target, lay, role)
    if not rf or len(rf) != len(rt):
        return []

    lines = ["resonator      now      target    turn"]
    for k, (a, b) in enumerate(zip(rf, rt), 1):
        need = ((a["f0"] / b["f0"]) ** 2 - 1.0) * a["ctot"] / a["ctrim"]
        want = b["ctrim"] * (1.0 + need)
        word = "" if abs(need) < 0.002 else ("  <- turn" if abs(need) > 0.01
                                             else "")
        lines.append(f"  {','.join(a['trim']):<6}{a['f0']/1e6:9.3f} "
                     f"{b['f0']/1e6:9.3f}  {100*need:+7.2f}%  "
                     f"{want*1e12:7.2f} pF{word}")

    cf, ct = couplings(nl, fitted, rf, lay, role), couplings(nl, target, rt, lay, role)
    for (n, a), (_m, b) in zip(cf, ct):
        lines.append(f"  {n:<6}k {a:8.5f}  {b:8.5f}  {100*(a/b-1):+7.2f}%"
                     f"   fixed part")
    qf = [r["q"] for r in rf if r["q"]]
    qt = [r["q"] for r in rt if r["q"]]
    if qf and qt:
        lines.append(f"  {'Qu':<6}  {np.mean(qf):8.1f}  {np.mean(qt):8.1f}  "
                     f"{100*(np.mean(qf)/np.mean(qt)-1):+7.2f}%   not tunable")
    return lines
