# The solver

A vectorised modified-nodal-analysis engine with no SPICE binary behind it,
and the drawing built on the same layout.

## How the solver works

The whole sweep is **one** `np.linalg.solve` on a stacked `(nfreq, dim, dim)`
complex matrix. Two choices make that work:

- **Affine assembly.** `A(ω) = G + jω·W` with `G`, `W` frequency-independent, so the
  element loop runs once per parameter vector rather than once per frequency. Only
  transmission lines break the form.
- **Norton ports.** A port is a current injection plus a conductance, not a
  voltage-source branch. Purely nodal, works for a floating differential port, keeps
  `A` complex-symmetric, and — the point — makes `A` identical for every excitation,
  so one factorisation serves all ports.

Values are normalised at build time to the reference impedance and the sweep centre,
so every passband matrix entry is O(1) and the unit-mismatch part of the condition
number disappears at zero per-evaluation cost. What remains is physics: near a
high-Q resonance the matrix is genuinely near-singular, `cond ≈ Q_L`.

### Speed

Measured, not estimated (see `tests/test_speed.py`): **1.7 ms** for a 401-point
sweep at dim 20, **0.43 ms** at dim 10. A 3-pole netlist is dim ~10–14.

This refuted the planning estimate of 10–15 ms by about 6×, and the architecture
changed because of it: netlist fitting is interactive after all, so it does not have
to be pushed offline behind the coupling-matrix model.

## What the engine refuses to do

Each of these is a real failure mode that returns a plausible wrong answer if you
let it through, so the engine raises instead:

- **An indefinite coupling matrix.** `k12 = k13 = +0.9` with `k23 = −0.9` has every
  pairwise `|k| < 1`, and describes an energy-generating network. Positive
  definiteness of the inductance matrix is the real condition; without the check the
  engine cheerfully returns `|S21| > 1`.
- **A node with no DC path to ground.** That is a genuine nullspace, not a rounding
  problem. The usual GMIN-to-ground fix is refused on purpose: a 1e-12 entry in a
  matrix of O(1) entries destroys exactly the conditioning the scaling just bought.
  A named error beats a `LinAlgError` at some arbitrary frequency.
- **DC.** An inductor branch is singular at ω = 0.

## Testing

```bash
./run -m pytest tests/ -q
```

Invariants are weighted above closed forms. A closed form proves the engine right
for one circuit; `SᴴS = I` on a lossless ladder proves it right about energy, and
catches essentially every stamp sign error at once. The suite also asserts
complex-symmetry of `A` (equivalent to reciprocity), passivity under loss, and that
the refusals above actually fire.

## Schematics from the netlist

```bash
./run -m vnafit.cli schematic examples/k9dp_bci_v1.net --out schematic.png
```

Automatic schematic layout is a hard problem in general — a netlist records
connectivity and says nothing about position, so a general placer has to invent
an aesthetic, and most that try produce something worse than the netlist.

This does not attempt the general problem. The circuits this tool exists for are
almost all **ladders**, and a ladder has a drawing everyone already agrees on: a
spine of series elements left to right, rungs of shunt elements down to ground,
and arcs for anything else. The heuristics are stated rather than tuned:

1. **The spine is the LONGEST simple path** from port 1 to port 2 through
   ungrounded elements — not the shortest. A filter's signal path goes through
   *every* resonator, while a bridging cross-coupling is a shortcut across them.
   Shortest-path drew a 3-pole cross-coupled filter as a 2-pole one with the
   middle resonator dumped in the leftover row.
2. **`K` couplings connect the ladder too.** Two magnetically coupled resonators
   sit side by side even though no wire joins them; without this a tapped helical
   pair has no ungrounded route from port to port at all and the whole circuit is
   "unplaceable". A spine link made of a coupling is drawn as a dashed arc, since
   there is no component in line there.
3. Transformers and transmission lines bridge their two "+" nodes, so they count
   as spine elements despite both windings touching ground.
4. Bridges arc **above** the spine (where designers draw cross-couplings),
   couplings **below**.
5. **Several shunt elements on one node fan out sideways**, each with its own
   label column. A parallel L‖C is the commonest thing here and stacking both on
   one vertical line made it unreadable.
6. **Nothing is ever silently omitted.** Anything the ladder cannot place is
   listed with its nets. A schematic that quietly drops a component is worse than
   no schematic, and a test asserts every element ends up somewhere.

Values are shown evaluated, not as their expressions. I built it the other way
first — `{1/(4*pi*pi*f0*f0*L)}` says "tuned to f0" where `3.63 pF` does not — and
drawing it settled the argument: at schematic type size the expression is an
unreadable smear that collides with its neighbour.
