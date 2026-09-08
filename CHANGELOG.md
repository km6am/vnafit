# Changelog

## 0.1.0 — unreleased

First public version.

**Modelling**
- Vectorised MNA solver: R, L, C, mutual `K`, ideal transformer, transmission
  line, and 2-port S-blocks; verified against closed forms, reciprocity, and
  losslessness to `|SᴴS − I| < 1e-13`.
- SPICE-subset netlist parser with `.param` expressions, per-element `Q=`,
  `.port`, `.fit` and `.ac`.
- Import from and export to ordinary SPICE. Ports are inferred from the source
  and load resistors and always reported; element values are promoted to
  parameters and grouped by their role in the ladder.

**Identifiability**
- `vnafit card`: what a sweep could determine about a netlist, from the circuit
  alone. Reports degenerate groups by their invariant, distinguishes structural
  determinacy from measurability at a stated noise level, and refuses to admit a
  gauge parameter to a fit.
- Profile likelihood and Jacobian/SVD reporting, both scaled by reduced χ².

**Measurement**
- Standalone NanoVNA-H4 driver (stdlib + numpy only) that always hands the
  instrument back free-running, including on SIGTERM.
- One Touchstone reader and writer honouring the option line.

**Live tuning**
- Tk window with the measured sweep, the model, a frozen target and a zoom on
  the passband or notch; schematic with click-to-pin parameter selection;
  optional refit on every sweep.
- Tuning advice in resonator frequencies and capacitance ratios, which the
  impedance scale cannot move.

**Retracted during development**, recorded because the numbers were published
before they were checked: an earlier `examples/tinyfilter.net` derived its
component values instead of reading them from Figure 3a of the design note, and
a live unit was reported as "2.1× over-coupled" on that basis. It is not — the
board is built to its design within a few percent. The fitted numbers were
right; the reference values were invented. See
[docs/worked-examples.md](docs/worked-examples.md).
