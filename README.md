# vnafit

Measure an RF filter with a NanoVNA, put a circuit model beside it, and tune the
hardware until they agree — with the model telling you which trimmer to turn and
being honest about which of its own numbers the measurement can actually pin down.

```bash
vnafit gui examples/tinyfilter.net          # live window, connect a VNA
vnafit card examples/tinyfilter.net --out card.png   # what a sweep could determine
vnafit fit model.net capture.s2p            # fit, with identifiability reported
```

Everything is a plain SPICE-style netlist. LTspice exports load directly.

---

## What it does

**Live tuning.** A window shows the measured sweep, the model, and a frozen
*target* — the values as loaded, the thing you are aiming at. It reads the
schematic, works out which parameters a measurement can determine, and prints
what to turn:

```
resonator      now      target    turn
  C1      146.617   146.000    +1.05%    32.38 pF  <- turn
  C2      145.567   145.983    -0.63%    35.83 pF
  C3      146.072   146.000    +0.12%    32.08 pF
  Cm1   k  0.04789   0.04493    +6.59%   fixed part
  Qu         113.1     198.0   -42.89%   not tunable
```

Frequencies and ratios, not picofarads — because a fit can walk a long way along
the impedance scale (L up, every C down) without changing the response, and a
readout in picofarads then asks you to change a 32 pF trimmer to 301 pF. See
[docs/identifiability.md](docs/identifiability.md).

**Parameter extraction.** Fit the netlist to a sweep and get error bars, a
profile, and a refusal to quote a value the data cannot support.

**Model checking before you measure.** `vnafit card` draws the schematic with
each fittable parameter outlined on the elements it drives, and says which are
determinable, which are degenerate, and which are pure gauge — from the circuit
alone, with no data.

## Why it exists

A coupling-matrix model (`f0`, `k`, `Qe`, `Qu`) is narrowband by construction.
On a capacitively-coupled filter it comes out 8.0 dB pessimistic below the
passband and 3.8 dB optimistic above; a hand-picked exponent pair patches that to
2.45 dB. All of it is one defect — the model knows a coupling *coefficient*, not
a coupling *element*. A netlist knows the element, so `k(f)` and `Qe(f)` follow
from the topology instead of being assumed.

## Install

```bash
git clone https://github.com/KM6AM/vnafit && cd vnafit
./bootstrap          # creates the conda env, or use pip below
pip install -e .
```

`./run` launches the GUI with whichever interpreter on the machine actually has
the dependencies — useful where the system `python` is not the one you want.
Requires Python 3.10+, numpy, scipy, matplotlib, pyserial, and tkinter for the
window. Hardware support is the NanoVNA-H4 (DiSlord firmware) over USB; every
other feature works from a `.s2p` file with no instrument attached.

## A first run, with no hardware

```bash
vnafit card examples/tinyfilter.net --out card.png
vnafit gui examples/tinyfilter.net captures/helical_2m_live.s2p
```

The second replays a stored sweep as though it were live, so the whole tuning
loop can be exercised without a VNA.

## Documentation

| | |
|---|---|
| [Netlist format](docs/netlist-format.md) | the SPICE subset, the RF extras (`Q=`, `.port`, `.fit`), and exchanging files with LTspice, EasyEDA, KiCad, ngspice |
| [What a measurement can determine](docs/identifiability.md) | structural vs practical identifiability, gauge freedoms, why a fit walks, and what the tool refuses to report |
| [The solver](docs/engine.md) | vectorised MNA, what it refuses to do, the schematic layout, testing |
| [The helical bench](docs/helical.md) | designing a 1–4 section helical, and the align / couple / gap / measure sequence as reduced netlists |
| [Calibration](docs/calibration.md) | deriving your own calibrations, applying them in software, stitching them, and what the file records so it can be checked |
| [The NanoVNA driver](docs/nanovna-driver.md) | the single-file driver on its own: API reference, what the H4 does not measure, and why the instrument is always handed back free-running |
| [Installing](docs/install.md) | why tkinter decides the install path, and why the launcher gates on the Python version as well as the imports |
| [Worked examples](docs/worked-examples.md) | real filters and real sweeps, including the conclusions that had to be withdrawn |

## Status

Working and used, version 0.1. The engine is verified against closed forms,
losslessness (`SᴴS = I` to 1e-13) and reciprocity; 201 tests. Interfaces may
still change.

**Known limits.** No `.subckt`, `.include` or device models. Loss is constant-`Q`,
which is not skin effect and is non-causal — fine for magnitude work, wrong for
group delay. Two-port only. The role heuristic that groups imported elements is a
guess about how filters are built, applied only where values are already equal
and always reported.

## Licence

MIT — see [LICENSE](LICENSE). Copyright © 2026 Badgie Miller, KM6AM.
