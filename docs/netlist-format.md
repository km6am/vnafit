# The netlist format

A SPICE subset plus the few things RF needs and SPICE has no word for.
Everything here is accepted by `vnafit` directly; ordinary SPICE netlists
from LTspice, EasyEDA, KiCad or ngspice are read as well, see the end.

```spice
2-section helical, 2 m
.param f0=145.4Meg  L=330n  C={1/(4*pi*pi*f0*f0*L)}

.port 1 p1 0 Z0=50
.port 2 p2 0

X1  p1 0 a 0  n=0.14        ; tap, as an ideal autotransformer
L1  a 0 {L} Q=670
C1  a 0 {C}
K1  L1 L2 0.0121            ; magnetic coupling between the resonators
L2  b 0 {L} Q=670
C2  b 0 {C}
X2  p2 0 b 0  n=0.14

.ac lin 401 95.4Meg 195.4Meg
```

Elements: `R L C` (with optional `Q=` / `tand=`), `K` mutual coupling, `X` ideal
transformer, `T` transmission line. Values take SPICE suffixes (`330n`, `1Meg`,
`4.7k`), bare parameter names, or `{...}` expressions. Expressions are evaluated by
an AST walker, not `eval` — a netlist may have come from someone else.

Deliberately absent: nonlinear devices, time-domain analysis. This is a linear AC
solver for passive RF networks; anything else belongs in ngspice.

## Exchanging netlists with other tools

```bash
./run -m vnafit.cli import bci.cir --out model.net    # from LTspice, EasyEDA, KiCad
./run -m vnafit.cli export model.net --out bci.cir    # back out again
```

**Where to draw one.** LTspice (*File → View SPICE Netlist*) is free and its
format is the de-facto standard. In the browser, EasyEDA (*File → Export NetList
→ Spice*) is free; SPICE-Online edits a live netlist directly; CircuitLab
exports SPICE but is paid. KiCad and Qucs-S both export from their schematic
editors. **Falstad CircuitJS, which is the one most hams reach for, has its own
save format and does not export SPICE at all** — you have to redraw it elsewhere.

**The catch, and it is structural.** A vnafit netlist declares *ports*, because
S-parameters are defined between reference impedances. An ordinary SPICE netlist
has no such concept — it has a voltage source and a load resistor, and the port
impedance is implicit in whatever resistance sits in series. So importing is an
*inference*, not a translation:

```
port 1: node 'N002', Z0 50 ohm  (far side of Rsrc, the source resistance)
port 2: node 'N005', Z0 50 ohm  (Rload to ground, matches the source resistance)
treated as fixture, not circuit: V1, Rsrc, Rload
ignored (meaningless to a linear AC solver): .backanno, .end
```

It always reports what it decided and `--port1 node:50` overrides it. Getting
that wrong means modelling a different network, so it is never silent.

Two things it refuses to do quietly:

- **Ideal transformers are not exported.** SPICE has no primitive, and writing
  one as a comment disconnects the circuit. It stops and says so;
  `--xfmr approximate` writes coupled inductors instead and labels them as not
  the same network.
- **`Q` is dropped on a wide sweep rather than approximated.** A constant
  resistance cannot stand in for a constant `Q` across 200:1. Measured on the
  BCI example: dropping the loss moved the round trip by 0.13 dB, while the
  constant-R stand-in moved it by 0.22 dB — the approximation was *worse* than
  the omission. It is used only below 4:1, and the file says which happened.
