# Worked examples

Real filters, real sweeps, and what came out -- including the parts that
did not work and the conclusions that had to be withdrawn.

## Worked example: the N6ARA TinyFilter

`examples/tinyfilter.net` — a 3-pole 0.25 dB Chebyshev for 2 m, built from the
K6STR design note. It is parameterised by the **physical parts**, not by
`f0`/`k`/`Qe`, because that is what the hardware exposes: three trimmers turn and
everything else is a fixed capacitor. So `c1 c2 c3` are tuning parameters and
`cm`/`cin` are a **build check** — if a fit moves them, a part is wrong or a
joint is bridged, not mistuned.

Every value is read off **Figure 3a** of the note, "Schematic with final
component values": 30 nH, end caps 6.27 pF, coupling 1.78 pF, nodes 32.04 /
36.06 / 32.04 pF. The lossless response is then 9.8 MHz wide with its flat top
running 140.7–150.4 MHz, against Figure 2b's ~141–150.

An earlier version of this file **derived** those values instead, from a ripple
bandwidth assumed to be 4 MHz — itself inferred from the note's "~3 dB insertion
loss", which is circular, because the loss is set mostly by unloaded Q. It got
the coupling half right (0.888 pF against 1.78) and produced a model 4.65 MHz
wide. Worse, the derived 5.01 MHz was then asserted in a test as a figure "the
note gives", which it does not. **Read the figure; do not re-derive the design.**

**It also completes the identifiability story.** `vnafit card` says all seven
parameters are structurally determinable — *including* `L`, where on the tapped
helical `L` is an exact gauge. The difference is the series end caps: they work
against the fixed 50 Ω, which does not scale.

Structurally. On the live unit it is another matter — fitting `L` halved it to
14.7 nH and doubled every capacitor, at condition number **5.6 × 10³**, and the
SVD named the direction: `l↑, all C↓`. That is the impedance scale running away.
Pinning `L` at its printed 30 nH drops the condition number to **54**, with the
same 0.69 dB fit and one fewer parameter. Necessary is not sufficient, and this
is what that looks like on real data.

**What the fitted unit turned out to be** (a wideband capture of a real unit,
80–300 MHz, `L` pinned):

| | Figure 3a | fitted | |
|---|---|---|---|
| `cm` | 1.78 pF | 1.895 pF | +6.4% |
| `cin` | 6.27 pF | 6.179 pF | −1.4% |
| `c1` / `c2` / `c3` | 32.04 / 36.06 / 32.04 pF | 31.67 / 36.06 / 31.96 | within 1.2% |
| resonator f0s | 146.00 / 145.98 / 146.00 | 146.61 / 145.57 / 146.07 | **1043 kHz apart** |
| `Qu` | 198 (assumed) | 113 | **−43%** |

So the board is **built and tuned to its design values within a few percent**,
and its 10.18 MHz bandwidth sits 4% above the 9.75 MHz those values give. Two
things are genuinely off: the three resonators are 1043 kHz apart where the
design puts them within 17 kHz, and the unloaded Q is 113.

An earlier version of this section read the same fit against the *derived*
values above and reported the filter as **2.1× over-coupled**, with a 10.2 MHz
bandwidth against a 5.0 MHz design. The fitted numbers were right; the design
values they were compared against were not. Nothing was wrong with the
measurement or the fit — the reference was invented.

Residual structure the model does not have: a notch near 288 MHz, which looks
like self-resonance in the coupling caps.

## Worked example: the K9DP QRP BCI filter

`examples/k9dp_bci_v1.net` and `..._v2.net` model both versions of the
[K9DP BCI filter](https://www.k9dp.com/qrp-bci-filter-build-instructions/) from
its published parts list. The kit page gives capacitor values and turn counts
but no schematic, no core type and no response curve, so most of the model is
reconstructed — and the netlist says line by line which numbers are printed on
the part and which are inferred.

The prototype falls out of the capacitor **ratio** alone. For a 5-element
high-pass the series caps go as `1/g_k`, so `C_outer/C_middle = g3/g1`:

| | printed | ratio | matches | fc |
|---|---|---|---|---|
| v1 | 3300 / 1000 pF | 3.30 | Butterworth (3.236) | 1.56 MHz |
| v2 | 820 / 470 pF | 1.745 | 0.1 dB Chebyshev (1.722) | 3.39 MHz |

Each then predicts the middle cap it was not fitted to — 1020 pF against 1000
printed, and 476 pF against 470 — which is the check, not the claim.

**The core, inferred by cross-check.** Neither kit states the toroid, but each
version independently implies the shunt inductance it needs, and dividing by
turns² gives the core's inductance factor: v1 gives 5.47 nH/N² from 24 turns,
v2 gives 5.29 nH/N² from 18 turns. Two different prototypes and two different
turn counts agreeing to 3% says both use the same core, near A_L ≈ 5.3–5.5
(Micrometals T44-2 is 5.2). Still an inference from two numbers.

```bash
./run -m vnafit.cli show examples/k9dp_bci_v2.net --highpass
./run -m vnafit.cli plot examples/k9dp_bci_v1.net examples/k9dp_bci_v2.net \
      --start 3e5 --stop 6e7 --out response.png
```

**What the model says that the kit page does not.** v2 buys its "much higher AM
band rejection" — 1 MHz goes from −21 dB to −60 dB — by moving the corner from
1.65 to 3.06 MHz. That is above 160 m, so v2 attenuates 160 m by **27–32 dB**.
It is a band trade, not a free upgrade, and worth knowing before you buy one for
a Top Band rig.

**Identifying the core from a sweep.** Because `A_L` sets the corner, it is
sharply determined: fitting a stand-in sweep generated with a T50-2 core
recovers `A_L = 4.9008–4.9012 nH/N²`, which excludes T44-2 (5.2) outright. That
is the intended bench workflow — sweep the real filter, fit, and let the tool
say what it actually identified.

## Can this measurement tell the laws apart?

`vnafit laws` fits a measurement under each coupling law and ranks them — and
then tries to talk you out of the ranking.

Ranking four residuals always produces a winner. The exponents only change the
**skirts**, so if a sweep does not reach far enough down them before hitting the
fixture's leakage floor, every law predicts nearly the same thing over the
usable band and the winner is noise. So the command also measures how far apart
the laws actually are at the fitted parameters, and refuses to name one unless
that separation beats three times the residual.

On the bench data here it does both things:

```
tinyfilter_wide.s2p: 1180 points, 117.5-300.0 MHz (x2.55), Qu pinned 198
  capacitive      (k~f^+1, Qe~f^-2)  rms   0.952 dB
  helical_tapped  (k~f^-1, Qe~f^+0)  rms   4.113 dB
  constant        (k~f^+0, Qe~f^+0)  rms   4.525 dB
  inductive       (k~f^-1, Qe~f^+2)  rms   5.302 dB
    capacitive      separation  17.50 dB   DISCRIMINATING
  VERDICT: capacitive -- the data can tell, and it does.
```

That confirms the `(+1, −2)` prediction for a capacitively coupled,
series-cap-fed filter, decisively — 4.5× better than any rival, on a 2.55:1 span.

The helical is the opposite case, and it corrected me:

```
Helical_30mmgap.s2p: 128 points, 129.4-161.2 MHz (x1.25), Qu pinned 670
  helical_tapped  rms 1.106 dB      constant rms 1.190 dB
    helical_tapped  separation   1.02 dB   underpowered
  VERDICT: this sweep CANNOT distinguish the laws.
```

`helical_tapped` "wins" by 7%, and that is meaningless: the laws only separate
by 1.02 dB over the usable band against a 1.19 dB residual. The second wideband
helical file ranks `constant` first instead — two files disagreeing is what an
underpowered comparison looks like from the inside. **So the `(−1, 0)` prediction
for a tapped helical is neither confirmed nor refuted; it is currently
untestable.**

What would settle it is a **lower fixture leakage floor**, not more averaging or
a wider sweep — the filter's skirts are already below the floor at ±11%:

| floor | usable band | constant vs helical_tapped |
|---|---|---|
| −70 dB (now) | 128–166 MHz | 1.12 dB |
| −80 dB | 116–183 MHz | 1.99 dB |
| −90 dB | 98–217 MHz | **3.47 dB** |

The criterion needs ~3.6 dB, so roughly **20 dB more port-to-port isolation**.
The existing notes already established that floor is fixture leakage rather than
receiver noise (8× narrower IF plus 8× averaging moved it 3.1 dB where noise
would have moved ~18), so shielding and ferrites are the lever.

## Measuring the exponents instead of fitting the skirts

The skirt-fit route above is dead on this instrument, and it is worth knowing why
before spending a bench session on it. Measured bare, with nothing on the ports,
8 averages at the narrowest IF, the H4's own residual crosstalk is **−86 dB**,
flat across 100–200 MHz. The saved helical files sit at −70 dB, so about 16 dB of
that is fixture. But the requirement is:

| floor | separation | enough? |
|---|---|---|
| −70 dB (fixture as measured) | 1.13 dB | no |
| **−86 dB (bare instrument)** | 2.78 dB | no |
| −90 dB | 3.47 dB | no |
| −100 dB (lab VNA) | 5.88 dB | yes |

**A perfect fixture would still not settle it on an H4.** The instrument itself
is the wall. My earlier "20 dB more isolation" was necessary but not sufficient.

`vnafit exponents` takes the other route, with 80 dB of headroom instead of 3:
retune the whole filter to several centre frequencies and read `k` and `Qe` in
the **passband** at each, then take the slopes against `f0`.

```bash
./run -m vnafit.cli exponents tune*.s2p --poles 2 --qu 670
```

```
   f0 (MHz)      k        Qe     fit rms
   119.1435  0.01214     56.7    0.392
   ...
   176.8755  0.01209     37.9    0.349
  k  ~ f0^-0.01 +/- 0.00
  Qe ~ f0^-1.02 +/- 0.00
  nearest topology: helical_tapped  (expects +0, -1; distance 0.02)
  -> which implies the within-sweep law `helical_tapped` (-1.0, 0.0)
```

**The trap, which I walked into.** Retuning exponents are *not* the within-sweep
exponents in `COUPLING_LAWS`. Those say how `k` drifts across one sweep at fixed
hardware — that shapes the skirts. Retuning slopes say how `k` moves when the
resonator capacitance changes. For parallel resonators retuned by `C` (so
`C ∝ f0⁻²`):

| element | quantity | retuning | within-sweep |
|---|---|---|---|
| fixed coupling cap `Cm` | `k = Cm/C` | `f0^+2` | `f0^+1` |
| fixed mutual `M` | `k = M/L` | `f0^0` | `f0^−1` |
| fixed tap ratio `n` | `Qe = ω₀CZ₀/n²` | `f0^−1` | `f0^0` |
| fixed series cap `Cs` | `Qe = C/(ω₀Cs²Z₀)` | `f0^−3` | `f0^−2` |

Verified against the engine: cap-coupled/cap-fed retunes at (+2.21, −3.26),
mutual/tapped at (−0.01, −1.02). Comparing a retuning slope against
`COUPLING_LAWS` would name the wrong topology with total confidence, so
`RETUNE_LAWS` is separate and a test guards the confusion.

**The protocol matters more than the maths.** Retune with the **tuning screw
only**. If the aperture or the taps move between sweeps, the coupling element
itself has changed and the slope measures your hands, not the physics.

## A helical is not a lumped resonator, and mistaking one for the other invents physics

The first live capture of the 2 m helical (`captures/helical_2m_live.s2p`,
102.8–185.0 MHz usable, 6 averages, **0.007 dB trace noise**) settled something I
had been about to get wrong.

Fitted with the usual **lumped** L‖C resonator, `helical_tapped` beat `constant`
by 0.33 dB — apparent support for the `k ∝ 1/f` prediction. Fitted with a
**shorted quarter-wave** resonator, which is what a helical actually is,
`constant` beats `helical_tapped` by 0.52 dB instead:

| | lumped | quarter-wave |
|---|---|---|
| `constant` | 2.575 dB | **1.972 dB** |
| `helical_tapped` | 2.249 dB | 2.496 dB |

**The coupling law was absorbing the resonator model's error.** Same five free
parameters throughout, so this is not a complexity trade.

The two resonators differ only in the detuning term:

```
lumped   x = f/f0 - f0/f
quarter  x = -(4/pi)*cot(pi*f/(2*f0))        set with resonator="quarter"
```

They agree to 2% inside ±3% of f0 and diverge **antisymmetrically** outside it —
a line is 11% *less* detuned at 0.7·f0 and 18% *more* at 1.26·f0. That puts its
low skirt above a lumped fit and its high skirt below, which is exactly the
asymmetry the measurement shows, and predicting the sign on *both* sides at once
is what makes it a test rather than another free knob. (It also resonates again
at 3·f0, which no lumped model knows about.)

A test generates data from a quarter-wave resonator with a genuinely constant
coupling, and asserts that a lumped fit invents a frequency-dependent one.

**What this does not settle.** 1.97 dB rms is a good wideband fit for five
parameters over 1.8:1, but the trace noise is 0.007 dB — the residual is ~280×
the noise and visibly structured, so none of these is the true model yet. And
`Qu` remains unmeasurable with the ports connected: `qu_leverage` reads 0.88 and
the residual moves 2.6% across a 66× change in `Qu`.

## Tuning with a model that is only roughly right

`vnafit track` is the headless tuning aid: sweep, fit, print, repeat. It is built
for the normal case, which is an **incomplete model** — nobody types a netlist
that matches their hardware to a tenth of a dB, and the tool should be useful
before they do.

```bash
./run -m vnafit.cli track --netlist examples/rough_2pole.net \
      --start 130e6 --stop 165e6 --seconds 900 --poles 2 --qu 800
```

```
   n   f_peak     IL     3dB BW    RL     k        Qe1/Qen   fit    model p/s
   6  147.0625   -0.90    3.1500   14.2  0.01549   106/110    0.68   0.64/0.61
  12  147.6750   -2.22    3.5875    6.2  0.01549    94/122    0.69   3.22/1.24
  18  147.9375   -4.16    4.4625    2.9  0.01570    85/145    0.89   3.82/0.67
```

Three tiers, deliberately:

- **Model-free** (`f_peak`, `IL`, `3dB BW`, `RL`) comes straight off the trace and
  is right even when every model is wrong. These are the numbers to tune by.
- **Fitted** (`k`, `Qe1/Qen`) needs a model, and `fit` is its residual so you can
  see how far to trust it.
- **`model p/s`** scores the netlist you supplied, passband and skirts separately.
  Never required; a failed fit prints a dash and the loop keeps going, because a
  blank screen is the worst outcome when both hands are on a trimmer.

Every sweep is appended to a CSV, so the session is reviewable afterwards — which
is where you find out whether a knob moved what you thought it did.

### What it resolves

From 200 sweeps of an untouched filter (1.5 s apart, 401 points over 55 MHz):

| | 1σ | 3σ |
|---|---|---|
| f0 (fitted) | 3.3 kHz | 10 kHz |
| f_peak (picked) | 9.7 kHz | 29 kHz |
| insertion loss | 0.003 dB | 0.009 dB |
| 3 dB BW | 9.7 kHz | 29 kHz |
| k | 0.000012 | 0.000036 |
| return loss | 0.49 dB | 1.5 dB |

The fitted `f0` is 3× more repeatable than the picked peak, because it uses every
point on the curve instead of the single highest one — which can only ever land
on a grid line. That ratio is the whole argument for fitting rather than reading
markers.

Return loss is by far the least repeatable number, which is worth knowing before
chasing a dB of it.

## Reconciling the two models

The old coupling-matrix model is not replaced — it is vendored as `coupled.py` and
kept as an independent second opinion. `synth.py` maps `(f0, k, Qe, Qu)` onto an
actual circuit, for the two topologies these filters use:

| | coupling | ports | predicted law |
|---|---|---|---|
| `synth.magnetic` — the helical | mutual M between the coils | tapped on the coil | `k ∝ 1/f`, `Qe` flat |
| `synth.capacitive` — the TinyFilter | series coupling caps | series caps into 50 Ω | `k ∝ f`, `Qe ∝ 1/f²` |

Both are *parallel*-resonator (J-inverter) filters. The same physical coupling
element gives the **opposite** exponent in the series-resonator picture, so a
crystal ladder would invert these.

The tests do not merely check that the models agree — agreement alone would be
passed by an engine that was secretly a coupling matrix. They check that each
netlist tracks **its own** predicted law into the skirts and visibly fails to
track the other one, that the two topologies are separated by far more than trace
noise (so the test is measuring something), and that the netlist produces
responses the coupling matrix provably cannot — a cross-coupling opens a second
path across the filter and puts a −105 dB transmission zero where the plain
filter has only a monotonic skirt, on a side chosen by the *sign* of the coupling.

That last one also retires a standing caveat: "the sign of k is not observable"
is true for a tridiagonal coupling matrix, where flipping a resonator's basis
vector absorbs it, and stops being true the moment a non-adjacent coupling exists.

**A new prediction, so far underived from measurement.** `COUPLING_LAWS` offered
only three named exponent pairs. The two exponents are set by two *independent*
choices — how the resonators couple, and how the ports couple — and a tapped
helical mixes them: magnetic coupling gives `k ∝ 1/f`, but a tap is an
autotransformer, so `Qe` is flat. That pair `(−1, 0)` is none of the three, and
forcing it into one mis-models the skirts on the very filter this began with. It
is now `helical_tapped`, and one wideband sweep settles it. Confidence: the
`k ∝ 1/f` half is solid, derived two ways; the flat-`Qe` half assumes an ideal
autotransformer and ignores leakage inductance, so it is a prediction, not a
result.

## Live tuning

`./run <netlist> [file.s2p]` opens the tuning window: the measured sweep with
the model drawn over it, one editable row per `.param` with a log slider beside
it, and a readout. Turn a screw, watch the traces converge. A 401-point netlist
evaluation is well under a millisecond, so the model keeps up with the slider.

Load a `.s2p` instead of connecting a VNA and the entire live path — thread,
plotting, model, readout — runs with no hardware on the bench.

Two things on screen are deliberate:

- **The residual is split into passband and skirts.** Every model agrees in the
  passband; the skirts are where the information about topology lives, and one
  rms number hides exactly the part you are trying to see.
- **The coupling-matrix model can be overlaid alongside the netlist**, and
  translated back into the `k` / `Qe` language the bench procedure is written in.
  Having both on one screen is how you catch either of them lying — and where
  the dotted line peels away from the dashed one is where the narrowband model
  stops being able to describe the circuit.

Sliders are logarithmic and span a decade either way, because these parameters
run from 1e-13 to 1e9 and a linear slider is useless across that.

## Using it from IPython or a notebook

```python
from vnafit import load, netlist, sweep
import matplotlib.pyplot as plt

d = load("capture.s2p")
d                       # <capture.s2p: 2-port, 801 pts, 100-200 MHz, peak S21 -0.75 dB at 146.75 MHz>

plt.plot(d.f_mhz, d.s21_db)
d.at(146e6)             # every quantity at one frequency, as a dict
d.band(140e6, 150e6)    # a slice, as a new object; the original is untouched
d.bandwidth(3.0)        # None if the sweep does not contain both crossings

m = netlist("model.net")
f, S = sweep(m, d)      # the model on the measurement's own grid
plt.plot(d.f_mhz, 20*np.log10(abs(S[:, 1, 0])))
```

Arrays, named the way you say them: `f`, `f_mhz`, `s11 s21 s12 s22`,
`s21_db`, `s21_deg`, `vswr`, `z_in`, `return_loss`, `insertion_loss`,
`group_delay`. Plus `columns()` for a dict you can hand to pandas, `peak()`,
`load_many("sweeps/*.s2p")`, and `d.plot()` for a quick look that returns the
axis so you can keep drawing on it.

Group delay differentiates the **unwrapped** phase; without that every 2π
crossing becomes a spike that reads as a resonance, and a test asserts the
difference. `bandwidth()` returns `None` rather than a number measured off the
edge of the data.

`import vnafit` pulls in numpy and scipy but **not** matplotlib or tkinter —
those load only when you plot or open the GUI. A test enforces it.
