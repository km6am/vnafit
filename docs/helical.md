# The helical bench

A helical filter is a parallel L‖C resonator per section, magnetically coupled
through an aperture, tapped at the end coils. None of that needs its own
solver, plotting or calibration — so `vnafit/helical.py` is only the part that
is specific: turning *"four sections, 146 MHz, 2 MHz wide"* into a netlist.
Everything after that is the ordinary window.

```
Helical...   →   sections [1-4]   centre MHz   bandwidth MHz   ripple dB   Qu
                 bench step: ( ) align  ( ) couple  ( ) gap  (•) measure
```

Choosing an order rebuilds the model; the schematic, the passband inset, the
card, the fitter, the target and the calibration then work on it unchanged.

## The bench sequence

Each step is **the same filter with everything but the thing being set stripped
out** — a smaller netlist, not a different mode. That is why the whole window
keeps working on each of them.

| step | circuit | what you read |
|---|---|---|
| **align** | one resonator, both probes slack | f₀, and the width gives Qu |
| **couple** | one resonator, real tap on port 1, slack probe on port 2 | the loaded 3 dB width gives Qe |
| **gap** | two resonators, both probes slack | the peaks split by `k·f₀` |
| **measure** | the whole filter, terminated | bandwidth, ripple, return loss |

The panel prints the number to aim at. Verified by building each stage, solving
it, and checking the number that comes out is the number on screen: align within
1.9%, couple within 0.2%, gap within 0.9%.

### Two things that are easy to get wrong, and this got wrong first

**A probe is slack relative to `Qu`, not relative to the tap.** The first
version made the probe 20× the real Qe, on the reasoning that 20× is obviously
loose. But what matters is the probe's Q against Qu, and 20 × 75 = 1500 against
Qu = 670 means the two probes carry **45% of the loading**: the measured width
came out 0.408 MHz where f₀/Qu is 0.218. On the stage whose entire purpose is
reading Qu. The probe is now specified by what it does — `PROBE_Q × Qu`, adding
1/`PROBE_Q` of the resonator's own loss per probe, 2% each at the default.

**The singly-loaded 3 dB width is not f₀/Qe.** It is
`f₀·(1/Qu + 1/Qe)` — the resonator's own loss is in there too. With Qu/Qe ≈ 9
that is an **11% difference**, and aiming at f₀/Qe over-couples by that much.
The panel gives the loaded figure and prints the naive one beside it, saying
which case it is right for.

## `lref` is never fitted

With tapped ports every impedance in the circuit scales together, so raising L
and lowering C at fixed f₀/k/Qe/Qu changes |S21| by about 10⁻¹¹ dB. It is an
exact gauge freedom. `design()` leaves it out of the `.fit` set — and the
identifiability analysis finds it independently, from the circuit alone,
reporting `lref dropped -- it is a gauge and does nothing`.

Everything else is fitted, and measurable: **per-resonator `f0i` and per-gap
`ki`**, which is the point of an alignment tool. A detuned section shows up as
its own frequency rather than being absorbed into a shared one.
