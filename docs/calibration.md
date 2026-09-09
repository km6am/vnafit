# Calibration

Calibrations you own: derived from standards you swept, kept in a file that
records what they are, applied in software, and stitched together when one span
is not enough.

The instrument's own calibration lives in five slots with no record of the span
it was taken over, no notes, and no way to get it back except as bare numbers.
That is enough to lose an afternoon to — this project did, on a filter whose
"flat 2.4 dB loss" turned out to be a calibration being interpolated a long way
outside where it was taken. **A calibration you cannot audit is a number you
have to trust.**

## Two paths, and they do not mix

There are exactly two ways to get corrected data, and **using both at once
corrects twice**. That failure is quiet: the result is smooth, physical and
wrong, because an error model applied a second time still produces a
well-behaved curve.

| | who corrects | the instrument's correction | what you get |
|---|---|---|---|
| **Instrument calibration** | the H4 | **on** — `recall N` chooses the slot | corrected S11/S21 straight from the wire |
| **Your calibration** | vnafit | **off** — forced when you adopt one | raw sweeps, corrected here with terms you can inspect |
| **Neither** | nobody | off | raw. A real state, and the easiest one to be in by accident |

Adopting a local calibration **switches the instrument's correction off for you**
and restores it when the window closes — through the same registry that hands
the display back on SIGTERM. `Recall and use` in the calibration window goes the
other way: it recalls the slot, turns the instrument's correction on, and drops
the local one.

### The status line always says which

Beside the connection indicator, and read **back from the instrument** rather
than remembered — the interesting failure is the one where what the software
believes and what the instrument is doing have come apart:

```
cal: instrument            the H4 is correcting
cal: 2m.calz               we are, and the H4 is not
cal: NONE -- raw           nobody is                        (red)
cal: BOTH -- corrected twice                                (red)
```

### Yes, you can choose a slot

`recall N` — `dev.recall_cal(2)`, or the slot box in the calibration window.
Two things the instrument will not tell you, which is most of the reason this
module exists:

- **the span its calibration was taken over.** It interpolates across whatever
  you sweep and reports nothing.
- **when it is interpolating.** `CALSTAT_INTERPOLATED` is bit 9 of `cal_status`,
  and the firmware's status loop prints bits 0–8. The instrument knows and there
  is no way to ask it.

## Making one

In the window: **`Cal…`**, which steps through the standards.

```
Load        connect a 50 Ω termination on PORT 1
Open        connect nothing on PORT 1, or an open standard
Short       connect a short on PORT 1
Thru        connect the two cables to each other        (optional)
Isolation   both cables terminated, not joined          (optional)
```

`Load` is first because it is the standard most often forgotten and the one
whose absence is least visible afterwards: without it there is no directivity
term, and a return loss is decorative. **Every capture is taken with the
instrument's own correction switched off** and restored afterwards —
calibrating through an existing calibration is circular, and the result looks
perfectly fine.

Or from the command line, from standards captured with `--no-cal`:

```bash
vnafit cal build --open o.s2p --short s.s2p --load l.s2p --thru t.s2p \
                 --out 2m.calz --notes "SMA kit, 2 m bench, cold"
```

## Using one

```bash
vnafit cal apply raw.s2p 2m.calz --out corrected.s2p
```

In the window, a calibration adopted through `Cal…` is applied to every sweep,
and `apply cal` turns it off without discarding it.

It is **never applied outside the span it was taken over**. Extrapolating error
terms is how a calibration quietly stops meaning anything, and refusing is the
whole point of the file. If the sweep has more points than the calibration, the
correction says so:

```
NOTE: the sweep has 4.0x more points than the calibration, so the terms are
interpolated between cal points -- the resolution is the cal's, not the sweep's
```

That is the same criticism this project levelled at the instrument for
interpolating silently, applied to itself.

## Stitching

```bash
vnafit cal apply raw.s2p hf.calz vhf.calz --out corrected.s2p
```

Each frequency is corrected by whichever calibration covers it, **preferring the
finer one** where they overlap, and the output records the crossovers. Where
nothing covers a stretch it refuses and names the gap rather than reaching for
the nearest cal.

Two calibrations are never blended in an overlap. Terms taken an hour and a few
degrees apart do not average into something meaningful; they average into a
discontinuity you cannot see afterwards.

## What is in the file

A `.calz` holds the error terms, the frequency grid, the span, the point count,
the instrument's identity, the date, your notes — **and the standards
themselves**. That makes it bigger and it makes it checkable:

```bash
vnafit cal show 2m.calz
```

```
130.0000-165.0000 MHz, 401 points, 87.500 kHz spacing
corrects: S11 and S21
made:     standards on 2026-09-08T14:02:11
notes:    SMA kit, 2 m bench, cold
standards embedded: load, open, short, thru
re-derived from the embedded standards: terms agree to 3.55e-16
```

That last line is the audit: the terms were recomputed from the sweeps in the
file and compared with the ones stored beside them.

## The algebra

Read from `apply_error_term` and the `eterm_calc_*` functions of the NanoVNA-D
firmware, so a sweep corrected here and the same sweep corrected on the
instrument agree rather than merely resembling each other:

```
S11c = (S11m − ED) / (ER + ES·(S11m − ED))
S21c = (S21m − EX) / ET  ×  (1 − ES·S11c)        ET = S21_thru − EX
```

and from the standards, with `ED = S11_load` and `O`, `S` the open and short
measurements less `ED`:

```
ES = (O + S) / (O − S)          ER = O·(1 − ES)
```

`Touchstone.normalize(thru)` is exactly the non-enhanced half of the S21
correction.

Verified against a synthetic instrument with known error terms: the solve
recovers ED, ES, ER and ET to **10⁻¹²**, and a DUT measured through those errors
comes back to **10⁻¹²** in both S11 and S21.

## Limits

**Standards are assumed ideal** — open = +1, short = −1, load = 0. A real kit
has a delay and a fringing capacitance; a standards model belongs here later.

**Leaving out the isolation sweep costs exactly the crosstalk term**, not a
vague degradation. Measured: with EX = 10⁻⁴ unmeasured, S21 is wrong by
1.6 × 10⁻⁴. That is irrelevant at −20 dB and decisive at −70.

**No full two-port.** The H4 measures S11 and S21 only, so there is no S22 or
S12 to solve the reverse terms from. This is a one-port SOL plus enhanced
response — the same model the instrument itself uses.

**No upload.** The firmware exposes `data 2..6` to read the error terms out and
nothing at all to write them back, so `vnafit cal from-slot` archives what the
instrument holds but cannot put one there. Note also that the instrument does
not report the frequency grid its calibration was taken on, so a slot archive
records point index rather than hertz — it is a record of the numbers, not
something that can be applied to a sweep.
