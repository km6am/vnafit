# What a measurement can determine

The part of this project that is not a filter simulator.  A sweep does
not determine every parameter of a model, and which ones it determines
is a property of the CIRCUIT -- computable before any measurement
exists.

## What a sweep could determine, before you measure anything

```bash
./run -m vnafit.cli card examples/k9dp_bci_v1.net --sigma 0.007 --out card.png
```

Which parameters a 2-port measurement can pin down is a property of the
**circuit**, not of the data — so it is computable up front, rather than
discovered after an afternoon of fitting.

Perturb each parameter in log space, build the Jacobian on a dense noise-free
grid, take its SVD. A null direction is a way of moving the parameters that
changes nothing, so the data can never separate them; in log space those read as
power laws, which is why the report can say *"only `al·turns²` is determined"*
instead of handing back an eigenvector.

The card is drawn in the coordinates that are actually determined. Reporting
`al` and `turns` as separate rows while also saying only their product is
determined contradicts itself, so the pair is collapsed into the netlist's own
name for the invariant first — found by moving along the null direction and
seeing which derived `.param` does not budge:

```
2 of 2 parameters are determinable (rank [2] at 5 random parameter points)
parameter        value  drives      peak dB/e-fold   best case  verdict
lsh          2.995e-06  L1,L2                24.11     0.0034%  identifiable
ql                 180  L1,L2                 0.13     0.9071%  identifiable

  al & turns replaced by lsh, which is the combination that is actually determined
  fc dropped -- it is a gauge and does nothing
```

On the tapped helical it finds the impedance-scale gauge by itself — the result
this project first derived by hand.

**Each parameter gets a boundary round exactly the elements it acts on**,
coloured to match its curve in the sensitivity panel below. Earlier versions
wrote parameter names as small coloured text under each element; five parameters
over four elements was an unreadable smear, and it never showed the thing worth
seeing, which is a parameter's *extent* — `k` touches one element, `f0` touches
six.

Three things this catches that are easy to miss by hand:

- **A parameter no element uses.** `fc` is declared as documentation; the
  dependency walk notices nothing depends on it.
- **A gauge.** With tapped ports the impedance scale cancels exactly, so `L` is
  unmeasurable no matter how good the sweep.
- **A product that masquerades as two parameters.** `al` and `turns` reach the
  circuit only as `al·turns²` — which also means an earlier claim in this README
  that fitting a sweep "identifies the core" held only because the turn count
  was fixed.

`best case` is the Cramér–Rao bound at the stated trace noise. It is a **bound**:
it assumes the model is exactly right and says nothing about how far a wrong
model has pushed the answer off centre. An earlier version scaled it by
√(reduced χ²) to look "realistic"; that was dropped, because model error biases
the answer as well as widening it and no inflation factor fixes a bias — and
whether the model is right is a question the residual plot answers better than
any scalar.

## Reporting what was actually measured

Every fit reports whether the data determined each number:

```
fit rms 0.5921 dB over 81 points
  fitted over 135.400-155.400 MHz
  noise model: floor -70 dBc + 0.050 dB relative
  k            0.0061045
  qe           201.279
  f0           1.45465e+08
  reduced chi2 7.2: the residual is 2.7x the assumed noise, so what is left
  over is MODEL error, not noise.  Error bars below are scaled up by that
  factor; they still assume the model shape is right.

3 parameters, 3 determined (condition number 619)
  error bars widened 2.7x because the residual exceeds the assumed noise
  k            +/-    0.7%
  qe           +/-    1.0%
```

Two independent tools, and they are meant to be cross-checked:

- **Profile likelihood** — scan one parameter, refit the rest. The number that
  makes this usable: for `m` residual points the 95% bound is `Δχ² = 3.84`, i.e.
  a rise of `√(1+3.84/m)`. **At m = 400 that is 0.5 %.** So "the profile looks
  flat" is not a test — a 0.5 % rise is invisible by eye and is the entire
  confidence interval. The tool draws the line.
- **Jacobian SVD** — the local picture, which additionally names *which*
  parameters trade against which.

They agree where the problem is near-linear (a test asserts it), and where they
disagree the nonlinearity is real and the profile wins. Both scale their
threshold by the reduced χ², because on bench data the residual is several times
the trace noise and an unscaled interval comes out several times too tight.

## What the model can and cannot tell you about a real filter

A netlist has more numbers in it than `(f0, k, Qe, Qu)` — every `L`, `C`, `R`, `M`.
That does not make them measurable. Raise every `L` and lower every `C` at fixed
`f0`, `k`, `Qe`, `Qu`, and ask whether `|S21|` notices:

| topology | change | `|S21|` moves by |
|---|---|---|
| tapped (helical) | `L` × 100 | **3 × 10⁻¹¹ dB** — machine precision |
| series-cap (TinyFilter) | `L` × 4 | 0.05 dB in-band, 0.09 dB wideband |

A tap is an ideal autotransformer, so every impedance scales together and the
network is **exactly** invariant — at every frequency, not just narrowband. The
absolute inductance of a helical resonator is not measurable from S-parameters,
and a fit that reports one is reporting its own seed.

A series input cap works against the fixed 50 Ω, which does not scale. That breaks
the invariance — but only at order `u² = (ω₀·Cs·Z₀)²`, and as `L` rises the cap
needed for a given `Qe` shrinks, `u → 0`, and it converges back to a transformer.
Against a NanoVNA's ~0.05 dB trace noise, `L` is **weakly identifiable at best**,
and only from a wide span.

So: fit in `(f0, k, Qe)` coordinates. Turn a netlist value loose only when a
profile shows it has a resolved minimum.

(An earlier estimate of "12 dB in-band" for the series-cap case was wrong — that
test changed `Cs` without retuning the resonator, so most of what it measured was
a detuned filter rather than a changed impedance level. `tests/test_identifiability.py`
holds the corrected numbers and the mechanism test that pins them down.)

## A caveat worth reading before trusting a fitted value

Constant-`Q` loss (`R = ωL/Q`) is the default because it is the unique loss law that
keeps `A` affine in ω — **not** because it is the physics. Skin effect gives
`R ∝ √f`, a 41 % difference over 100→200 MHz. Over a filter's span loss only moves
the peak by ~0.1–0.2 dB and is invisible in the skirts, so the two are
indistinguishable here. It is also non-causal (violates Kramers-Kronig): fine for
magnitude work, invalid for any future time-domain or group-delay feature.
