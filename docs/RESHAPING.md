# Reshaping

Geometric facial reshaping. Started 2026-08-31, reversing the decision recorded in
`docs/HANDOFF.md` section 8b at the user's direction.

## What changed, and what did not

The engine now has two registries.

| | `TOOL_REGISTRY` | `RESHAPE_REGISTRY` |
|---|---|---|
| what it does | changes colour and luminance | moves pixels |
| can it move geometry | no, and an AST test proves it | yes, that is the point |
| contracts | `tool_schemas()` | `reshape_schemas()` |
| reached by | the CLI, the API, the pipeline | asked for by name, deliberately |

**Nothing about the six retouching tools weakened.** They are still covered by the AST test over
18 warping function names, still route every pixel through `apply_masked`, and still may read
only `analysis.masks` and `analysis.geometry`. A run that used only `TOOL_REGISTRY` is still
provably identity-preserving, and `test_the_two_registries_never_mix` is what keeps that sentence
true.

That test is the load-bearing one in this whole area. If the two registries ever merge, the
guarantee does not get weaker, it becomes **unverifiable**, and a future agent is one tool call
away from reshaping a face nobody asked to reshape.

## How the warp works

A displacement field built from **handles**: landmark positions paired with a target offset. Each
handle contributes a Gaussian bump and the field is their sum, so the deformation decays smoothly
to nothing. `cv2.remap` resamples through it.

```
d(p)   = sum_i offset_i * G_sigma(p - h_i)          sigma = radius / 2
out(p) = rgb(p + d(p) * region(p))
```

Not a thin-plate spline. A TPS reproduces the handle offsets exactly and oscillates between them;
this under-shoots where handles are sparse and stays monotone, which is the safer failure mode on
a face. The under-shoot is not a problem because operations state a **measured target** and
rescale until the measurement matches, which converges in one pass since the field is linear in
the offsets.

### The mask primitives are the wrong tool, and were tried first

* `normalized_lowpass` computes `blur(x*w)/blur(w)`. With a handful of impulses that is the
  *average of the offsets* over the whole influence area: a plateau with no falloff, which warps
  the surrounding face as hard as the nose. Measured: the achieved narrowing saturated at 0.046
  regardless of strength.
* `blur_mask` clips its output to `[0, 1]`. Applied to a displacement field in pixels it
  silently truncated every displacement to exactly one pixel, which is what the measurement
  showed before anyone looked at the code.

Both are correct for what they were built for, masks and illumination. Neither belongs anywhere
near a vector field.

## Three guardrails, all in `apply_warp`

`apply_warp` is to reshaping what `apply_masked` is to the photometric tools: the single place
that writes the result, so a new operation cannot forget a rail.

1. **Bounded displacement.** Nothing moves further than `max_displacement` times the interocular
   distance. Refused, not clamped.
2. **Containment.** The field is multiplied by a region mask and the result outside it is copied
   from the original, so a nose operation cannot move an eye. Verified on the output.
3. **No folding.** A field whose Jacobian determinant goes non-positive has folded the image onto
   itself, which is the melted look of a badly used liquify. Checked and refused.

## Keeping the calibration discipline

The worry going in was that reshaping has no measurable ground truth: "a better nose" is a matter
of taste. Geometric reshaping turns out to dodge this entirely, because the *operation* is
measurable even though the aesthetic is not. `narrow_nose` does not ask for "nicer"; it asks to
remove 12 percent of the alar width at strength 1.0, and it verifies the result in landmark
space. What people think of the result is a separate question from whether the tool did what it
said.

Measured on a portrait at 193 px interocular:

| strength | requested | achieved | max displacement | min Jacobian |
|---|---|---|---|---|
| 0.3 | 0.0360 | 0.0360 | 2.63 px | 0.900 |
| 0.6 | 0.0720 | 0.0720 | 5.25 px | 0.801 |
| 1.0 | 0.1200 | 0.1200 | 8.76 px | 0.668 |

**Measure both ends the same way.** The first version read the baseline as the full extent of the
nose ring and the result as the distance between two handle-group means. Two different
estimators, so the rescaling loop was correcting a quantity that contained a constant offset as
well as the narrowing, and the achieved reduction overshot the request by about 19 percent at
every strength while converging on nothing. This engine has now made that mistake twice; the
first time was measuring `skin_smoothing` by re-decomposing its own output.

## Why the nose first

It is surrounded by skin on every side, so the warp has nothing awkward to resolve behind it.
The jaw borders hair, neck and background, which is where cheap filters give themselves away.
The brows are hair rather than skin and a warp alone will not reshape them convincingly.

## Not done yet

- Jaw and brows.
- Other nose operations: bridge width, tip refinement, length.
- No pipeline or CLI integration. Reshaping is called directly, on purpose, until the shape of
  the API for it is settled.
- Reshaping is invisible to `diagnostics.summarize`. `landmark_drift` will report a real
  geometric change as a warning, which is correct for the retouching tools and wrong here; a
  reshaping run needs its own diagnostic that checks the change against what was requested.
