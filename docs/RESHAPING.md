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

A displacement field built from **handles**: landmark positions paired with a target offset,
interpolated by a compactly supported radial basis function, Wendland's C2 kernel.

```
psi(r) = (1-r)^4 (4r+1),  zero for r >= 1        positive definite in R^3, hence in R^2
Phi_ij = psi(|h_i - h_j| / R)                    N x N, N is 4 to 30
w      = solve(Phi, offsets)                     microseconds
d(p)   = sum_j w_j psi(|p - h_j| / R)            evaluated only in the handles' bounding box
out(p) = rgb(p + d(p) * region(p))               cv2.remap, INTER_CUBIC
```

Two properties earn it its place:

* **It reproduces each handle offset exactly**, so a nose wing slides rather than deforming
  lumpily.
* **Its support is exactly the union of discs of radius R**, so containment is a property of the
  kernel rather than only a mask multiply verified afterwards.

### Two kernels were wrong before this one

**Version 1, the mask primitives.** `normalized_lowpass` computes `blur(x*w)/blur(w)`; with a
handful of impulses that is the *average of the offsets* over the influence area, a plateau with
no falloff. And `blur_mask` clips to `[0, 1]`, which silently truncated every displacement to
exactly one pixel. Both are correct for masks and illumination and wrong for a vector field.

**Version 2, a Gaussian scatter.** Its docstring claimed the field "under-shoots wherever handles
are sparse and stays monotone, which is the safer failure mode on a face". Measured on a real
portrait at the project's own defaults, it **over**-shot by 1.96x to 2.53x, with a spread of 0.57
between the three handles that define one nose wing. A recorded design rationale was factually
wrong, and the two-pass rescale in `nose.py` corrected the aggregate width while structurally
unable to correct the per-handle inconsistency underneath it.

| | Gaussian scatter | Wendland RBF |
|---|---|---|
| achieved / asked, per handle | 1.96 to 2.53 | 0.99 to 1.01 |
| spread between handles of one wing | 0.570 | 0.026 |
| field build, 4160x2774 | 59.4 ms | 0.9 ms |
| support | unbounded in principle | exactly R |

Not a thin-plate spline either, and not Moving Least Squares. Four separate routes to TPS (scipy,
numpy, OpenCV's shape module, scikit-image) were researched and all rejected: TPS is globally
supported, so it contradicts the containment requirement, and three of the four would add a
dependency to a project that deliberately has four. MLS rigid (Schaefer, McPhail and Warren,
SIGGRAPH 2006) is the one credible alternative and is worth revisiting **if jaw work produces
visible shear that Wendland cannot avoid**; it is also globally supported and about 60x slower
here. If it is ever built, implement from the paper, not from GitHub: the widely referenced
`imgwarp-opencv` carries LGPL v2 headers on exactly the MLS files despite an MIT root licence
added ten years later.

### Two traps in the kernel, both measured

* **Enlarging the radius does not improve conditioning.** The usual advice about RBF
  interpolation concerns globally supported kernels. Here `cond(Phi)` rises monotonically *with*
  the radius: 1.00 at 0.01 of the interocular distance, 9.2 at 0.16, 3.2e4 at 2.56. Growing R to
  "fix" conditioning would also destroy the locality that is the reason for choosing the kernel.
* **Coincident handles make Phi singular** (condition number 1.9e17) and `np.linalg.solve`
  raises. MediaPipe genuinely returns near-coincident nose-ring landmarks on a steep
  semi-profile, and `_alar_extremes` selects by axis projection without checking distance.
  Handles closer than 1 percent of the radius are merged, averaging their offsets, and a ridge
  covers what survives merging.

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

Measured on a portrait at 193 px interocular, after the kernel change:

| strength | requested | achieved | max displacement | min Jacobian |
|---|---|---|---|---|
| 0.3 | 0.0360 | 0.0360 | 2.43 px | 0.845 |
| 0.6 | 0.0720 | 0.0720 | 4.86 px | 0.690 |
| 1.0 | 0.1200 | 0.1200 | 8.10 px | 0.483 |

`influence_radius` was re-measured rather than carried over, because the falloff shape changed
and the old value carried no evidence for the new kernel. Across six portraits at strength 1.0,
raising it improves the minimum Jacobian monotonically (0.34 at 0.16, 0.52 at 0.22, 0.65 at 0.30,
0.74 at 0.40) but starts moving features it must not touch, because the surrounding `region` mask
is a dilated nose that reaches the upper lip: worst change inside the eyes or lips is 0.001 8-bit
levels at 0.16, 4.1 at 0.22, 12.9 at 0.30. **Containment wins.** 0.16 is the largest value that
keeps the leak below one level. The resulting 0.34 worst-case Jacobian is a local compression
rather than a fold, and checked visually on the worst fixture it is not visible.

**Contain once.** The first version multiplied the field by `region` in `nose.py` and then passed
the same `region` to `apply_warp`, which multiplies again: the delivered field was
`field * region**2`. This is the same squared-mask mistake `reduce_dark_circles` made with its
alpha, in a different module, three days apart. It measured near-zero on this warp because the
Jacobian minimum sits where `region` is already 0 or 1, but it weakened the correction along the
seam. The operation now measures on the contained field and hands `apply_warp` the raw one.

**Measure both ends the same way.** The first version read the baseline as the full extent of the
nose ring and the result as the distance between two handle-group means. Two different
estimators, so the rescaling loop was correcting a quantity that contained a constant offset as
well as the narrowing, and the achieved reduction overshot the request by about 19 percent at
every strength while converging on nothing. This engine has now made that mistake twice; the
first time was measuring `skin_smoothing` by re-decomposing its own output.

## The ceiling is a property of the face, not a setting

`max_alar_narrowing` was 0.12, chosen by judgement rather than measurement, and it was too timid:
at strength 0.5 that is a 6 percent change on the reference portrait which is very hard to see.

Raising it to a fixed 0.18 is not safe either, and the measurement is unambiguous. Across six
portraits at the default radius, 0.18 produced a worst-case minimum Jacobian of **0.004** on one
face, one step from melting, while the reference portrait sat at a comfortable 0.22. **How far a
nose can be narrowed before the field folds is a property of that nose**, so a fixed fraction is
the wrong parameterisation on its own.

The operation now backs off until the measured Jacobian clears `min_jacobian` (0.25), which is
the margin in front of `apply_warp`'s absolute refusal to fold. Two nested calibrations: the
inner one scales the handle offsets until the measured landmark movement matches the request,
the outer one reduces the request itself when this face cannot take it.

| portrait | requested | achieved | min Jacobian | fold-limited |
|---|---|---|---|---|
| reference | 0.180 | 0.157 | 0.325 | yes |
| blonde | 0.180 | **0.180** | 0.268 | no |
| laotian | 0.180 | 0.122 | 0.325 | yes |
| laughing woman | 0.180 | 0.139 | 0.325 | yes |
| woman | 0.180 | 0.152 | 0.325 | yes |
| young man | 0.180 | 0.143 | 0.325 | yes |

Worst case across all six went from a Jacobian of 0.004 to 0.268, with the leak into eyes and
lips still 0.00 8-bit levels and the worst displacement 0.055 of the interocular distance against
a 0.08 rail. `fold_limited` is reported in `stats` so a caller can tell "this nose cannot take
more" apart from "this setting is too low", the same distinction `reduce_dark_circles` draws with
its gamut headroom.

**Widening the radius is not the escape.** It does raise the Jacobian, but at radius 0.20 the
warp starts moving the lips (1.84 levels worst case across the six), because the containment
region is a dilated nose that reaches them.

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
