# ALGORITHMS

Formulas, parameter meanings, and the measurements each choice was calibrated against.

Notation: `L`, `a`, `b` are CIELAB channels (`L` in `[0,100]`, `a`/`b` in `[-127,127]`).
`G(x, g, r, e)` is the guided filter of `x` guided by `g` with radius `r` and regularization
`e`. `IOD` is the interocular distance in pixels. Every radius below is a fraction of `IOD`,
never an absolute pixel count, which is what makes a given strength look the same at any
resolution.

---

## 0. Shared primitives

### Guided filter (He, Sun, Tang, ECCV 2010)

Implemented from the paper over `cv2.boxFilter`, in `core/ops.py`. For a window of radius `r`:

```
mean_g = box(g, r)          mean_x = box(x, r)
cov    = box(g*x, r) - mean_g*mean_x
var    = box(g*g, r) - mean_g*mean_g
A      = cov / (var + e)    B = mean_x - A*mean_g
out    = box(A, r) * g + box(B, r)
```

`e` has the units of `g` squared, so **`sqrt(e)` is the contrast at which the filter switches
from smoothing to preserving**. That interpretation is what makes the parameters tunable rather
than magic. `subsample > 1` computes `A` and `B` on a downscaled image (the fast guided filter,
He and Sun 2015); measured mean deviation from the full version is under 0.02 on unit-range
data, and it is roughly three times faster.

Chosen over bilateral filtering, anisotropic diffusion and wavelets. Reasoning in
`docs/RESEARCH.md` section 4.1 and 4.3.

### Three-band decomposition

```
fine_base = G(L, L, r_fine, e_fine)
base      = G(L, L, r_base, e_base)
mid       = fine_base - base
fine      = L - fine_base
```

`base + mid + fine == L` exactly (verified by a test to 1e-5). Both cutoffs are edge-aware,
which is why this replaces a wavelet decomposition: reinjecting detail next to an eyelid does
not ring.

### Normalized low-pass

```
lowpass(x, w, r) = gaussian(x*w, r) / gaussian(w, r)
```

A smooth low-pass that reads only from where `w > 0`. Used for two different jobs: estimating
illumination over the face (where smoothness is required, see 4.), and interpolating a repaired
surface from the skin around a blemish (where excluding the blemish is required, see 5.).

### Guided feathering

```
alpha = clip(G(mask, luma(rgb), r_feather, 1e-3), 0, 1)
```

The mask boundary snaps to real image edges, so a coarse polygon does not bleed onto an
eyebrow. This is the "guided feathering" step FabSoften describes.

**A feathered mask must be re-bounded.** Feathering is a low-pass, so it pushes the mask outward
wherever the image has no edge to snap to. Measured: the skin mask reached alpha 0.46 outside
the face oval, and the under-eye region reached 0.22 over the eye aperture. Both bounds are
therefore re-applied after feathering. Two tests pin this.

**Do not guided-feather a mask whose boundary is soft anatomy.** The whole point of this
operation is that the alpha follows image edges, which means the alpha *carries image texture*.
Compositing `original*(1-a) + edited*a` then imprints a copy of that texture, scaled by the size
of the edit. Measured on a texture-free lift through an infraorbital mask: guided feathering
scored 0.902 high-frequency retention on one portrait and 1.097 on another (the sign flips with
the subject), while a plain blur stayed within half a percent of 1.0 on both. Use it where the
mask must avoid something with a real edge (hair, lips, the face oval); use `blur_mask` where
the region's boundary is a soft gradient with nothing to snap to. `reduce_dark_circles` is the
worked example, in section 3.

**Scale the radius to the region, not to the interocular distance.** The infraorbital region is
about `0.42 * eye_width` tall; a feather radius of `0.09 * IOD` was roughly half of that, and it
smoothed the alpha to a peak of 0.40, silently capping every correction at 40 percent of its
nominal value. A mask that never reaches 1.0 anywhere is nearly always a bug.

### Color management

Working buffer: `float32`, RGB, sRGB-encoded, `[0,1]`. Decisions:

- **Perceptual operations in CIELAB.** Blemish detection, chroma work, teeth classification.
- **Photometric operations in linear light.** A gain on sRGB-encoded values is not an exposure
  change; it shifts saturation. Only the lighting tool converts, and it converts explicitly.
- **One Lab roundtrip per tool.** OpenCV's `float32` Lab conversion is not bit-exact
  (measured max error 0.0016 in `[0,1]`, about 0.4 of an 8-bit level). One roundtrip is
  invisible; ten would not be. Each tool converts once, edits, converts back once.
- **No intermediate quantization.** The buffer stays `float32` from load to save.
- **Soft highlight rolloff instead of clipping** wherever a value can exceed 1.0:
  `f(x) = knee + h*(x-knee)/(h + (x-knee))` with `h = 1-knee`. Monotonic, `f(knee)=knee`,
  asymptotic to 1.

---

## 1. `whiten_teeth`

### Mask

The mouth opening is the filled inner-lip landmark ring, eroded by `0.015*IOD`. Inside it,
teeth are scored **relative to the measured lip color**, so the operation is independent of
skin tone and white balance:

```
L_high = percentile(L, mouth, 92)         a_lip = percentile(a, lips, 55)
bright  = smoothstep(L, 0.38*L_high, 0.66*L_high)
not_red = 1 - smoothstep(a, 0.42*a_lip, 0.88*a_lip)
teeth = erode_and_feather(bright * not_red * mouth)
```

Two tests only, and the second one is redness rather than chroma. A "low chroma" term was tried
first and removed: chroma in general is not a teeth-versus-not-teeth signal, because yellow teeth
have plenty of it, and the term rejected exactly the discolored areas the tool exists to correct.
Measured on a subject with yellow teeth it reduced the mask to the whitish specular highlight on
each tooth. What separates teeth from gums, tongue and inner lip is the `a*` axis.

Three confidence tests then gate the tool, all of which must pass:

| test | floor |
|---|---|
| absolute area | `max(24, (0.05*IOD)^2)` px |
| horizontal extent of the mask over the mouth's width | 0.35 |
| share of the mouth opening | 0.04 |

The extent test carries the weight, because it is scale invariant. A share-of-the-mouth
threshold alone does not work: on a wide grin showing tongue and dark interior, teeth are
legitimately a small fraction of the opening, and coverage measured 0.079 to 0.116 across
resolutions of the **same** photo, so a 0.10 floor made detection flip on and off with the input
size. A closed mouth fails the area test and the tool reports a no-op rather than bleaching a
tongue.

### Transform

```
b' = b * (1 - 0.55*s)                      # yellow, multiplicative
a' = a * (1 - 0.30*s)                      # red, much less
d  = s * min(6.0, 0.60*(94 - L_p60))       # absolute L* offset, headroom-limited
L' = L + d * (1 - smoothstep(L, 82, 94))   # tapered to zero near the cap
```

Two deliberate choices:

- **Multiplicative chroma, not an offset.** A pixel that is already neutral stays neutral, and a
  shadowed pixel keeps its proportion of the color, so the shading between teeth survives.
- **A near-constant `L*` offset, not a headroom-proportional gain.** A constant offset preserves
  the `L*` spread between teeth exactly. The first implementation used
  `L' = L + g*(cap - L)`, which compressed measured inter-tooth contrast from 14.7 to 9.5 `L*`
  units at full strength: visibly a flat white slab. The taper near the cap makes pure white
  unreachable by construction rather than by a clamp, and already-bright teeth barely move
  (measured `d = 0.0` on a subject whose teeth sit at `L* = 91`).

Measured at strength 1.0: mean `b*` 8.9 to 4.0, inter-tooth contrast retained at 0.73 to 1.00.

---

## 2. `skin_smoothing`

Attenuate the **mid** band of luminance and the low-frequency unevenness of chroma. Never touch
the fine band.

```
r_fine = 0.012*IOD   e_fine = 18     # sqrt(e) ~ 4.2 L*: pores stay in `fine`
r_base = 0.130*IOD   e_base = 400    # sqrt(e) ~ 20 L*: `base` is an illumination map
r_base *= 1 + 0.45*(roughness - 1)   # dynamic, see below

L'   = base + mid*(1 - 0.85*s) + fine*1.0
a'   = a_low + (a - a_low)*(1 - 0.70*s),   a_low = G(a, L, 0.140*IOD, 6)
b'   likewise
```

`fine_gain` is fixed at 1.0. Pores are never reduced at any strength, which is a structural
property rather than a tuning result. Chroma is smoothed harder than luminance because uneven
skin tone lives almost entirely in `a*`/`b*` while skin texture lives in `L*`.

### Dynamic parameters

In the spirit of FabSoften's attribute-aware smoothing, the base radius grows with the measured
roughness of this particular skin:

```
roughness = clip(mean(|L - G(L, L, r_base, e_base)|, over skin) / 1.6, 0.4, 2.5)
```

so a coarse-textured face gets a wider mid band than a smooth one at the same strength.

### Calibration

Both cutoffs were measured, not guessed, using two independent metrics: blotch energy from a
**fixed linear** band-pass (a difference of Gaussians, the same operator before and after) and
pore energy from a Laplacian.

| `r_base` | result at `s=0.6` |
|---|---|
| 0.055 | effect invisible: the mid band was too narrow to contain actual blotches |
| 0.130 | blotchiness clearly reduced, texture intact |
| 0.200 | skin starts to read as creamy |

| `e_fine` | `e_base` | blotch (rel) | pore (rel) |
|---|---|---|---|
| 1.2 | 9 | 0.89 | 1.07 |
| 1.2 | 250 | 0.38 | 0.55 |
| 6 | 250 | 0.45 | 0.82 |
| **20** | **250** | **0.51** | **0.99** |
| 20 | 1000 | 0.41 | 0.88 |

The first values tried (`e_fine = 1.2`, `e_base = 9`) reduced blotch energy by only 11 percent
at full strength, because at `sqrt(9) = 3 L*` the base layer preserves exactly the contrast
range blotches live in, so they never entered the mid band at all.

End-to-end at strength 1.0 through the feathered mask: blotch energy 0.61 to 0.70 of the
original, pore energy 0.80 to 1.00.

**Do not measure this tool by re-decomposing its own output.** The guided filter is data
dependent, so attenuating the mid band redistributes energy between the bands on
re-decomposition and the measurement understates the effect by an order of magnitude. Use a
fixed linear band-pass. A test documents this.

---

## 3. `reduce_dark_circles`

Per eye, independently, so asymmetric lighting cannot make one side wrong.

```
L_ref = percentile(L, cheek_reference, 55)
a_ref, b_ref = percentile(a, ref, 50), percentile(b, ref, 50)

deficit     = max(0, L_ref - L)
deficit_low = max(0, NLP(deficit, support, 0.13*IOD))          # normalized low-pass
headroom    = max(0, (L + 16) * (cbrt(1/max_linear_channel) - 1))
headroom_lo = box(headroom, 0.13*IOD)

lift = min( 0.60*s * deficit_low,  12.0,  0.70 * headroom_lo )
L'   = L + lift * support                                       # support is 0/1
a'   = a + 0.55*s * NLP(a_ref - a, support, 0.13*IOD) * support
b'   likewise
out  = composite(original, L'a'b', alpha=region)                # the soft alpha, applied ONCE
```

The cheek reference is the band immediately below the infraorbital region, built with the same
landmark interpolation but at 1.05 times the eye width, minus the region itself.

Four things make this read as retouching rather than concealer:

1. **Only the low-frequency deficit is corrected.** The high-frequency part of the deficit is
   the crease, the lash shadow, the edge of the eye bag and the skin texture. Leaving it alone
   is what preserves the anatomy.
2. **Every limit is one-sided and low-frequency.** See the rewrite notes below for why a
   pointwise limit is a texture-flattening operator rather than a safety rail.
3. **The lift is bounded by the remaining sRGB gamut headroom**, so it can never clip a
   channel. `L*` depends on luminance alone, so scaling all three channels to put the
   brightest exactly on the gamut edge multiplies `L* + 16` by the cube root of that scale,
   which gives the headroom in `L*` units in closed form and costs one pass.
4. **Chroma is also low-passed** before being pulled toward the reference, so the fine bluish
   veining that makes an eye area look real is not erased.

Measured across six portraits at strength 1.0: peak alpha 1.00 on every subject, delivered
recovery 0.24 to 0.50 of the measured deficit (ceiling 0.60, so reduction not removal holds),
residual deficit 1.6 to 22.5 `L*`, high-frequency retention 0.999 to 1.025, clipped pixels
introduced at most 0.2 percent of the region.

### The 2026-08-31 rewrite, and why each of the five defects was invisible

This tool was measured end to end and found to be delivering **2.6 percent** of the deficit on
a real portrait while its own stats reported 47 percent. Five independent defects compounded.
None of them raised an error, a warning, or a bad statistic. Each is now covered by a test that
fails without its fix (see `tests/test_tools.py`, verified by reintroducing each defect in turn).

| # | defect | effect | evidence |
|---|---|---|---|
| 1 | the soft region was multiplied in **twice**, once into the Lab edit and again by `apply_masked` | delivered `lift * alpha**2` | true recovery 0.026 vs 0.467 reported |
| 2 | the alpha was built with `feather_mask` | guided feathering makes alpha follow image edges, so compositing imprints a copy of the skin texture scaled by the lift | texture-free +9 `L*` control: 0.902 on one portrait, **1.097** on another, 0.993 to 1.005 with a plain blur |
| 3 | feather radius was `0.09 * IOD`, about half the region's own height | smoothed the alpha to a peak of 0.40, capping every correction at 40 percent | peak alpha 0.40 -> 1.00 |
| 4 | the region was gated on `skin_probability` | the model is seeded from cheek and forehead, so it penalised the dark circle *for being dark* | region 1735 px -> 760 px; the worse the defect, the weaker the mask |
| 5 | the deficit was estimated with a guided filter | edge-aware at `eps=12` means `sqrt(12) = 3.5 L*`, almost exactly skin-texture amplitude, so texture survived into the estimate and entered the lift with inverted sign; it also averaged across the region boundary | retention 0.90 to 0.96 guided, 0.95 to 1.04 with `normalized_lowpass` |

Defect 2 is the one worth remembering. **The sign of the artifact flips between subjects**, so a
one-sided texture test passes on roughly half of all portraits for entirely the wrong reason.
The regression test asserts a two-sided band.

Defect 4 generalises: a color model fitted to healthy skin will always score a defect as less
skin-like. The replacement, `infraorbital_color_gate`, drops `L*` entirely (darkness is the
thing being corrected), uses a hard rejection instead of a soft weight (a soft weight silently
scales the whole correction instead of rejecting anything), and puts a standard-deviation floor
on the chroma covariance for the same reason `SKIN_L_VARIANCE_FLOOR` exists. Across six
portraits the floor raised the region kept on the deepest circle from 70.8 to 99.6 percent while
rejecting exactly as much hair-like content.

Two further changes came out of the rewrite rather than the original diagnosis:

- **The pointwise cap against the raw deficit was removed.** It had been added to stop the
  recovered fraction reaching 0.69 against a 0.60 ceiling. It is a texture-flattening operator:
  `deficit = L_ref - L` pointwise, so the cap permits a larger lift in a dark texture trough
  than on a bright texture peak. Measured retention 0.917 with it, 0.993 without. Switching the
  estimate to `normalized_lowpass` fixed the over-recovery properly, at its source: the guided
  estimate had been spreading the deepest values across the whole region, which on a subject
  with mild circles recovered 0.90 of the deficit, effectively erasing it. It now lands at 0.49.
- **A gamut headroom limit was added.** On fair skin under warm light the red channel under the
  eye is often already near 1.0. One fixture had 20 percent of the region above 0.98, where a
  9 `L*` lift clipped **99 percent** of it into a flat patch, measuring 0.76 retention. Clipping
  is exactly the plastic look the brief forbids. `{side}_headroom_limited_fraction` is now
  reported so a caller can distinguish "this subject cannot take more" from "this setting is
  too low".

`max_lightness_delta` went 9.0 -> 12.0 only after all of the above: a deep circle measures a
16.7 `L*` deficit, of which the stated 60 percent is 10.0, so 9.0 was clipping the tool below
its own documented ceiling on precisely the cases that need it most.

**The recurring lesson, now three tools deep.** `balance_face_lighting` already replaced an
edge-aware illumination estimate with `normalized_lowpass` and got a strictly better result on
both axes. The same swap here fixed both a texture defect and an over-correction defect at once.
Guided filtering is the right tool when you are *separating* signal from an edge. It is the
wrong tool when you want an estimate that is genuinely smooth, and it is the wrong tool for a
mask alpha, because an alpha that follows image edges carries image content into whatever it
composites.

### Infraorbital contour

There is no landmark ring for it, so it is interpolated, the same way the OpenCV G-API sample
interpolates eye and forehead contours:

```
lid    = lower eyelid chain, outer corner -> inner corner
u      = (inner - outer) / |inner - outer|
n      = (-u_y, u_x), sign chosen so n points toward the chin landmark
depth(t) = 0.42 * eye_width * (0.45 + 0.55*sin(pi * t^0.85))     t in [0,1] along the lid

tail   = inner + normalize(u + 0.60*n) * (0.22*eye_width) * linspace(0,1)   # tear trough
polygon = (path + 0.06*eye_width*n)  ++  reverse(path + depth*n)            path = lid ++ tail
```

The sine profile is shallow at both corners and deepest under the middle of the eye, which is
where the shadow actually sits. Using the lid normal rather than image rows keeps it correct on
a rolled or upside-down face.

The lower-lid ring ends at the inner canthus, but the nasojugal fold continues down and toward
the nose, and that is often the darkest part of the whole area: measured on one portrait, 550 px
at a 16.1 `L*` deficit sitting entirely outside the region, against 18.7 `L*` inside it. Without
the tail the tool brightens the flat infraorbital area and leaves the trough, which reads as an
unfinished retouch. The tail is short and its depth tapers, on purpose: further medially and
steeper lies the nasal sidewall shadow, which is facial *form*, not a blemish. At an extension
of 0.35 the mask visibly reached the nose crease; 0.22 stops short of it on every fixture.

The alpha is a plain blur at `0.15` of the region's **own depth**, never a guided feather, and
never a radius expressed as a fraction of the interocular distance. See defects 2 and 3 above.

---

## 4. `balance_face_lighting`

A homomorphic correction in linear light.

```
Y   = luma_709(srgb_to_linear(rgb))
low = lowpass(log Y, face_weights, 0.42*IOD)        # smooth, NOT edge-aware
tgt = percentile(low, skin, 50)

dev = tgt - low
g_log = dev * 0.38*s*1.30   where dev > 0     # shadows lifted slightly more
      = dev * 0.38*s        where dev <= 0
g_log += log1p(-0.45*s*specular)
g     = clip(exp(soft_limit(g_log, ln 1.42, ln 1/0.72)), 0.72, 1.42)

out = linear_to_srgb(soft_rolloff(linear * g, knee=0.88))
```

with `soft_limit(x, up, down) = limit * tanh(x / limit)`.

Four things this gets right, three of which were found by measuring a wrong version first:

1. **The illumination estimate must be smooth.** An edge-preserving guided filter was used
   first. Because its estimate snaps to the image's own edges, the resulting gain field opposed
   those edges and cut measured local contrast to 0.65: that is flattening the picture, not
   evening out the lighting. Switching to a smooth normalized low-pass raised local contrast
   retention to 0.91 to 1.00 **and** increased the illumination-spread reduction (2.30 to 1.71
   stops, against 3.22 to 2.76 before). Strictly better on both axes.
2. **Specular compression happens in the log domain, before the limiter.** Applying it after the
   clamp broke the clamp: the effective gain reached 0.32 against a documented floor of 0.72.
3. **The specular knee is relative to the face's own bright skin** (the 75th percentile of
   smoothed sRGB luma inside the skin mask), not absolute. An absolute knee of 0.78 fired across
   the whole face of a brightly lit light-skinned subject.
4. **`tanh` saturation before the hard clamp.** A hard clamp alone puts a discontinuity in the
   gain field exactly where the gain is most extreme, which is visible as an edge.

The exponent is small by design. Fully flattening illumination removes the modelling that makes
a face look three-dimensional, which is the "grey skin" failure. The mask is `face_photometric`,
wider than the skin mask: it includes the nose, chin, lips and eyebrows, because a low-frequency
gain field with a narrow mask produces a visible seam. Eyebrow **exposure** therefore changes
and eyebrow **texture** does not; a test asserts the second part.

### Measuring this tool

Local contrast must be measured on `log` linear luminance. A smooth multiplicative gain is an
additive offset in log space, and a Laplacian is blind to a smooth offset, so the metric is
invariant to any legitimate exposure change. The same measurement on `L*` reads 0.65 to 0.78 for
a correct result, because CIELAB is a power law of luminance. Verified: a uniform 1.3x linear
gain scores 0.98.

---

## 5. `skin_cleanup`

### Detection

Multi-scale blob detection on `L*` and `a*`, restricted to the skin mask. Scales are
`(0.006, 0.012, 0.020, 0.032, 0.048) * IOD`.

```
r = IOD*scale/2       ring = max(r+2, 3r)
annulus(x) = (box(x, ring)*(2*ring+1)^2 - box(x, r)*(2*r+1)^2) / ((2*ring+1)^2 - (2*r+1)^2)

dark  = annulus(L) - box(L, r)
red   = box(a, r) - annulus(a)
resid = L - annulus(L)
sigma = sqrt(box(resid^2, 0.18*IOD)) + 0.35
score = max(dark/sigma, red/(0.60*sigma + 0.35))
depth = max over scales of (annulus(L) - L)        # pointwise, for the mole test
```

Three details, each of which was a bug in a previous version:

- **The annulus strictly excludes the centre block.** A plain difference of two concentric box
  filters lets the blemish contaminate its own reference. Measured maximum response at the
  largest scale: 1.18 for the concentric form, 2.29 for the annulus.
- **`sigma` measures local texture, not local variance.** The plain local standard deviation
  measures shading: it returned a median of 8.8 `L*` where the texture sigma is 3.2, and the
  detector never fired at all.
- **Mole depth is pointwise and taken as the maximum over scales.** Measured on synthetic moles,
  a box-filtered depth reported 6.9 `L*` for a mole that is 22.6 deep, and recording depth at
  the score-maximizing scale reported 12.8 for a 3 px mole, because at the finest scale the
  annulus falls inside the mole. Either mistake lets moles be healed.

Threshold: `2.6 + (1.5 - 2.6)*s`, in units of sigma. Calibrated against measured distributions:
skin sits at a median score near 0.1 with a 99.9th percentile near 1.9, and genuine blemishes
reach 2.5 to 8.

### What is kept

Per connected component:

| filter | rule |
|---|---|
| permanent | 95th-percentile `depth` >= 17 `L*` (mole, strong freckle) |
| too large | area > `pi*(0.0375*IOD)^2`, or extent > `0.15*IOD` |
| too elongated | bounding-box aspect ratio > 3.2 (wrinkle, hair strand, crease) |
| too small | area below the smallest scale's footprint |
| budget | total accepted area capped at 6 percent of the skin mask, strongest first |

Beard, brows, lashes, lips, nostrils and eyes are excluded by the mask, not by the detector.

The area budget is the guard against indiscriminate erasure: a detector failure becomes a weak
result, never a destroyed face.

The safety margin dilated around a permanent core is only `0.025*IOD`. It used to be
`1.6 * max(scale) = 0.077*IOD`, which is 32 px on a 185 px interocular face, and that was
measured swallowing every genuine blemish within 17 to 27 px of a mole (three of six synthetic
blemishes lost to mole exclusion zones alone). The real protection is the per-component test: a
mole's soft rim is connected to its core and is rejected with it.

### Healing

```
surface  = G(channel, L, 0.010*IOD, 1.2)          # everything above the pore scale
donor    = skin * (1 - dilate(blobs, 0.020*IOD))
repaired = lowpass(surface, donor, max(0.050*IOD, 2*r_heal))
channel' = channel + 0.92*s * (repaired - surface)
alpha    = feather(dilate(blobs, 0.012*IOD)) * skin
```

The fine detail layer, `channel - surface`, is never modified. The blemish disappears because
its energy lives in the surface; the pores survive because the detail layer is untouched. There
is no cloning, so there is no repeated-texture artifact.

Two calibrations:

- **The cutoff must be at the pore scale, not the blemish scale.** At `0.030*IOD` a blemish
  landed partly in the detail layer, which is preserved on purpose, and only about 40 percent of
  a synthetic spot was removed even at full strength.
- **The donor exclusion must be clearly larger than the detected core.** The detector fires on a
  blemish's centre while the blemish extends further. Growing only `0.006*IOD` left the blemish
  inside the donor region, so the repair read it straight back in and removed 27 percent of it.
  At `0.020*IOD` most of it is removed.

### Known accuracy

Recall depends strongly on the skin's own texture, which follows from normalizing by local
texture rather than by an absolute contrast. Measured with 6 synthetic blemishes at 10 `L*`
depth per fixture, at 1400 px:

| subject | blemishes found | mole area touched |
|---|---|---|
| smooth young skin | 6 / 6 | 0.0 % |
| heavily freckled skin | 3 / 6 | 11.5 % of one mole |
| dark, low local contrast | 0 / 6 | 0.0 % |
| mid-tone, textured | 2 / 6 | 0.0 % |

On heavily freckled or very dark low-contrast skin, a mild spot genuinely is not statistically
unusual for that skin. This is a property of the method, not a defect to be threshold-tuned
away: lowering the threshold to catch those cases would start removing real skin. Blemish
detection also needs the face to be reasonably large; below roughly 150 px of interocular
distance a spot is only a couple of pixels across and is not separable from texture.

---

## 5c. `equalize_skin_tone`

```
target = percentile(a, skin, 50)
field  = NLP(a, skin, 0.30*IOD)                  # normalized low-pass over the skin only
a'     = a + clip(s * (target - field), -8, +8)
b'     likewise.   L is copied through untouched.
```

Promoted from `scripts/` into the engine on 2026-08-31. It corrects **regional** colour, which
`skin_smoothing` structurally cannot: that tool pulls each pixel toward its own local mean, and
when a whole cheek is uniformly red the local mean is red too, so there is nothing to pull
toward. Measured on a portrait with ruddy cheeks, region range of median `a*` 6.5 originally:
5.8 with the default chroma radius, 4.5 with the chroma radius widened to 0.45 and attenuation
0.90, **3.2** with this. Widening the radius is not a fix, it is a weaker version of the wrong
operation.

Two properties make it safe, and both are structural rather than tunable limits:

* **`L*` is never written.** Modelling, texture and the three-dimensional form of the face are
  untouched by construction. A test pins it (`test_skin_tone_never_touches_luminance`), and that
  test matters more than it looks: a tool that writes a smooth low-frequency field to `L*` over
  the skin *is* a dodge-and-burn operator, which is exactly the photometric reshaping section 5b
  exists to prevent. This tool stays on the safe side of that line by only ever touching chroma.
* **The field is low-frequency**, so detail cannot change. Measured pore energy 4.91 to 4.91,
  unchanged to three significant figures at every strength.

Runs after `skin_smoothing`, because it fixes what smoothing cannot, and before
`reduce_dark_circles`, because that measures its reference against cheek colour and a settled
cheek is a more stable reference.

**A guardrail caught the first implementation.** `_region_spread` originally probed five discs at
landmark positions (forehead, nose, both cheeks, a temple) to report the defect as one number.
`test_tools_read_only_masks_and_geometry_from_the_analysis` rejected it, correctly: a tool that
can index `analysis.landmarks` can impose a geometric profile. The replacement measures the 5th
to 95th percentile range of the low-frequency field itself, which needs no landmarks and is a
better metric anyway, because it covers the whole skin instead of five sample points.

---

## 5b. Photometric reshaping, and three detectors that did not work

Recorded because the negative result is the useful part, and because it is the kind of thing
someone will try to rebuild.

**The gap.** Every geometric guardrail in this engine checks that no pixel moved. A retoucher
does not need to move a pixel to reshape a face. Dodge and burn narrows a nose by lightening the
dorsum and darkening the sidewalls, defines a jaw by deepening its shadow, reshapes a brow by
lightening one edge and darkening the other. Perceived anatomy changes; every existing guardrail
passes. It nearly happened by accident: the under-eye medial extension was cut at `0.22` of the
eye width because at `0.35` the mask reached the nasal sidewall, and brightening there slims the
nose. Nothing enforced that; it was a judgement call.

**Control used to test each detector.** A real dodge-and-burn sculpt built from landmark
geometry: a Gaussian ridge on the feature midline minus two Gaussian troughs on the flanks,
applied as a multiplicative gain in linear light, at 0.04 to 0.12 amplitude, for both a nose slim
and a jaw narrow. Compared against the five real tools at strength 1.0, over six portraits.

| detector | idea | result |
|---|---|---|
| standardized low-frequency log-luma correlation | a uniform gain is an additive offset in log, a uniform range compression is a scale, so standardizing removes both and only the *shape* of the shading survives | **failed.** Sculpt 0.9999 to 1.0000, `balance_face_lighting` 0.9855 to 0.9973. The attack scored better than the legitimate tool at every radius from 0.04 to 0.25 IOD and in every region. |
| gradient-direction retention | a monotonic remap preserves gradient direction exactly (`grad f(g) = f'(g) grad g`, `f' > 0`), so relighting cannot rotate a gradient but an added dipole can | **failed.** Sculpt 0.9997 to 1.0000, real tools 0.9909 to 0.9997. Same inversion. Any average over the face dilutes a local sculpt below the distributed noise of legitimate edits. |
| worst local variance of the change field | a relight is locally a near-constant gain, a dodge-and-burn is locally a dipole, so the change field should have high local variance at feature scale | **failed, by an order of magnitude and inverted.** Sculpt 0.007 to 0.020 stops, real tools 0.044 to 0.715. |

**Why they all fail, which is the real finding.** A sculpt *is* a smooth, low-frequency,
low-amplitude luminance change. That is precisely what legitimate retouching also is, and the
engine's own tools are far more locally violent than any sculpt because they work on texture and
blemishes. No statistic separates them, because the difference is not in the signal. It is in
*where the modulation field came from*.

A region-containment rule was tried next and also rejected: a candidate `facial_form` mask (the
nose flank plus the oval rim) is written into by `skin_smoothing` at up to 50 8-bit levels and by
`skin_cleanup` at up to 45, entirely legitimately, because that region is ordinary skin with
pores and blemishes on it. Containment there would break real work.

**What was shipped instead: a rule on the tool's inputs.**

- A legitimate modulation field is derived from the **image** (a local mean, an illumination
  estimate, a measured deficit) or from a mask that the perception layer built.
- A sculpt's field is derived from **landmark coordinates**: a profile imposed by anatomy
  regardless of what the pixels say.

So tools may read `analysis.masks` and `analysis.geometry` (a scalar length, for filter radii)
and nothing else, and may not construct a coordinate grid. You cannot express "lighten the
midline, darken the sidewalls" from a mask and a radius. Two AST tests enforce it, and they are
redundant on purpose: sculpting needs both the coordinates and a grid, so blocking either is
enough. Verified by writing an actual sculpt into a tool; an attempt that hid the landmark access
behind `getattr` was still caught by the grid half.

This is the same shape of rule as the geometry guardrail, and it passed on the existing code with
no changes: the five tools already only reached for `analysis.masks` (9 references) and
`analysis.geometry` (5).

**Not airtight, and the docs say so.** Someone determined could recover a midline from a mask, for
instance with a distance transform, which is why `distanceTransform` is in the forbidden list too.
The point is to make photometric reshaping something you have to deliberately work around rather
than something you can drift into.

---

## 6. Guardrails, as implemented

| requirement | how it is enforced |
|---|---|
| no geometric warping | `apply_masked` accepts only same-shape arrays and composes per pixel. An AST test fails if any tool module so much as references `warpAffine`, `remap`, `resize`, `seamlessClone`, `inpaint` or a dozen others. |
| no writes outside the mask | `apply_masked(strict=True)` measures the maximum deviation where alpha is below 1e-3 and raises. Measured end to end at maximum strength: 0.02 to 0.04 of one 8-bit level. |
| landmarks unchanged | `diagnostics.landmark_drift` re-runs the landmarker on the output. Threshold calibrated by control experiments: an identical image gives exactly 0.0, while a uniform 5 percent exposure shift or 1/255 of noise already gives 0.004 to 0.013, so the warning sits at 0.025. |
| texture preserved | `diagnostics.texture_retention`. Measured 0.93 to 1.04 at maximum strength. |
| strength 1.0 stays natural | every ceiling lives in `core/config.py` with its rationale, and the quality claims above are asserted as numbers in `tests/test_tools.py`. |
| reproducible | no randomness anywhere. Two runs are byte-identical, across engine instances too. |
