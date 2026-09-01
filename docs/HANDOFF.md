# HANDOFF

State of the Portrait Retouching Engine, written 2026-08-31 so that a fresh session can pick up
without re-deriving anything. Read this first, then `docs/ALGORITHMS.md` for the formulas.

**Section 7 (under-eye correction) is done, 2026-08-31.** It is kept below as a record of what
was wrong and what replaced it, because the lessons generalise. The open decisions in section 6
are still open and are the natural next thing to pick up.

---

## 1. What this project is

`facr_portrait` is **Mundo 1** of a deliberate two-part split:

> AI decides what to edit. Deterministic image tools perform the edit.

This repository is the tools half. A separate agent project (Mundo 2) will drive it later. The
agent does not exist and must not be built here.

Twenty non-negotiables were set by the user. The ones that constrain design decisions most:

- preserve identity, never modify facial geometry, never change apparent age;
- no generative model of any kind, and no external API;
- work on the original photograph;
- preserve natural skin texture, avoid a plastic result **at moderate strength** (the wording
  matters, see section 6);
- do not remove structural wrinkles;
- everything parameterized, reproducible, inspectable in debug, and independently runnable;
- do not over-architect.

## 2. Current state

Complete and verified. `211 passed` in about 35 seconds, `ruff` clean.

```
portrait_engine/
  core/        types, colorspace, image, ops, masking, config, pipeline
  perception/  landmark_sets, landmarks, face_parsing (optional, off), masks, face_analyzer
  tools/       base (registry + JSON contracts), 5 tools
  debug/       visualize, recorder
  diagnostics.py, cli.py, api/app.py
docs/          RESEARCH, ARCHITECTURE, ALGORITHMS, THIRD_PARTY, HANDOFF (this file)
scripts/       fetch_test_images, evaluate_folder, render_me_pro, experiment_porcelain,
               experiment_skin_tone_equalize
tests/         7 files
```

Five tools: `skin_cleanup`, `skin_smoothing`, `reduce_dark_circles`, `whiten_teeth`,
`balance_face_lighting`. Perception is MediaPipe Face Landmarker (478 points, Apache-2.0) plus a
per-image skin color model. The BiSeNet parsing backend exists but is off and ships no weights.

Three architectural rules that must not be broken:

1. Perception knows nothing about the tools. It produces an immutable `FaceAnalysis`.
2. Tools run no models. They are pure functions over arrays.
3. **Every pixel written passes through `core.masking.apply_masked`.** That is where the identity
   guardrails live. A test fails if a tool module even references `warpAffine`, `remap`,
   `resize`, `inpaint` and a dozen similar names.

## 3. How to work here

```bash
source .venv/bin/activate            # Python 3.13, package installed with -e ".[dev,api]"
pytest -q
ruff check portrait_engine scripts tests examples
python scripts/evaluate_folder.py data/input data/output/eval --strength-sweep
portrait-retouch photo.jpg --preset natural --debug
```

Fixtures: `data/input/`, six CC-licensed Wikimedia portraits, regenerate with
`python scripts/fetch_test_images.py`. `data/` is git-ignored except `ATTRIBUTION.md`. One
fixture (`old_man_imagicity_231.jpg`) is deliberately undetectable and is the no-face test case.

**Do not put deliverables under `data/`.** It is git-ignored and some editors hide ignored
directories, which already caused a delivered file to look missing to the user. Working outputs
belong somewhere visible.

## 4. The working method the user expects

Every image-processing constant is justified by a recorded measurement, never chosen by eye.
`core/config.py` carries the rationale in comments, `docs/ALGORITHMS.md` carries the sweep tables
**and the wrong versions that were tried first**. Keep doing this. Several plausible
implementations were measurably wrong in ways invisible until measured:

| what looked fine | what measurement showed |
|---|---|
| guided-filter `eps_base = 9` | blotch energy reduced only 11 percent at full strength |
| edge-aware illumination estimate | local contrast cut to 0.65; a smooth estimate gave 0.91 to 1.00 **and** more leveling |
| box-filtered mole depth | read 6.9 L* for a mole that is 22.6 deep, so moles were healed |
| per-pixel mole rejection | 95 percent of mole area still edited (the rim formed its own component) |
| 32 px exclusion zone around moles | swallowed every blemish within 17 to 27 px of a mole |
| a "low chroma" term in the teeth mask | rejected exactly the yellow areas the tool exists to fix |
| headroom-proportional teeth L* lift | compressed inter-tooth contrast from 14.7 to 9.5 |
| 10 percent mouth-coverage floor | detection flipped on and off between 1400 and 1600 px of the same photo |

Two measurement traps worth remembering:

- **Do not measure a band-attenuating tool by re-decomposing its own output.** The guided filter
  is data dependent and the energy redistributes. Use a fixed linear band-pass.
- **Measure local contrast of a gain-based tool in the log domain.** A smooth multiplicative gain
  is an additive offset in log space, so a Laplacian is blind to it. The same measurement on `L*`
  reads 0.65 to 0.78 for a correct result.

## 5. What lives outside the engine right now

Two prototypes in `scripts/`, deliberately not imported by `portrait_engine`. Both are measured
and documented in their own module docstrings.

### `experiment_porcelain.py`

A porcelain skin look. Four of the five ingredients are existing `SkinSmoothingConfig` fields
pushed past their safe ceilings (`max_mid_attenuation` 1.00, `max_chroma_attenuation` 0.95,
`radius_base` 0.200, and critically `fine_gain` 0.35, the pore band the engine never touches).
The fifth is `bloom`, an Orton-style lift in linear light, which is what supplies the luminous
quality. Guardrails hold: leak 0.06 to 0.14 of one 8-bit level, drift 0.011 to 0.024. The cost is
pore retention, 0.94 to about 0.54 to 0.70.

It also contains `crease_soften`, **off by default and it must stay off**. It is what makes the
look work on a face with structural wrinkles, and it measurably removes 10 to 15 years of
apparent age, which violates non-negotiable 7 and the "no structural wrinkle removal" exclusion.
Without it, porcelain is convincing on subjects with little structural wrinkling and uncanny on
those without (smooth skin, intact crow's feet).

### `experiment_skin_tone_equalize.py`

`equalize_skin_tone`: pulls the very-low-frequency chroma field of the skin toward the face's own
median. **This is a genuine missing capability, not a tuning gap.** `skin_smoothing` pulls each
pixel toward its *local* mean, which does nothing when a whole cheek is uniformly ruddy, because
the local mean is ruddy too. Measured on the user's own photo (a* by region: forehead 14.2,
temple 13.4, nose 18.4, cheeks 19.9 and 19.3):

| approach | region range |
|---|---|
| original | 6.5 |
| `skin_smoothing` at default chroma settings | 5.8 |
| `skin_smoothing`, `radius_chroma` 0.45, attenuation 0.90 | 4.5 |
| `equalize_skin_tone` at 0.9 | **3.2** |

Pore energy unchanged at every strength (4.91 to 4.91). Only `a*` and `b*` move; `L*` is not
touched. Because the target is the face's own median rather than a neutral, the face-to-ear
relationship does not change (measured -1.6 before and after), so it does not create the
"corrected face, red ears" mismatch.

**Recommendation already given to the user: add this as a sixth tool.** About 20 lines, no
conflict with any non-negotiable, fits the mask contract unchanged. Awaiting their go-ahead.

## 6. Open decisions awaiting the user

0. **Facial reshaping as a v2 product.** Decided: out of scope for v1, and the photometric
   loophole is now closed. See section 8b. Nothing to do unless the user reopens it.
1. **How to expose the porcelain look**, if at all. My recommendation: a skin-look registry
   (`natural` default, `porcelain` preset) plus a `skin_glow` tool for the bloom, both visible in
   the JSON tool contracts. **Do not widen `strength`.** `1.0` must keep meaning "maximum
   natural" because the future agent will rely on that contract, and it is the anchor for
   non-negotiable 12. `crease_soften` stays out of the engine. A question was asked and the user
   did not answer it; do not assume.
2. **Whether to promote `equalize_skin_tone`** into the engine as a sixth tool. Recommended yes.
3. Where working outputs should be written by default so they are visible and optionally
   versionable (`results/`?). Currently the CLI writes to `output/`, which is git-ignored.

## 7. DONE: professional under-eye correction

Completed 2026-08-31. The full write-up, with every measurement, is in `docs/ALGORITHMS.md`
section 3. Summary of the outcome:

| | before | after |
|---|---|---|
| delivered recovery of the deficit (real, measured on the output) | 0.026 | 0.49 |
| peak mask alpha | 0.40 | 1.00 |
| high-frequency retention across six portraits | 0.90 to 0.97 | 0.999 to 1.025 |
| tear trough covered | no | yes |
| clipped pixels introduced | up to 99 percent of the region | at most 0.2 percent |

**The diagnosis in the original section 7 below was right but incomplete.** It found two mask
defects. There were five, and the two it missed were the largest. The tool was multiplying its
soft alpha in twice, delivering `lift * alpha**2`, so the real correction was 2.6 percent of the
deficit while the stats reported 47 percent. The stats read a pre-compositing intermediate,
which is how it stayed hidden. Stats are now measured from the output image.

Five things changed, each with a test that fails without it (verified by reintroducing each
defect in turn):

1. the soft alpha is applied exactly once, by `apply_masked`;
2. the infraorbital alpha is a plain blur, not a guided feather, because an edge-aware alpha
   carries image texture into whatever it composites;
3. the blur radius is a fraction of the region's own depth, not of the interocular distance;
4. `skin_probability` was replaced by `infraorbital_color_gate`: chroma only, hard rejection,
   with a variance floor;
5. the deficit is estimated with `normalized_lowpass`, not a guided filter.

Plus two that came out of the rewrite: the pointwise deficit cap was removed (it is a
texture-flattening operator, not a safety rail) and a gamut headroom limit was added (fair skin
under warm light is often already near 1.0 in the red channel under the eye).

**Two lessons worth carrying to the next tool.**

- A color model fitted to healthy skin will always score a defect as *less* skin-like. Gating a
  correction on it is self-defeating in direct proportion to how bad the defect is.
- Guided filtering is for separating signal at an edge. It is the wrong tool for an estimate
  that must be genuinely smooth, and the wrong tool for a mask alpha. This is now the third
  place in the engine where swapping it for `normalized_lowpass` improved things on every axis.

**What was deliberately not done.** Item 3 of the original plan, reducing the *contrast* of the
eye bag edge, was not attempted. It remains the riskiest idea in the file and it sits closest to
the "do not flatten the eye area" and "do not remove structural wrinkles" boundaries. Items 1
and 2 turned out to be sufficient. Do not start item 3 without a specific reason to.

---

### Original diagnosis, kept for the record

The user's words: improve dark circles and hide them as professionally as possible, treating this
as coupled with the skin work. They are happy with the current skin quality and want the eye area
brought to the same level.

### What the tool does today

`tools/dark_circles.py`. Per eye: measure a reference from the cheek band below, compute the
luminance deficit, low-pass it with a guided filter, and add back a fraction. Only the
low-frequency deficit is corrected, which is what preserves creases and the eye bag edge.
Chroma is pulled partially toward the reference. Three limits: 60 percent of the low-frequency
deficit at strength 1.0, an absolute ceiling of 9 L*, and a pointwise cap against the raw deficit.

Texture retention in the region measures 0.97 to 0.99, so the "do not flatten the eye area"
requirement is met. The problem is the opposite one: it does not do enough.

### The diagnosis, measured on `me.jpeg` (IOD 193 px)

The nominal ceiling is 60 percent recovery. Measured at strength 1.0: **47 to 50 percent** of a
16.7 L* deficit, leaving 8 to 9 L*. Visually the darkest part, right under the lash line and the
tear trough at the inner corner, is barely touched.

The cause is that the mask is far weaker than it looks. Tracing it stage by stage:

| stage | area (sum of alpha) | peak alpha |
|---|---|---|
| raw interpolated polygon | 2355 px | 1.00 |
| minus `eyes_dilated` | 1735 px | 1.00 |
| times `skin_probability` and the oval | 760 px | 0.93 |
| times `(1 - protected)` | 738 px | 0.93 |
| **after guided feathering** | **592 px** | **0.40** |

Two distinct failures, both worth fixing:

1. **The skin color model penalizes the dark circle itself.** 1735 px down to 760 px. The model
   is seeded from cheek and forehead patches, so a region that is darker and often bluer or
   purpler scores as "less skin". This is self-defeating: the worse the dark circle, the weaker
   the mask that is supposed to treat it. The under-eye region should probably not be gated on
   `skin_probability` at all, or should be gated on a version of it with the luminance term
   relaxed.
2. **Feathering collapses the peak alpha from 0.93 to 0.40.** The feather radius is `0.09 * IOD`,
   about 17 px, while the region itself is only about `0.42 * eye_width` tall, roughly 36 px. The
   mask is barely twice the feather radius, so the guided filter smooths it toward zero. The
   effective correction is therefore capped near 40 percent of what the tool computes, at any
   strength. `eyes_dilated` at `0.055 * IOD` also removes the band closest to the lid, which is
   exactly where a dark circle is darkest.

### Suggested lines of attack, in the order I would try them

1. **Fix the mask before touching the correction math.** Relax or remove the `skin_probability`
   gate for the infraorbital region, shrink the feather radius relative to the region's own
   height rather than to `IOD`, and reconsider how much of the lid margin `eyes_dilated` removes.
   Expect this alone to roughly double the effective correction, at which point the existing
   60 percent ceiling may be enough. Re-measure before changing any ceiling.
   *(Outcome: the first two were right and are done. The `eyes_dilated` change was measured and
   rejected: a smaller lid clearance made the result slightly worse, 0.432 to 0.420, because
   the region then reached into the lash shadow where the lift is unreliable. The ceiling was
   indeed left at 0.60.)*
2. **Separate the tear trough from the flat infraorbital area.** A retoucher treats them
   differently: the trough is a shadow cast by anatomy, deepest medially, and the flat area is
   pigmentation. The current sine depth profile is symmetric and does not encode this. Landmark
   geometry can distinguish them.
3. **Consider a second, gentler pass on local contrast at the eye bag edge.** Professionals
   reduce the *contrast* of the bag edge, not only the darkness. This is the riskiest idea and
   sits closest to the "do not flatten the eye area" and "do not remove structural wrinkles"
   boundaries. Do not attempt it before 1 and 2, and measure texture retention throughout.
4. Only after the above, revisit `max_luminance_recovery` and `max_lightness_delta`.

### Constraints that still apply

Reduction, not removal. There must be a residual deficit at strength 1.0, and there is a test for
it (`test_dark_circles_reduce_without_removing`). Texture retention in the region must stay above
0.9 (`test_dark_circles_preserve_texture`). The tool must never darken the region
(`test_dark_circles_only_brighten`). Do not weaken these tests to make a change pass.

### Useful artifacts already generated

At the repository root, regenerated 2026-08-31 against the finished tool:
`undereye_before_after.jpg` (both eyes at 100 percent, strength 1.0), `undereye_ladder.jpg`
(strength 0, 0.35, 0.7, 1.0), `undereye_mask.jpg` (the alpha at medial extension 0.00, 0.22 and
0.35, which is how the tear-trough extent was chosen), `me_undereye.jpg` (full frame).

## 8. The user's photo, as a worked example

`me.jpeg` at the repo root, 2774x4160. Personal, git-ignored via a root-level image rule. The
user is happy with the result and it is the reference for "professional" quality.

Diagnosis: evenly lit (0.77 stops, zero specular area), good luminance texture, so the work was
acne plus regional redness. Notable specifics: the face is small in frame, so at
`max_side=1600` the IOD is only 74 px, below the roughly 150 px that blemish detection needs.
**Process this photo at full resolution.** The subject wears orthodontic brackets; the teeth mask
includes them, and it does not matter because `b*` is reduced multiplicatively, so neutral metal
barely moves while the yellow of the tooth does.

The recipe is now a runnable script, `scripts/render_me_pro.py`, so the reference result can be
regenerated after any change to the tools:

```python
PRO = replace(DEFAULT_CONFIG.skin_smoothing, max_mid_attenuation=0.75,
              max_chroma_attenuation=0.85, radius_chroma=0.24, eps_chroma=20.0, fine_gain=1.0)

skin_cleanup          0.85     43 spots, 0.91 % of the skin
skin_smoothing        0.55     with PRO
equalize_skin_tone    0.60     experimental, from scripts/
reduce_dark_circles   0.85     was 0.30, see below
whiten_teeth          0.20
balance_face_lighting 0.15     light, the photo is already even
```

Result: texture retention 0.976, out-of-region leak 0.0175/255 with zero pixels over one level,
landmark drift 0.0090 rms against a noise floor of 0.004 to 0.013, 1.6 percent of the frame
changed, no warnings. Under-eye: a 17.1 `L*` deficit reduced by 7.2, leaving 9.9.

**On the under-eye strength.** The original recipe said 0.30, but at the time the tool delivered
2.6 percent of the measured deficit, so that number meant nothing. Against the rewritten tool the
behaviour is linear and costs nothing measurable: recovery runs 0.15 / 0.25 / 0.35 / 0.42 / 0.50
at strengths 0.30 / 0.50 / 0.70 / 0.85 / 1.00, while texture retention stays at 0.975 to 0.976
and the leak stays at 0.018 across the whole range. 0.85 was chosen by eye from the ladder as the
point where the area reads corrected rather than treated, with the eye bag edge and the creases
still fully present. 1.00 is defensible too and still leaves 8.6 `L*` of residual depth.

Output written with `keep_metadata=False`. The source carries EXIF; for a personal photo that is
worth stripping, though the engine preserves it by default, which is correct for a professional
workflow.

## 8b. Facial reshaping: out of scope for v1, deliberately

Decided with the user 2026-08-31.

**Geometric reshaping is not possible in this engine and that is by construction**, not by
omission. Three mechanisms block it: an AST test over 18 warping function names that fails the
moment one is typed in a tool, `apply_masked` raising `MaskContractError` on any shape change,
and the `landmark_drift` diagnostic. Tools are pure functions returning an array of the same
shape. This implements seven of the twenty non-negotiables.

**Photometric reshaping was the gap, and is now closed.** Dodge and burn reshapes a face without
moving a pixel, so every guardrail above passes while the perceived anatomy changes. Three
statistical detectors were built and measured against a real sculpt; all three failed, and in
each case the attack scored *better* than the legitimate tools. The full write-up is
`docs/ALGORITHMS.md` section 5b. What shipped instead is a rule on the tools' **inputs**: a tool
may read only `analysis.masks` and `analysis.geometry`, and may not build a coordinate grid,
because a sculpt's modulation field has to come from landmark coordinates while a legitimate one
comes from the image. Two AST tests, redundant by design.

**If reshaping is ever wanted, it is a separate product, not a sixth tool.**

The value of this engine is not that it retouches well. It is that it can *demonstrate* the photo
is still the same person, which is what a professional or regulated use (several jurisdictions
already require disclosure of retouched images in advertising) needs and what a generic beauty app
cannot offer. If reshaping tools enter the same registry, the Mundo 2 agent can call them and that
guarantee disappears **silently**, which is the actual risk. Not that the promise is dropped, but
that it is dropped without anyone noticing.

So: its own module, its own registry, its own opt-in, never mixed with these five. Technically it
is not hard. MediaPipe already provides 478 landmarks, so a triangulated or thin-plate-spline warp
is known work; the difficulty is making it look natural, not making it move.

## 9. Known limits of the engine, unchanged

- Hair, beard and neck separation is the weakest part of perception. No semantic segmentation by
  default; hair is inferred from an anatomical prior (beard zone plus oval rim) and relative
  local texture. A clean-shaven jaw in deep shadow is partly excluded from the skin mask.
- Blemish recall depends on skin texture: 6 of 6 on smooth young skin, 3 of 6 on heavily freckled
  skin, 0 of 6 on dark low-contrast skin. The detector normalizes by local texture, so on very
  textured skin a mild spot genuinely is not unusual. Lowering the threshold starts removing real
  skin.
- Blemish detection needs roughly 150 px of interocular distance or more.
- Teeth detection needs an open mouth; otherwise the tool reports a no-op.
- Single face only, by policy. Group photos raise rather than retouch the largest face.
- MediaPipe misses small or strongly profile faces. A two-stage detect-then-crop pass would fix
  this and is not implemented.
- Model weight licensing is documented but not legally verified. See `docs/THIRD_PARTY.md`, in
  particular the CelebAMask-HQ non-commercial restriction on BiSeNet weights.
