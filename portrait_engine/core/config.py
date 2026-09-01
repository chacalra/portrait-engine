"""Per-tool constants and the meaning of ``strength``.

``strength`` is always in ``[0, 1]``:

* ``0.0`` = no modification at all (the tool returns the input unchanged),
* ``1.0`` = the **maximum this engine considers safe and natural**, not the maximum the
  arithmetic allows.

Every ceiling below was chosen so that ``1.0`` still passes the quality criteria in the brief
(no plastic skin, no bleached teeth, no flat eye area, no HDR look). They are the tuning
surface of the engine: change these, not the tool bodies.

Radii and sizes are fractions of the interocular distance, never absolute pixels, so a given
strength looks the same at any resolution.
"""

from __future__ import annotations

from dataclasses import dataclass, field

#: Default execution order. Data, not code, so a future agent can propose another one.
#:
#: Cleanup precedes smoothing: smoothing a spot first turns a small dark point into a wide
#: soft blotch that is harder to remove. Dark circles come after smoothing because they read
#: the low-frequency luminance, which should already be stable. Teeth are independent.
#: Lighting is last because it is the only global operation and should judge the cleaned face.
DEFAULT_ORDER: tuple[str, ...] = (
    "skin_cleanup",
    "skin_smoothing",
    # After smoothing, because it corrects what smoothing structurally cannot: a whole cheek
    # that is uniformly red has a red local mean, so there is nothing local to pull toward.
    # Before the eye work, because that measures its reference against cheek colour and a
    # settled cheek is a more stable reference.
    "equalize_skin_tone",
    "reduce_dark_circles",
    "whiten_teeth",
    "balance_face_lighting",
)


@dataclass(frozen=True)
class SkinSmoothingConfig:
    """Three-band frequency separation. See ``docs/ALGORITHMS.md``."""

    #: Band cutoffs as fractions of the interocular distance. `fine` separates pores from
    #: blotches; `base` separates blotches from illumination. Calibrated by sweeping on real
    #: portraits: at 0.055 the mid band was too narrow to contain actual blotches and the
    #: effect was invisible even at strength 1.0. Above roughly 0.20 the skin starts to read
    #: as creamy.
    radius_fine: float = 0.012
    radius_base: float = 0.130
    #: Guided-filter regularization, in **squared** L* units, so ``sqrt(eps)`` is the contrast
    #: in L* at which each cutoff starts preserving structure instead of smoothing it.
    #:
    #: These were calibrated by sweeping on a real portrait and measuring two numbers: blotch
    #: energy in a fixed linear band-pass, and pore energy from a Laplacian. The first values
    #: tried (1.2 and 9.0) were far too small: at ``eps_base = 9`` the base layer preserves
    #: anything above 3 L* of contrast, which is exactly the range blotches live in, so the
    #: blotches never entered the mid band and the tool reduced blotch energy by only 11
    #: percent at full strength. The sweep:
    #:
    #: ==========  ==========  =============  ============
    #: eps_fine    eps_base    blotch (rel)   pore (rel)
    #: ==========  ==========  =============  ============
    #: 1.2         9           0.89           1.07
    #: 1.2         250         0.38           0.55
    #: 6           250         0.45           0.82
    #: 20          250         0.51           0.99
    #: 20          1000        0.41           0.88
    #: ==========  ==========  =============  ============
    #:
    #: ``eps_fine`` around 18 (about 4 L*) keeps pores fully intact, and ``eps_base`` around 400
    #: (about 20 L*) makes the base layer a smooth illumination map while still respecting a
    #: real edge such as the jaw.
    eps_fine: float = 18.0
    eps_base: float = 400.0
    #: At strength 1.0, attenuate the mid band by this much. 0.85 removes most blotchiness
    #: while still leaving some, which is what separates retouching from a beauty filter.
    max_mid_attenuation: float = 0.85
    #: Fine band gain. Kept at 1.0: pores are never reduced. Values above 1.0 would be
    #: FabSoften-style texture *enhancement*, which risks amplifying sensor noise.
    fine_gain: float = 1.0
    #: Chroma (a*, b*) smoothing at strength 1.0. Uneven skin tone lives in chroma, and skin
    #: texture does not, so chroma can be smoothed harder than luminance.
    max_chroma_attenuation: float = 0.70
    radius_chroma: float = 0.140
    eps_chroma: float = 6.0
    #: Dynamic parameter adaptation, in the spirit of FabSoften's attribute-aware smoothing:
    #: rougher measured skin gets a slightly larger base radius.
    dynamic_radius_gain: float = 0.45


@dataclass(frozen=True)
class TeethWhiteningConfig:
    #: Multiplicative reduction of b* (yellow) at strength 1.0. Multiplicative, not an offset,
    #: so shadows between teeth keep their relative color and cannot go flat.
    max_yellow_reduction: float = 0.55
    #: Multiplicative reduction of a* (red), smaller: teeth are yellow, not red.
    max_red_reduction: float = 0.30
    #: L* lift at strength 1.0, as an **absolute** offset in L* units, tapered to zero as a
    #: pixel approaches `l_cap`. A near-constant offset preserves the L* spread between teeth
    #: exactly, which a headroom-proportional gain does not: measured on a test portrait, a
    #: proportional lift of 0.35 compressed inter-tooth contrast from 14.7 to 9.5 L* units.
    max_lightness_delta: float = 6.0
    #: The lift is also limited to this fraction of the distance from the teeth's own
    #: lightness to `l_cap`, so already-bright teeth barely move.
    lightness_headroom_fraction: float = 0.60
    #: Ceiling. Pure white (L*=100) is unreachable by construction, not by a clamp.
    l_cap: float = 94.0
    #: Width of the taper below `l_cap`, in L* units.
    l_taper: float = 12.0
    #: Confidence floor for the teeth mask, expressed three ways. Below any of them the mask is
    #: considered unreliable and the tool becomes a no-op instead of bleaching a tongue.
    #:
    #: A share-of-the-mouth threshold alone does not work, and this was measured: on a wide grin
    #: showing tongue and dark mouth interior, teeth are legitimately a small fraction of the
    #: opening, and coverage hovered between 0.079 and 0.116 across resolutions of the *same*
    #: photo, so a 0.10 floor made detection flip on and off with the input size. The horizontal
    #: extent test is scale invariant and semantically right: a real set of teeth spans the
    #: mouth, a stray bright blob does not.
    min_mouth_coverage: float = 0.04
    #: Minimum teeth area, as a fraction of the interocular distance, squared.
    min_area_fraction: float = 0.05
    #: The teeth mask must span at least this share of the mouth opening's width.
    min_width_ratio: float = 0.35


@dataclass(frozen=True)
class DarkCircleConfig:
    #: Fraction of the measured low-frequency luminance deficit corrected at strength 1.0.
    #: Capped well below 1.0 on purpose: this is a reduction, not a removal, and a residual
    #: deficit at strength 1.0 is a tested invariant.
    #:
    #: A pointwise clamp against the **raw** deficit used to be applied here as well. It was
    #: removed: because `raw deficit = L_ref - L` pointwise, the clamp permits a larger lift in
    #: a dark texture trough than on a bright texture peak, so it compresses pore-scale
    #: contrast (measured high-frequency retention 0.917 with it, 0.993 without). The lift is
    #: now purely low-frequency, which cannot alter texture at all.
    max_luminance_recovery: float = 0.60
    #: Fraction of the chroma deviation (blue, purple, red cast) neutralized at strength 1.0.
    max_chroma_recovery: float = 0.55
    #: Only the low-frequency part of the deficit is corrected, at this radius. That is what
    #: keeps creases, the eye bag edge and skin texture intact.
    #: Estimated with `normalized_lowpass`, not a guided filter: the estimate must contain no
    #: texture at all, and must not average across the region boundary. See the tool.
    deficit_radius: float = 0.13
    #: Absolute ceiling on the L* lift, so a very dark shadow cannot become a bright patch.
    #: Raised from 9.0: a deep dark circle measures a 16.7 L* deficit, of which the stated
    #: 60 percent is 10.0, so 9.0 was clipping the tool below its own documented ceiling on
    #: exactly the cases that need it most. 12.0 leaves the ceiling reachable while still
    #: bounding a shadow across half a face.
    max_lightness_delta: float = 12.0
    #: Fraction of the *remaining gamut headroom* the lift may consume. On fair skin under warm
    #: light the red channel is often already near 1.0 under the eye: on one fixture 20 percent
    #: of the region had a channel above 0.98, and a 9 L* lift clipped 99 percent of it, which
    #: flattens texture into a flat patch (measured high-frequency retention 0.76). This limit
    #: is a low-frequency field, unlike the pointwise deficit clamp that was removed, so it
    #: bounds the lift without touching texture.
    gamut_headroom_fraction: float = 0.70
    #: The cheek band used as the "healthy skin" reference, below the under-eye region.
    reference_offset: float = 0.30
    reference_height: float = 0.28


@dataclass(frozen=True)
class SkinToneConfig:
    #: Radius of the regional colour field, as a fraction of the interocular distance. Must be
    #: large: it has to average over a whole region, not over a blotch. At a small radius this
    #: degenerates into a weaker copy of `skin_smoothing`, which already does the local job.
    field_radius: float = 0.30
    #: Hard ceiling on the a*/b* shift, so a mis-segmented skin mask cannot recolour a face.
    max_shift: float = 8.0


@dataclass(frozen=True)
class FaceLightingConfig:
    #: Exponent applied to the illumination-flattening gain at strength 1.0. Low on purpose:
    #: fully flattening illumination destroys the three-dimensional form of a face.
    max_flatten_exponent: float = 0.38
    #: Hard clamp on the per-pixel gain in linear light. Prevents halos and the HDR look.
    gain_min: float = 0.72
    gain_max: float = 1.42
    #: Radius of the illumination estimate. Large: this must capture lighting, not features.
    illumination_radius: float = 0.42
    #: Specular highlight taming. The knee is **relative** to the face's own bright skin (the
    #: 75th percentile of smoothed sRGB luma inside the skin mask), not absolute: an absolute
    #: knee fired across the whole face of a brightly lit subject.
    max_specular_compression: float = 0.45
    #: Floor for the relative knee, so a very dark photo does not get its skin compressed.
    specular_knee_floor: float = 0.55
    #: How far above the knee the compression ramps in, in sRGB luma units.
    specular_excess_low: float = 0.055
    specular_excess_high: float = 0.20
    #: Shadow lift applied to the low-frequency component only, so local contrast survives.
    max_shadow_lift: float = 0.30


@dataclass(frozen=True)
class SkinCleanupConfig:
    #: Blemish diameters to look for, as fractions of the interocular distance. A pimple is
    #: roughly 1 to 5 percent of the interocular distance on a normal portrait.
    scales: tuple[float, ...] = (0.006, 0.012, 0.020, 0.032, 0.048)
    #: Detection threshold in units of **local texture** standard deviation, at strength 0 and
    #: at strength 1.0. Higher strength lowers the threshold, so more spots qualify. Calibrated
    #: against measured score distributions on real portraits: skin sits at a median score near
    #: 0.1 with a 99.9th percentile near 1.9, and genuine blemishes reach 2.5 to 8.
    threshold_high: float = 2.6
    threshold_low: float = 1.5
    #: Window for the texture-sigma estimate, as a fraction of the interocular distance.
    texture_window: float = 0.18
    #: Floor added to sigma, so perfectly smooth skin cannot produce an infinite score.
    sigma_floor: float = 0.35
    #: Redness responses are normalized by a smaller multiple of sigma, because a* varies less
    #: across skin texture than L* does.
    redness_sigma_scale: float = 0.60
    #: Fraction of the blemish's low-frequency deviation removed at strength 1.0.
    max_correction: float = 0.92
    #: A blob darker than this many L* units below its surroundings is treated as a mole or a
    #: freckle worth keeping, not a temporary blemish.
    mole_darkness_l: float = 17.0
    #: Safety margin around a deep-dark core, as a fraction of the interocular distance.
    #:
    #: Small on purpose. The real protection is the per-component rejection in
    #: `_filter_components`: a mole's soft rim is connected to its dark core, so it lands in the
    #: same component and is rejected with it. This margin only covers a rim pixel that a
    #: one-pixel gap disconnected from the core.
    #:
    #: It used to be 1.6 times the largest blob radius, which is 32 px on a 185 px interocular
    #: face. That was measured swallowing every genuine blemish within 17 to 27 px of a mole:
    #: three of six synthetic blemishes were lost purely to a mole's exclusion zone.
    mole_protect_margin: float = 0.025
    #: A blob larger than this fraction of the interocular distance is anatomy, not a blemish.
    max_blob_fraction: float = 0.075
    #: Total area cap: the tool refuses to edit more than this fraction of the skin mask,
    #: which is the guard against indiscriminate erasure.
    max_area_fraction: float = 0.06
    #: Radius separating the preserved **fine texture** from the repaired **surface**. It must
    #: be at the pore scale, not the blemish scale: at 0.030 a blemish landed partly in the
    #: detail layer, which is preserved by design, and only about 40 percent of a synthetic spot
    #: was removed even at full strength. At the pore scale the whole blemish lives in the
    #: surface and can be repaired, while the pores stay in the detail layer.
    heal_radius: float = 0.010
    #: Guided-filter regularization for that split, in squared L* units.
    heal_eps: float = 1.2
    #: How far the detected core is grown before it is removed from the donor region, as a
    #: fraction of the interocular distance. This has to be clearly larger than the detected
    #: core: the detector fires on a blemish's centre, while the blemish itself extends further,
    #: and if the donor still contains the blemish the repair reads it back in. Measured on
    #: synthetic spots, growing only 0.006 removed 27 percent of a blemish; 0.020 removes most.
    donor_dilate: float = 0.020
    #: How far the core is grown for the compositing alpha. Smaller than `donor_dilate`, so the
    #: repair reads from clean skin but is applied only over the blemish.
    alpha_dilate: float = 0.012


@dataclass(frozen=True)
class EngineConfig:
    """Everything tunable, in one place."""

    skin_smoothing: SkinSmoothingConfig = field(default_factory=SkinSmoothingConfig)
    teeth_whitening: TeethWhiteningConfig = field(default_factory=TeethWhiteningConfig)
    dark_circles: DarkCircleConfig = field(default_factory=DarkCircleConfig)
    skin_tone: SkinToneConfig = field(default_factory=SkinToneConfig)
    face_lighting: FaceLightingConfig = field(default_factory=FaceLightingConfig)
    skin_cleanup: SkinCleanupConfig = field(default_factory=SkinCleanupConfig)
    order: tuple[str, ...] = DEFAULT_ORDER
    #: Guided-filter subsampling factor. 2 is visually indistinguishable and roughly 3x faster.
    subsample: int = 2


DEFAULT_CONFIG = EngineConfig()
