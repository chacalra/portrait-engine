"""Under-eye dark circle reduction.

Method
------
The single decision that makes this look like retouching rather than a concealer patch: only
the **low-frequency** part of the luminance deficit is corrected.

For each eye, separately:

1. Measure a reference from the cheek band directly below the infraorbital region: the
   luminance ``L_ref`` and the chroma ``a_ref``, ``b_ref`` of neighbouring healthy skin. This
   is per-eye, so asymmetric lighting does not make one side wrong.
2. Compute the deficit ``max(0, L_ref - L)``: how much darker each pixel is than the
   neighbouring cheek.
3. Low-pass that deficit with ``normalized_lowpass``. The high-frequency part of the deficit
   is the crease, the eyelash shadow, the edge of the eye bag and the skin texture. Leaving it
   untouched is what preserves the anatomy of the eye area. The low-pass must be genuinely
   smooth, not edge-aware: see the note in the code, and the same lesson in
   ``balance_face_lighting``.
4. Add back a **fraction** of that low-frequency deficit, at most 60 percent at strength 1.0
   and never more than 12 L* in absolute terms, further limited by the remaining sRGB gamut
   headroom so the correction can never clip a channel. This is a reduction, not a removal, by
   construction: measured across six portraits the delivered recovery runs 0.23 to 0.52, so
   the under-eye area stays measurably darker than the cheek at any strength.
5. Pull chroma partially toward the reference, which removes the blue, purple or red cast
   without changing the skin's hue.

The mask is the infraorbital region derived from the lower eyelid landmarks, then blurred at a
radius set by the region's own height. It is deliberately **not** guided-feathered: an
edge-aware alpha follows image texture and compositing then imprints a copy of that texture,
scaled by the lift.

Every limit here is one-sided and low-frequency on purpose. A pointwise limit against the raw
deficit reads brighter on a texture peak than in a trough, so it flattens the very pore-scale
contrast this tool exists to keep. All four historical defects in this tool, and their
measurements, are recorded in ``docs/ALGORITHMS.md``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from portrait_engine.core import ops
from portrait_engine.core.colorspace import lab_to_rgb, rgb_to_lab, srgb_to_linear
from portrait_engine.core.config import DEFAULT_CONFIG, DarkCircleConfig
from portrait_engine.core.masking import apply_masked, edited_area_fraction
from portrait_engine.core.types import FaceAnalysis, ToolResult
from portrait_engine.tools.base import no_op, register, resolve_image, validate_strength


@register
def reduce_dark_circles(
    image: Any,
    analysis: FaceAnalysis,
    strength: float = 0.20,
    *,
    config: DarkCircleConfig | None = None,
    subsample: int | None = None,
    strict: bool = True,
    debug: bool = False,
) -> ToolResult:
    """Reduce the visual intensity of under-eye dark circles while preserving the natural
    anatomical shadow, by correcting only the low-frequency luminance and color deficit
    relative to the neighbouring cheek.

    Args:
        strength: 0 leaves the image untouched, 1.0 recovers up to 60 percent of the measured
            low-frequency luminance deficit, capped at 12 L* and by the remaining gamut
            headroom. Full removal is not reachable and is not the goal.

    Returns:
        A :class:`~portrait_engine.core.types.ToolResult`. ``stats`` reports, per eye, the
        measured deficit, the lift actually present in the output image, and the residual
        deficit, so it can be verified that depth remains. ``{side}_headroom_limited_fraction``
        says how much of the correction the gamut ceiling refused, which distinguishes "this
        subject cannot take more" from "this setting is too low".
    """
    strength = validate_strength(strength)
    config = config or DEFAULT_CONFIG.dark_circles
    subsample = DEFAULT_CONFIG.subsample if subsample is None else subsample
    params = {"strength": strength}

    rgb = resolve_image(image)
    if strength == 0.0:
        return no_op(rgb, "reduce_dark_circles", params, "strength is zero")

    geometry = analysis.geometry
    lab = rgb_to_lab(rgb)
    lightness, a_star, b_star = lab[..., 0], lab[..., 1], lab[..., 2]

    new_l = lightness.copy()
    new_a = a_star.copy()
    new_b = b_star.copy()
    total_alpha = np.zeros_like(lightness)
    stats: dict[str, Any] = {}
    pending: list[tuple[str, np.ndarray, np.ndarray, float]] = []
    any_side = False

    radius = geometry.scaled(config.deficit_radius, 3)

    # How much L* each pixel can gain before its brightest channel leaves the sRGB gamut.
    # L* depends on luminance alone, so scaling every channel by `scale` (which places the
    # brightest channel exactly on the gamut edge) scales Y by `scale` and therefore multiplies
    # `L* + 16` by its cube root. Low-passed at the same radius as the deficit so that this
    # limit, like the deficit, contains no texture and cannot flatten any.
    scale = 1.0 / np.maximum(srgb_to_linear(rgb).max(axis=-1), 1e-4)
    headroom = np.maximum((lightness + 16.0) * (np.cbrt(scale) - 1.0), 0.0)
    headroom_low = ops.box_filter(headroom, radius)

    for side in ("left", "right"):
        region = analysis.masks[f"{side}_under_eye"]
        reference = analysis.masks[f"{side}_under_eye_reference"]
        if float(region.sum()) < 32.0 or float(reference.sum()) < 32.0:
            stats[f"{side}_skipped"] = True
            continue

        l_ref = ops.masked_percentile(lightness, reference, 55.0)
        a_ref = ops.masked_percentile(a_star, reference, 50.0)
        b_ref = ops.masked_percentile(b_star, reference, 50.0)
        if not np.isfinite(l_ref):
            stats[f"{side}_skipped"] = True
            continue

        # `support` is a hard 0/1 confinement. The soft alpha is applied exactly once, by
        # `apply_masked` below. Multiplying by `region` here as well squared it: with a peak
        # alpha of 0.40 the tool delivered 2.6 percent of the deficit while reporting 47.
        support = (region > 1e-4).astype(np.float32)

        # `normalized_lowpass`, not a guided filter. The guided filter was wrong here twice
        # over. It is edge-aware, so at any usable `eps` it retains texture in the deficit
        # estimate, and since `deficit = L_ref - L` that texture enters the lift with inverted
        # sign and subtracts from the real thing. And it averages indiscriminately across the
        # region boundary, pulling in deficits measured on unrelated skin. `blur(x*w)/blur(w)`
        # is genuinely smooth and only averages values from inside the region. Measured
        # high-frequency retention across six portraits: 0.90 to 0.96 with the guided filter,
        # 0.95 to 1.04 with this. It also removed an over-correction: on a subject with mild
        # circles the guided estimate spread the deepest values across the whole region and
        # recovered 0.90 of the deficit, effectively erasing it, against a stated ceiling of
        # 0.60. This lands at 0.47 there.
        deficit = np.maximum(l_ref - lightness, 0.0)
        deficit_low = np.maximum(ops.normalized_lowpass(deficit, support, radius), 0.0)

        # Two limits: a fraction of the low-frequency deficit, and an absolute ceiling in L*
        # units as a rail against a shadow across the face being read as a dark circle.
        #
        # There used to be a third, a pointwise cap against the raw `deficit`. It has been
        # removed because it is a texture-flattening operator, not a safety limit. `deficit`
        # is `l_ref - L` pointwise, so the cap allows a larger lift in a dark texture trough
        # than on a bright texture peak, which compresses exactly the pore-scale contrast this
        # tool exists to preserve. Measured over a fixed region: with the cap 0.917
        # high-frequency retention, without it 0.993. The lift is now a purely low-frequency
        # field, so it cannot alter texture at all, which is a stronger guarantee than the cap
        # was providing.
        ceiling = config.max_luminance_recovery * strength
        wanted = np.minimum(ceiling * deficit_low, config.max_lightness_delta)
        allowed = config.gamut_headroom_fraction * headroom_low
        lift = np.minimum(wanted, allowed)

        new_l = new_l + lift * support

        chroma_pull = config.max_chroma_recovery * strength
        # Low-pass the chroma deviation too: a hard chroma correction would erase the fine
        # bluish veining that makes an eye area look real.
        da = ops.normalized_lowpass(a_ref - a_star, support, radius)
        db = ops.normalized_lowpass(b_ref - b_star, support, radius)
        new_a = new_a + chroma_pull * da * support
        new_b = new_b + chroma_pull * db * support

        total_alpha = np.maximum(total_alpha, region)
        any_side = True

        weight = float(region.sum())
        stats[f"{side}_reference_L"] = float(l_ref)
        stats[f"{side}_mean_deficit_L"] = float((deficit * region).sum() / weight)
        stats[f"{side}_area_px"] = weight
        stats[f"{side}_peak_alpha"] = float(region.max())
        # How much of the correction the gamut ceiling refused. On skin that is already close
        # to blowing a channel this is the reason a correction comes out weaker than the
        # strength implies, and it is not a fault to be tuned away: lifting further would clip.
        # Reported so that a caller, or the future agent, can tell "limited by the subject"
        # apart from "limited by the setting".
        stats[f"{side}_headroom_limited_fraction"] = float(
            ((wanted > allowed + 0.05) * region).sum() / weight
        )
        pending.append((side, region, deficit, weight))

    if not any_side:
        return no_op(rgb, "reduce_dark_circles", params, "no usable infraorbital region")

    edited = lab_to_rgb(np.stack([new_l, new_a, new_b], axis=-1))
    # `edited` is the correction at full strength inside each region's hard support. The soft
    # alpha is applied here and only here, so the composited lift is `lift * alpha`, not
    # `lift * alpha**2`. `apply_masked` still enforces that nothing changed outside it.
    out = apply_masked(rgb, edited, total_alpha, strict=strict, name="reduce_dark_circles")

    # Report what actually reached the pixels, measured on the composited result, rather than
    # the pre-alpha lift. The old stats read the `lift` field before compositing and so
    # overstated the correction by the peak alpha, hiding the squared-alpha bug for as long as
    # it existed. Anything reported here is now measurable in the output image.
    applied = rgb_to_lab(out)[..., 0] - lightness
    for side, region, deficit, weight in pending:
        mean_deficit = float((deficit * region).sum() / weight)
        mean_lift = float((applied * region).sum() / weight)
        stats[f"{side}_mean_lift_L"] = mean_lift
        stats[f"{side}_residual_deficit_L"] = mean_deficit - mean_lift
        stats[f"{side}_recovered_fraction"] = mean_lift / max(mean_deficit, 1e-6)

    stats["edited_area_fraction"] = edited_area_fraction(total_alpha)
    stats["deficit_radius_px"] = radius
    stats["texture_retention"] = _texture_retention(lightness, lightness + applied, total_alpha)

    debug_images: dict[str, np.ndarray] = {}
    if debug:
        debug_images["under_eye_alpha"] = total_alpha
        debug_images["under_eye_lift"] = np.clip((new_l - lightness) / config.max_lightness_delta, 0.0, 1.0)
    return ToolResult(
        image=out, tool="reduce_dark_circles", params=params, stats=stats, changed=True,
        debug=debug_images,
    )


def _texture_retention(before_l: np.ndarray, after_l: np.ndarray, alpha: np.ndarray) -> float:
    """High-frequency energy after over before, inside the region. Should stay near 1.0:
    the whole point of correcting only low frequencies is that texture is untouched."""
    before = ops.high_frequency_energy(before_l, alpha)
    after = ops.high_frequency_energy(after_l, alpha)
    return float(after / before) if before > 1e-6 else 1.0
