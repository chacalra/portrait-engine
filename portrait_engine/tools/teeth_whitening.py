"""Teeth whitening.

Method
------
Work in CIELAB and change the color of the teeth **multiplicatively**, never by an offset.

* ``b*`` (the yellow-blue axis) carries almost all of the discoloration, so it is scaled down
  by up to 55 percent at strength 1.0. Scaling rather than subtracting means a pixel that is
  already neutral stays neutral and a shadowed pixel keeps its proportion of the color, so the
  shading between teeth survives. An offset would flatten it.
* ``a*`` is scaled down much less. Teeth are yellow, not red, and pulling red out aggressively
  is what makes teeth look grey and dead.
* ``L*`` is lifted by a near-constant **absolute** offset, tapered to zero as a pixel
  approaches a cap of 94, and additionally limited to a fraction of the distance from the
  teeth's own lightness to that cap. A constant offset preserves the L* spread between teeth
  exactly. A headroom-proportional gain, which was the first implementation, does not: it
  compressed measured inter-tooth contrast from 14.7 to 9.5 L* units at strength 1.0, which is
  visible as a flat white slab. Pure white stays unreachable by construction, not by a clamp,
  and already-bright teeth barely move.

The mask comes from the perception layer and is derived by comparing the mouth opening against
the measured lip color, so gums and tongue are excluded. Three confidence tests then gate the
operation: absolute area relative to the face size, horizontal extent relative to the mouth
opening, and a low sanity floor on area share. If any fails, this tool reports itself as a no-op
instead of guessing, which is the normal outcome for a closed mouth.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from portrait_engine.core import ops
from portrait_engine.core.colorspace import lab_to_rgb, rgb_to_lab
from portrait_engine.core.config import DEFAULT_CONFIG, TeethWhiteningConfig
from portrait_engine.core.masking import apply_masked, edited_area_fraction
from portrait_engine.core.types import FaceAnalysis, ToolResult
from portrait_engine.perception.masks import _width_ratio
from portrait_engine.tools.base import no_op, register, resolve_image, validate_strength


@register
def whiten_teeth(
    image: Any,
    analysis: FaceAnalysis,
    strength: float = 0.15,
    *,
    config: TeethWhiteningConfig | None = None,
    strict: bool = True,
    debug: bool = False,
) -> ToolResult:
    """Whiten teeth naturally by reducing the yellow cast and lifting luminance into the
    remaining headroom, keeping inter-tooth shadows and never touching lips or gums.

    Args:
        strength: 0 leaves the image untouched, 1.0 is the strongest result this engine
            considers natural (roughly a 55 percent reduction of the yellow component).

    Returns:
        A :class:`~portrait_engine.core.types.ToolResult`. ``changed`` is False when no
        reliable teeth region was found, which is the normal outcome for a closed mouth.
    """
    strength = validate_strength(strength)
    config = config or DEFAULT_CONFIG.teeth_whitening
    params = {"strength": strength}

    rgb = resolve_image(image)
    if strength == 0.0:
        return no_op(rgb, "whiten_teeth", params, "strength is zero")

    teeth = analysis.masks["teeth"]
    if float(teeth.sum()) < 1.0:
        return no_op(rgb, "whiten_teeth", params, "no reliable teeth region detected")

    mouth = analysis.masks["mouth_interior"]
    mouth_area = float(analysis.masks.area("mouth_interior"))
    coverage = float(teeth.sum()) / max(mouth_area, 1.0)
    min_area = max(24.0, (analysis.geometry.interocular_px * config.min_area_fraction) ** 2)
    width_ratio = _width_ratio(teeth, mouth)

    # Three independent confidence tests. All must pass, and none of them is a share-of-the-mouth
    # threshold on its own: on a wide grin showing tongue and dark interior, teeth are
    # legitimately a small fraction of the opening, and a 10 percent floor made detection flip on
    # and off between 1400 px and 1600 px versions of the same photo.
    if float(teeth.sum()) < min_area:
        return no_op(
            rgb, "whiten_teeth", params,
            f"the teeth region is only {teeth.sum():.0f} px, below the {min_area:.0f} px floor "
            "for this face size",
        )
    if width_ratio < config.min_width_ratio:
        return no_op(
            rgb, "whiten_teeth", params,
            f"the detected region spans only {width_ratio:.0%} of the mouth's width, below the "
            f"{config.min_width_ratio:.0%} floor: it does not look like a row of teeth",
        )
    if coverage < config.min_mouth_coverage:
        return no_op(
            rgb, "whiten_teeth", params,
            f"teeth cover only {coverage:.1%} of the mouth opening, below the "
            f"{config.min_mouth_coverage:.0%} sanity floor",
        )

    lab = rgb_to_lab(rgb)
    lightness, a_star, b_star = lab[..., 0], lab[..., 1], lab[..., 2]

    yellow_factor = 1.0 - config.max_yellow_reduction * strength
    red_factor = 1.0 - config.max_red_reduction * strength
    new_b = b_star * yellow_factor
    new_a = a_star * red_factor

    # Absolute L* offset, limited both by a fixed ceiling and by how bright the teeth already
    # are, then tapered to zero near the cap so nothing can clip to white.
    reference_l = ops.masked_percentile(lightness, teeth, 60.0)
    if not np.isfinite(reference_l):
        reference_l = float(config.l_cap)
    headroom_limit = config.lightness_headroom_fraction * max(config.l_cap - reference_l, 0.0)
    delta = strength * min(config.max_lightness_delta, headroom_limit)
    taper = 1.0 - ops.smoothstep(lightness, config.l_cap - config.l_taper, config.l_cap)
    new_l = lightness + delta * taper

    edited = lab_to_rgb(np.stack([new_l, new_a, new_b], axis=-1))

    # The mask is already feathered, but erode the effective alpha by the strength-independent
    # ramp once more so the vermilion border is never reached even if the polygon was loose.
    alpha = np.clip(teeth, 0.0, 1.0)
    out = apply_masked(rgb, edited, alpha, strict=strict, name="whiten_teeth")

    weight = float(alpha.sum())
    stats: dict[str, Any] = {
        "edited_area_fraction": edited_area_fraction(alpha),
        "teeth_area_px": weight,
        "mouth_coverage": coverage,
        "width_ratio": width_ratio,
        "yellow_factor": yellow_factor,
        "lightness_delta": float(delta),
        "reference_L": float(reference_l),
        "mean_b_before": float((b_star * alpha).sum() / max(weight, 1.0)),
        "mean_b_after": float((new_b * alpha).sum() / max(weight, 1.0)),
        "mean_L_before": float((lightness * alpha).sum() / max(weight, 1.0)),
        "mean_L_after": float((new_l * alpha).sum() / max(weight, 1.0)),
        "max_L_after": float(new_l[alpha > 0.5].max()) if (alpha > 0.5).any() else 0.0,
        "inter_tooth_contrast_before": _contrast(lightness, alpha),
        "inter_tooth_contrast_after": _contrast(new_l, alpha),
    }

    debug_images: dict[str, np.ndarray] = {}
    if debug:
        debug_images["teeth_alpha"] = alpha
    return ToolResult(
        image=out, tool="whiten_teeth", params=params, stats=stats, changed=True,
        debug=debug_images,
    )


def _contrast(channel: np.ndarray, alpha: np.ndarray) -> float:
    """Interquartile spread of ``channel`` inside the mask.

    Reported so a caller can verify that whitening did not flatten the shading between teeth:
    if this collapses, the result is a white slab, not teeth.
    """
    sel = channel[alpha > 0.5]
    if sel.size < 16:
        return 0.0
    return float(np.percentile(sel, 85) - np.percentile(sel, 15))
