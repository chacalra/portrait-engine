"""Even out regional skin colour: ruddy cheeks, a sallow temple, a flushed chin.

Why this is a separate tool and not a setting on ``skin_smoothing``
------------------------------------------------------------------
``skin_smoothing`` removes **local** unevenness. It pulls every pixel toward its own local
mean, which is exactly right for a blotch. It can do nothing at all about **regional**
unevenness, because when a whole cheek is uniformly red the local mean is red too, so there is
nothing to pull toward. Widening its chroma radius does not fix this; it was measured, and the
defect is structural rather than a tuning problem.

Measured on a portrait with ruddy cheeks (median ``a*`` by region: forehead 14.2, temple 13.4,
nose 18.4, cheeks 19.9 and 19.3, a spread of 6.5):

===================================================  ============
approach                                             region range
===================================================  ============
original                                                      6.5
``skin_smoothing``, chroma radius 0.14 (default)              5.8
``skin_smoothing``, chroma radius 0.45, attenuation 0.90       4.5
``equalize_skin_tone`` at 0.9                                 3.2
===================================================  ============

Method
------
Estimate the very-low-frequency ``a*`` and ``b*`` field over the skin with
``normalized_lowpass``, and shift each region toward one face-wide target taken as the skin's
own median. This is what a retoucher does with a hue/saturation adjustment or a colour
dodge-and-burn pass.

Two properties make it safe:

* **``L*`` is never touched.** Modelling, texture and the three-dimensional form of the face
  are unaffected by construction, not by a limit that could be raised.
* **The correction is a low-frequency field**, so local detail cannot be altered. Measured pore
  energy went 4.91 to 4.91, unchanged to three significant figures at every strength tested.

The radius must be large. It has to average over a whole region, not over a blotch; at a small
radius this degenerates into a weaker copy of ``skin_smoothing``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from portrait_engine.core import ops
from portrait_engine.core.colorspace import lab_to_rgb, rgb_to_lab
from portrait_engine.core.config import DEFAULT_CONFIG, SkinToneConfig
from portrait_engine.core.masking import apply_masked, edited_area_fraction
from portrait_engine.core.types import FaceAnalysis, ToolResult
from portrait_engine.tools.base import no_op, register, resolve_image, validate_strength


@register
def equalize_skin_tone(
    image: Any,
    analysis: FaceAnalysis,
    strength: float = 0.20,
    *,
    config: SkinToneConfig | None = None,
    strict: bool = True,
    debug: bool = False,
) -> ToolResult:
    """Even out regional skin colour differences, such as ruddy cheeks or a sallow temple, by
    shifting the low-frequency colour field toward the face's own median without touching
    luminance, texture or the shape of the face's shading.

    Args:
        strength: 0 leaves the image untouched. 1.0 would move each region's low-frequency
            chroma all the way onto the face median, which is too far: some regional variation
            (a warmer cheek, a cooler temple) is what makes skin look like skin. The useful
            range is 0.5 to 0.7.

    Returns:
        A :class:`~portrait_engine.core.types.ToolResult`. ``stats`` reports the spread of
        median ``a*`` across face regions before and after, so the correction can be checked
        against the defect it is meant to fix rather than against its own parameters.
    """
    strength = validate_strength(strength)
    config = config or DEFAULT_CONFIG.skin_tone
    params = {"strength": strength}

    rgb = resolve_image(image)
    if strength == 0.0:
        return no_op(rgb, "equalize_skin_tone", params, "strength is zero")

    mask = analysis.masks["skin"]
    if float(mask.sum()) < 256.0:
        return no_op(rgb, "equalize_skin_tone", params, "no usable skin region")

    lab = rgb_to_lab(rgb)
    radius = analysis.geometry.scaled(config.field_radius, 8)

    stats: dict[str, Any] = {"field_radius_px": radius}

    # L* is copied through untouched. That is the guarantee, and it is structural.
    channels = [lab[..., 0]]
    spread_before = spread_after = 0.0
    for index, name in ((1, "a"), (2, "b")):
        channel = lab[..., index]
        target = ops.masked_percentile(channel, mask, 50.0)
        if not np.isfinite(target):
            channels.append(channel)
            continue
        # `normalized_lowpass`, so the field only averages skin and does not pull the
        # background or the lips into the estimate.
        field = ops.normalized_lowpass(channel, mask, radius)
        shift = np.clip(strength * (target - field), -config.max_shift, config.max_shift)
        channels.append(channel + shift)
        stats[f"target_{name}"] = float(target)
        stats[f"max_shift_{name}"] = float(np.abs(shift * mask).max())
        if index == 1:
            spread_before = _spread(field, mask)
            spread_after = _spread(field + shift, mask)

    edited = lab_to_rgb(np.stack(channels, axis=-1))
    out = apply_masked(rgb, edited, mask, strict=strict, name="equalize_skin_tone")

    # The defect, as one number, before and after: how much the regional colour field varies
    # across the skin. Reported so the tool is judged on the thing it exists to fix rather than
    # on its own parameters.
    stats["region_spread_a_before"] = spread_before
    stats["region_spread_a_after"] = spread_after
    stats["edited_area_fraction"] = edited_area_fraction(mask)

    debug_images: dict[str, np.ndarray] = {}
    if debug:
        debug_images["skin_alpha"] = mask
        debug_images["chroma_field"] = np.clip((lab[..., 1] - 5.0) / 25.0, 0.0, 1.0)
    return ToolResult(
        image=out, tool="equalize_skin_tone", params=params, stats=stats, changed=True,
        debug=debug_images,
    )


def _spread(field: np.ndarray, mask: np.ndarray) -> float:
    """Range of the low-frequency chroma field across the skin, 5th to 95th percentile.

    Measured on the field rather than by probing discs at landmark positions. That was the first
    implementation and it is a worse metric as well as a forbidden one: a tool may not read
    `analysis.landmarks` (see `test_tools_read_only_masks_and_geometry_from_the_analysis`,
    which caught it), and the field is a more complete answer anyway because it covers the whole
    skin instead of five sample points.
    """
    low = ops.masked_percentile(field, mask, 5.0)
    high = ops.masked_percentile(field, mask, 95.0)
    if not (np.isfinite(low) and np.isfinite(high)):
        return 0.0
    return round(float(high - low), 3)
