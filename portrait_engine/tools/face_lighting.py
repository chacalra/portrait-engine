"""Local face lighting balance.

Method
------
A homomorphic correction in **linear light**, not a tone curve on sRGB values.

1. Convert to linear light and take Rec.709 luminance. A gain applied to sRGB-encoded values
   is not an exposure change: it shifts saturation and hue. In linear light a multiplicative
   gain is exactly what moving a light or changing exposure does.
2. Estimate the illumination as the low-frequency component of ``log(luminance)``, with a
   **smooth** normalized low-pass over the face region at a large radius (42 percent of the
   interocular distance). Smoothness is essential and was learned the hard way: an
   edge-preserving guided filter was used first, and because its estimate snaps to the image's
   own edges, the resulting gain field opposed those edges and cut measured local contrast to
   0.65. A smooth estimate evens out lighting; an edge-aware one flattens the picture.
3. Build a gain field that pulls that low-frequency component partially toward the median of
   the face: ``gain = exp(k * (target - low))``. Two separate coefficients are used so
   shadows can be lifted a little more than highlights are pulled down, which is what a
   retoucher actually does.
4. Clamp the gain hard, to ``[0.72, 1.42]`` at the extreme. The clamp is what prevents the HDR
   look and the halos: an unbounded local gain around a strong shadow edge is exactly how
   halos appear.
5. Tame specular highlights separately, with a soft rolloff on the excess above a knee rather
   than a clip, so a lit forehead or nose stops being a blown patch but keeps its gradient.
6. Apply the gain to all three linear channels equally, so hue and saturation are preserved,
   then re-encode to sRGB through a soft highlight rolloff instead of a hard clip.

The exponent is deliberately small. Fully flattening illumination removes the modelling that
makes a face look three-dimensional, which is the "grey skin" failure in the brief. The mask
is wider than the skin mask (it includes the nose and chin) because a low-frequency gain field
with a narrow mask produces a visible seam.

No relighting model, no generative step: this is a photometric transform of the original
pixels.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from portrait_engine.core import ops
from portrait_engine.core.colorspace import (
    LUMA_709,
    linear_to_srgb,
    soft_highlight_rolloff,
    srgb_to_linear,
)
from portrait_engine.core.config import DEFAULT_CONFIG, FaceLightingConfig
from portrait_engine.core.masking import apply_masked, edited_area_fraction
from portrait_engine.core.types import FaceAnalysis, ToolResult
from portrait_engine.tools.base import no_op, register, resolve_image, validate_strength

_LOG_FLOOR = 1e-3


@register
def balance_face_lighting(
    image: Any,
    analysis: FaceAnalysis,
    strength: float = 0.20,
    *,
    config: FaceLightingConfig | None = None,
    subsample: int | None = None,
    strict: bool = True,
    debug: bool = False,
) -> ToolResult:
    """Even out uneven face lighting with a bounded local gain in linear light: lift shadows,
    tame excessive highlights on forehead, nose and cheeks, and keep the face's natural
    three-dimensional modelling.

    Args:
        strength: 0 leaves the image untouched, 1.0 applies the strongest flattening this
            engine considers natural. The per-pixel gain is clamped to [0.72, 1.42] regardless
            of strength.

    Returns:
        A :class:`~portrait_engine.core.types.ToolResult`. ``stats`` reports the illumination
        spread inside the skin before and after, plus the gain range actually used.
    """
    strength = validate_strength(strength)
    config = config or DEFAULT_CONFIG.face_lighting
    subsample = DEFAULT_CONFIG.subsample if subsample is None else subsample
    params = {"strength": strength}

    rgb = resolve_image(image)
    if strength == 0.0:
        return no_op(rgb, "balance_face_lighting", params, "strength is zero")

    region = analysis.masks["face_photometric"]
    skin = analysis.masks["skin"]
    if float(region.sum()) < 64.0:
        return no_op(rgb, "balance_face_lighting", params, "no usable face region")

    geometry = analysis.geometry
    linear = srgb_to_linear(rgb)
    luma = (linear @ LUMA_709).astype(np.float32)
    log_luma = np.log(np.maximum(luma, _LOG_FLOOR))

    # Reference: the median illumination of the skin, which is the tone the face should settle
    # around. Measured on skin only, so hair and background cannot drag it.
    weights = skin if float(skin.sum()) > 64.0 else region

    radius = geometry.scaled(config.illumination_radius, 5)
    illumination = ops.normalized_lowpass(log_luma, np.maximum(weights, region * 0.35), radius)
    target = ops.masked_percentile(illumination, weights, 50.0)
    if not np.isfinite(target):
        target = float(np.median(illumination))

    deviation = target - illumination
    shadow_k = config.max_flatten_exponent * strength * (1.0 + config.max_shadow_lift)
    highlight_k = config.max_flatten_exponent * strength
    gain_log = np.where(deviation > 0, deviation * shadow_k, deviation * highlight_k)

    # Specular taming, added in the log domain so it goes through the same limiter as the
    # flattening gain. The knee is evaluated on a mid-radius smooth of the sRGB-encoded luma so
    # that skin texture is never mistaken for a highlight.
    mid_radius = geometry.scaled(0.06, 2)
    smooth_srgb = ops.guided_filter(
        linear_to_srgb(luma), linear_to_srgb(luma), mid_radius, 1e-3, subsample=subsample
    )
    # The knee is relative to the face's own bright skin, not an absolute value. An absolute
    # knee of 0.78 was tried first and fired on the whole face of a brightly lit, light-skinned
    # subject, dropping measured local contrast to 0.65. Relative to the skin's 75th percentile
    # it fires only on genuine hotspots.
    skin_bright = ops.masked_percentile(smooth_srgb, weights, 75.0)
    if not np.isfinite(skin_bright):
        skin_bright = config.specular_knee
    excess = smooth_srgb - max(float(skin_bright), config.specular_knee_floor)
    specular = ops.smoothstep(excess, config.specular_excess_low, config.specular_excess_high)
    gain_log = gain_log + np.log1p(-config.max_specular_compression * strength * specular)

    # Soft saturation, then a hard clamp as a safety net. A hard clamp on its own creates a
    # discontinuity in the gain field exactly where the gain is most extreme, which is visible
    # as an edge. `tanh` compresses smoothly toward the same limits instead.
    gain = np.clip(
        np.exp(_soft_limit(gain_log, np.log(config.gain_max), -np.log(1.0 / config.gain_min))),
        config.gain_min,
        config.gain_max,
    ).astype(np.float32)

    corrected = soft_highlight_rolloff(linear * gain[..., None], knee=0.88)
    edited = linear_to_srgb(corrected)
    out = apply_masked(rgb, edited, region, strict=strict, name="balance_face_lighting")

    spread_before = _spread(illumination, weights)
    illumination_after = ops.normalized_lowpass(
        np.log(np.maximum((srgb_to_linear(out) @ LUMA_709).astype(np.float32), _LOG_FLOOR)),
        np.maximum(weights, region * 0.35),
        radius,
    )
    stats: dict[str, Any] = {
        "edited_area_fraction": edited_area_fraction(region),
        "illumination_radius_px": radius,
        "target_log_luma": float(target),
        "illumination_spread_before_stops": spread_before / float(np.log(2.0)),
        "illumination_spread_after_stops": _spread(illumination_after, weights) / float(np.log(2.0)),
        "gain_min_used": float(gain[region > 0.5].min()) if (region > 0.5).any() else 1.0,
        "gain_max_used": float(gain[region > 0.5].max()) if (region > 0.5).any() else 1.0,
        "specular_area_px": float((specular * region).sum()),
        "clipped_pixels": int((out[region > 0.5] >= 1.0).sum()),
        "local_contrast_retention": _local_contrast_retention(rgb, out, skin),
    }

    debug_images: dict[str, np.ndarray] = {}
    if debug:
        debug_images["lighting_alpha"] = region
        debug_images["lighting_gain"] = np.clip((gain - config.gain_min) / (config.gain_max - config.gain_min), 0.0, 1.0)
        debug_images["lighting_specular"] = specular
    return ToolResult(
        image=out, tool="balance_face_lighting", params=params, stats=stats, changed=True,
        debug=debug_images,
    )


def _spread(field: np.ndarray, weights: np.ndarray) -> float:
    """p90 minus p10 of a log-luminance field inside a mask. In log units, so dividing by
    ``ln 2`` gives photographic stops, which is the number a photographer can judge."""
    high = ops.masked_percentile(field, weights, 90.0)
    low = ops.masked_percentile(field, weights, 10.0)
    if not np.isfinite(high) or not np.isfinite(low):
        return 0.0
    return float(high - low)


def _soft_limit(x: np.ndarray, positive_limit: float, negative_limit: float) -> np.ndarray:
    """Smoothly saturate ``x`` toward ``positive_limit`` above zero and ``negative_limit``
    below, using ``tanh``. Both limits are given as positive magnitudes in log-gain units."""
    up = float(max(positive_limit, 1e-6))
    down = float(max(abs(negative_limit), 1e-6))
    limit = np.where(x >= 0, up, down)
    return (limit * np.tanh(x / limit)).astype(np.float32)


def _local_contrast_retention(before: np.ndarray, after: np.ndarray, mask: np.ndarray) -> float:
    """High-frequency energy of ``log`` linear luminance, after over before.

    Measured in the log domain on purpose. A spatially smooth multiplicative gain, which is
    exactly what this tool applies, becomes an *additive* offset in log space, and a Laplacian
    is blind to a smooth offset. So this metric is invariant to any legitimate exposure change
    and only drops if the tool actually destroyed local contrast. A value near 1.0 means texture
    and modelling survived.

    Measuring the same thing on L* instead reads 0.65 to 0.78 for a correct result, because
    CIELAB is a power law of luminance and the texture amplitude in L* legitimately changes with
    exposure. That earlier metric was misleading, not the tool.
    """
    weight = float(mask.sum())
    if weight < 64.0:
        return 1.0
    before_log = np.log(np.maximum((srgb_to_linear(before) @ LUMA_709).astype(np.float32), _LOG_FLOOR))
    after_log = np.log(np.maximum((srgb_to_linear(after) @ LUMA_709).astype(np.float32), _LOG_FLOOR))
    hf_before = ops.high_frequency_energy(before_log, mask)
    hf_after = ops.high_frequency_energy(after_log, mask)
    return float(hf_after / hf_before) if hf_before > 1e-6 else 1.0
