"""Skin smoothing by three-band frequency separation.

Method
------
This is not a blur behind a mask. The luminance of the skin is split into three bands with
two guided filters of different radius, and only the **middle** band is attenuated:

* ``base``: illumination and large-scale tone. Untouched. Touching it would flatten the face.
* ``mid``: blotches, uneven tone, the soft shading around a healing spot. **This is what gets
  attenuated**, by up to 85 percent at strength 1.0.
* ``fine``: pores, fine lines, real skin texture. Reinjected at unity gain, always.

Because the fine band survives unchanged, the pores are still there at any strength. That is
the difference between retouching and a beauty filter, and it is a structural property of the
algorithm rather than something that has to be tuned carefully.

Both band cutoffs are edge-aware, which is why guided filtering was chosen over a wavelet
decomposition: reinjecting detail next to an eyelid or an eyebrow does not ring. See
``docs/RESEARCH.md`` section 4.3 for the comparison.

Chroma is treated separately and can be smoothed harder (70 percent at strength 1.0), because
uneven skin tone lives almost entirely in ``a*``/``b*`` while skin texture lives in ``L*``.

Dynamic parameters, in the spirit of FabSoften's attribute-aware smoothing: the base radius
grows with the measured roughness of this particular skin, so a coarse-textured face gets a
wider mid band than a smooth one at the same strength.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from portrait_engine.core import ops
from portrait_engine.core.colorspace import lab_to_rgb, rgb_to_lab
from portrait_engine.core.config import DEFAULT_CONFIG, SkinSmoothingConfig
from portrait_engine.core.masking import apply_masked, edited_area_fraction
from portrait_engine.core.types import FaceAnalysis, ToolResult
from portrait_engine.tools.base import no_op, register, resolve_image, validate_strength


def measure_skin_roughness(lightness: np.ndarray, skin: np.ndarray, radius: int) -> float:
    """Roughness of this skin, as the ratio of mid-band energy to a nominal value.

    Returns a value near 1.0 for average skin, above 1.0 for coarse or blotchy skin. Used to
    scale the base radius, so the tool adapts to the subject instead of using one constant.
    """
    base = ops.guided_filter(lightness, lightness, radius, 9.0, subsample=2)
    energy = float(np.abs(lightness - base)[skin > 0.5].mean()) if (skin > 0.5).any() else 0.0
    nominal = 1.6  # L* units, measured as typical mid-band energy on normal portrait skin
    return float(np.clip(energy / nominal, 0.4, 2.5))


@register
def skin_smoothing(
    image: Any,
    analysis: FaceAnalysis,
    strength: float = 0.25,
    *,
    config: SkinSmoothingConfig | None = None,
    subsample: int | None = None,
    strict: bool = True,
    debug: bool = False,
) -> ToolResult:
    """Even out skin irregularities while keeping pores and fine texture, by attenuating only
    the mid-frequency band of luminance and the low-frequency unevenness of chroma.

    Args:
        strength: 0 leaves the image untouched, 1.0 attenuates the mid band by 85 percent,
            which is the strongest setting that still reads as photographic retouching.

    Returns:
        A :class:`~portrait_engine.core.types.ToolResult` whose ``stats`` include
        ``texture_retention``, the ratio of high-frequency energy after to before inside the
        skin mask. Values below roughly 0.55 indicate a plastic result.
    """
    strength = validate_strength(strength)
    config = config or DEFAULT_CONFIG.skin_smoothing
    subsample = DEFAULT_CONFIG.subsample if subsample is None else subsample
    params = {"strength": strength}

    rgb = resolve_image(image)
    if strength == 0.0:
        return no_op(rgb, "skin_smoothing", params, "strength is zero")

    skin = analysis.masks["skin"]
    if float(skin.sum()) < 64.0:
        return no_op(rgb, "skin_smoothing", params, "skin mask is empty")

    geometry = analysis.geometry
    lab = rgb_to_lab(rgb)
    lightness, a_star, b_star = lab[..., 0], lab[..., 1], lab[..., 2]

    radius_fine = geometry.scaled(config.radius_fine, 1)
    nominal_base = geometry.scaled(config.radius_base, 3)
    roughness = measure_skin_roughness(lightness, skin, nominal_base)
    radius_base = max(
        radius_fine + 1,
        int(round(nominal_base * (1.0 + config.dynamic_radius_gain * (roughness - 1.0)))),
    )
    radius_chroma = geometry.scaled(config.radius_chroma, 3)

    bands = ops.decompose_bands(
        lightness,
        lightness,
        radius_fine=radius_fine,
        radius_base=radius_base,
        eps_fine=config.eps_fine,
        eps_base=config.eps_base,
        subsample=subsample,
    )
    mid_gain = 1.0 - config.max_mid_attenuation * strength
    new_l = bands.recombine(mid_gain=mid_gain, fine_gain=config.fine_gain)

    chroma_gain = 1.0 - config.max_chroma_attenuation * strength
    new_a = _smooth_channel(a_star, lightness, radius_chroma, config.eps_chroma, chroma_gain, subsample)
    new_b = _smooth_channel(b_star, lightness, radius_chroma, config.eps_chroma, chroma_gain, subsample)

    edited = lab_to_rgb(np.stack([new_l, new_a, new_b], axis=-1))
    out = apply_masked(rgb, edited, skin, strict=strict, name="skin_smoothing")

    before_hf = ops.high_frequency_energy(lightness, skin)
    after_hf = ops.high_frequency_energy(rgb_to_lab(out)[..., 0], skin)
    stats: dict[str, Any] = {
        "edited_area_fraction": edited_area_fraction(skin),
        "skin_area_px": float(skin.sum()),
        "roughness": roughness,
        "radius_fine_px": radius_fine,
        "radius_base_px": radius_base,
        "mid_gain": mid_gain,
        "chroma_gain": chroma_gain,
        "mid_energy_before": float(np.abs(bands.mid)[skin > 0.5].mean()) if (skin > 0.5).any() else 0.0,
        "texture_retention": float(after_hf / before_hf) if before_hf > 1e-6 else 1.0,
    }

    debug_images: dict[str, np.ndarray] = {}
    if debug:
        debug_images["skin_alpha"] = skin
        debug_images["band_mid"] = np.clip(np.abs(bands.mid) / 8.0, 0.0, 1.0)
        debug_images["band_fine"] = np.clip(np.abs(bands.fine) / 4.0, 0.0, 1.0)
    return ToolResult(
        image=out, tool="skin_smoothing", params=params, stats=stats, changed=True,
        debug=debug_images,
    )


def _smooth_channel(
    channel: np.ndarray, guide: np.ndarray, radius: int, eps: float, gain: float, subsample: int
) -> np.ndarray:
    """Attenuate a chroma channel's deviation from its edge-aware local mean.

    The guide is luminance, so chroma smoothing still respects the edges of features: the
    color of a lip or an eyebrow does not bleed into the surrounding skin.
    """
    low = ops.guided_filter(channel, guide, radius, eps, subsample=subsample)
    return (low + (channel - low) * gain).astype(np.float32)
