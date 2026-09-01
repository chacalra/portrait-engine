"""Blemish cleanup: reduce small temporary imperfections without erasing skin.

Two separate problems, solved separately.

Detection
---------
A multi-scale blob detector on the ``L*`` and ``a*`` channels, restricted to the skin mask. At
each scale the response is ``annulus_mean - centre_mean``, where the annulus strictly excludes
the centre block. Excluding it matters: a plain difference of two concentric box filters lets
the blemish contaminate its own reference, and the response collapses at the larger scales (it
measured a maximum of 1.18 against 2.29 for the annulus form on the same photo).

The response is normalized by the **local texture** standard deviation, measured as the spread
of the high-pass residual at the same scale. The threshold is therefore in units of "how
unusual is this spot for this skin", not in absolute L*, which is what lets one threshold work
on smooth young skin and on coarse weathered skin. Normalizing by the plain local standard
deviation instead, which was the first attempt, measures shading rather than texture: it
returned a median sigma of 8.8 L* where the texture sigma is 3.2, and the detector never fired.

Two channels because blemishes come in two flavours: a spot that is *darker* than its
surroundings, and a spot that is *redder* without being darker (an inflamed pimple).

What is deliberately kept:

* **moles and strong freckles.** A blob more than 17 L* units darker than its surroundings is
  treated as a permanent feature and skipped. Temporary blemishes are low contrast; moles are
  not.
* **anything large.** A blob wider than 7.5 percent of the interocular distance is anatomy, a
  shadow or a highlight, not a blemish.
* **anything elongated.** Wrinkles, hair strands and creases are rejected by an eccentricity
  test, because a blemish is round.
* **beard, brows, lashes, lips, nostrils, eyes.** Excluded by the mask, not by the detector.
* **a hard total-area cap.** The tool refuses to edit more than 6 percent of the skin mask. If
  more candidates than that qualify, only the strongest are taken. This is the guard against
  indiscriminate erasure: a detector failure becomes a weak result, never a destroyed face.

Healing
-------
Frequency-separated healing, which is what a retoucher does with a healing brush over a
frequency-separated file, rather than ``inpaint`` over the pixels.

The skin is split into a **surface** and a **fine detail** layer, with the cutoff at the pore
scale. Only the surface is repaired, by interpolating it from the ring of skin around the
blemish with a normalized low-pass. The original detail layer is then reinjected unchanged.
The blemish disappears because its energy lives in the surface; the pores survive because the
detail layer is never touched. There is no cloning, so there is no repeated-texture artifact.

The cutoff has to be at the pore scale and not at the blemish scale. With the split at 3
percent of the interocular distance, a blemish landed partly in the "detail" layer, which is
preserved on purpose, and only about 40 percent of a synthetic spot was removed even at full
strength.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from portrait_engine.core import ops
from portrait_engine.core.colorspace import lab_to_rgb, rgb_to_lab
from portrait_engine.core.config import DEFAULT_CONFIG, SkinCleanupConfig
from portrait_engine.core.masking import apply_masked, edited_area_fraction
from portrait_engine.core.types import FaceAnalysis, ToolResult
from portrait_engine.tools.base import no_op, register, resolve_image, validate_strength

#: The ring used to estimate "what this area should look like" is this multiple of the blob
#: radius. Large enough to sit outside the blemish, small enough to stay local.
RING_FACTOR = 3.0


def annulus_mean(channel: np.ndarray, radius: int, ring: int) -> np.ndarray:
    """Mean of ``channel`` over the square annulus between ``radius`` and ``ring``.

    The centre block is algebraically removed, so the value is a reference for "what the skin
    around here looks like" that is not contaminated by the blemish itself.
    """
    inner = ops.box_filter(channel, radius)
    outer = ops.box_filter(channel, ring)
    inner_area = float((2 * radius + 1) ** 2)
    outer_area = float((2 * ring + 1) ** 2)
    return ((outer * outer_area - inner * inner_area) / (outer_area - inner_area)).astype(np.float32)


def detect_blemishes(
    lab: np.ndarray,
    skin: np.ndarray,
    interocular_px: float,
    strength: float,
    config: SkinCleanupConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Multi-scale blob detection restricted to the skin. Returns ``(binary mask, stats)``."""
    lightness, a_star = lab[..., 0], lab[..., 1]
    valid = skin > 0.35
    if not valid.any():
        return np.zeros_like(lightness), {"threshold": 0.0, "raw_candidate_px": 0}

    threshold = config.threshold_high + (config.threshold_low - config.threshold_high) * strength
    texture_window = max(6, int(round(interocular_px * config.texture_window)))

    best_score = np.zeros_like(lightness)
    depth = np.zeros_like(lightness)
    sigma_report: list[float] = []

    for scale in config.scales:
        radius = max(1, int(round(interocular_px * scale * 0.5)))
        ring = max(radius + 2, int(round(radius * RING_FACTOR)))

        ring_l = annulus_mean(lightness, radius, ring)
        ring_a = annulus_mean(a_star, radius, ring)
        dark = ring_l - ops.box_filter(lightness, radius)
        red = ops.box_filter(a_star, radius) - ring_a

        # Texture scale at this blob size: spread of the high-pass residual over a wide window.
        residual = lightness - ring_l
        sigma = np.sqrt(
            np.maximum(ops.box_filter(residual * residual, texture_window), 0.0)
        ) + config.sigma_floor
        sigma_report.append(float(np.median(sigma[valid])))

        score = np.maximum(dark / sigma, red / (sigma * config.redness_sigma_scale + config.sigma_floor))
        best_score = np.maximum(best_score, score)

        # Depth for the mole test is measured **pointwise** and as the **maximum over all
        # scales**, independently of which scale won the score. Two mistakes were made here and
        # both let moles through:
        #
        #  * using the box mean instead of the pointwise value: a synthetic mole 25 L* deep
        #    measured 6.9 through the box filter and 22.6 pointwise;
        #  * recording the depth at the scale that maximized the *score*: at the finest scale
        #    the annulus falls inside a 3 px mole, so it reported 12.8 instead of 24 and the
        #    mole was healed.
        depth = np.maximum(depth, ring_l - lightness)

    candidate = (best_score > threshold) & valid

    # Keep permanent features. A blob far darker than its surroundings is a mole or a strong
    # freckle, not a temporary blemish.
    #
    # This has to be a *regional* test, not a per-pixel one. The first version simply masked out
    # pixels darker than the threshold, and moles survived only in their centres: the soft rim
    # of every mole still passed the score test, formed its own connected component, and got
    # healed. Measured on synthetic moles, 95 percent of mole area was still being edited.
    #
    # The fix is the per-component rejection in `_filter_components`, where a mole's rim is
    # connected to its core and is rejected with it. The exclusion zone below is only a small
    # safety margin: a wide one was measured swallowing every real blemish that happened to sit
    # near a mole.
    core = (depth >= config.mole_darkness_l) & valid
    protect_radius = max(1, int(round(interocular_px * config.mole_protect_margin)))
    protected_cores = ops.dilate_mask(core.astype(np.float32), protect_radius) > 0.05
    candidate &= ~protected_cores

    stats: dict[str, Any] = {
        "threshold": float(threshold),
        "raw_candidate_px": int(candidate.sum()),
        "sigma_median_per_scale": [round(v, 2) for v in sigma_report],
        "score_p999": float(np.percentile(best_score[valid], 99.9)),
        "score_max": float(best_score[valid].max()),
    }
    if not candidate.any():
        return np.zeros_like(lightness), stats

    stats["permanent_core_px"] = int(core.sum())
    mask, shape_stats = _filter_components(
        candidate, best_score, depth, skin, interocular_px, config
    )
    stats.update(shape_stats)
    return mask, stats


def _filter_components(
    candidate: np.ndarray,
    score: np.ndarray,
    depth: np.ndarray,
    skin: np.ndarray,
    interocular_px: float,
    config: SkinCleanupConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Reject components that are permanent, too large, too elongated, or over the area budget."""
    count, labels, stats_cv, _ = cv2.connectedComponentsWithStats(
        candidate.astype(np.uint8), connectivity=8
    )
    max_radius = interocular_px * config.max_blob_fraction * 0.5
    max_area = np.pi * max_radius**2
    min_area = max(3, int(round((interocular_px * config.scales[0] * 0.5) ** 2)))
    budget = config.max_area_fraction * float(skin.sum())

    kept: list[tuple[float, int, int]] = []  # (score, area, label)
    rejected = {"permanent": 0, "too_large": 0, "too_elongated": 0, "too_small": 0}
    for label in range(1, count):
        _, _, w, h, area = stats_cv[label]
        selection = labels == label
        if area < min_area:
            rejected["too_small"] += 1
            continue
        if float(np.percentile(depth[selection], 95)) >= config.mole_darkness_l:
            rejected["permanent"] += 1
            continue
        if area > max_area or max(w, h) > 4.0 * max_radius:
            rejected["too_large"] += 1
            continue
        if max(w, h) / max(1.0, min(w, h)) > 3.2:
            rejected["too_elongated"] += 1
            continue
        kept.append((float(score[selection].mean()), int(area), label))

    kept.sort(reverse=True)
    mask = np.zeros(candidate.shape, dtype=np.float32)
    used = 0.0
    accepted = 0
    for _score, area, label in kept:
        if used + area > budget:
            continue
        mask[labels == label] = 1.0
        used += area
        accepted += 1

    return mask, {
        "components_found": int(count - 1),
        "components_accepted": accepted,
        "components_rejected": rejected,
        "area_budget_px": budget,
        "area_used_px": used,
        "area_budget_hit": bool(kept and used + min(k[1] for k in kept) > budget),
    }


@register
def skin_cleanup(
    image: Any,
    analysis: FaceAnalysis,
    strength: float = 0.20,
    *,
    config: SkinCleanupConfig | None = None,
    subsample: int | None = None,
    strict: bool = True,
    debug: bool = False,
) -> ToolResult:
    """Reduce small temporary skin imperfections (spots, pimples, isolated marks) by repairing
    only the smooth surface layer inside each detected blemish and reinjecting the original
    skin texture, keeping moles, freckles, beard and all facial features.

    Args:
        strength: 0 leaves the image untouched. Higher values lower the detection threshold and
            increase the correction applied to each blemish. 1.0 removes 92 percent of the
            surface deviation of every blob above a 1.9-sigma local contrast threshold, subject
            to a hard cap of 6 percent of the skin area.

    Returns:
        A :class:`~portrait_engine.core.types.ToolResult`. ``stats`` reports how many blobs were
        found, accepted and rejected and why, plus the area budget actually used.
    """
    strength = validate_strength(strength)
    config = config or DEFAULT_CONFIG.skin_cleanup
    subsample = DEFAULT_CONFIG.subsample if subsample is None else subsample
    params = {"strength": strength}

    rgb = resolve_image(image)
    if strength == 0.0:
        return no_op(rgb, "skin_cleanup", params, "strength is zero")

    skin = analysis.masks["skin"]
    if float(skin.sum()) < 256.0:
        return no_op(rgb, "skin_cleanup", params, "skin mask is too small")

    geometry = analysis.geometry
    lab = rgb_to_lab(rgb)
    blobs, detect_stats = detect_blemishes(
        lab, skin, geometry.interocular_px, strength, config
    )
    if float(blobs.sum()) < 3.0:
        return no_op(
            rgb, "skin_cleanup", params,
            f"no blemish passed the filters (threshold {detect_stats['threshold']:.2f} sigma)",
        )

    # Two different dilations, for two different jobs.
    #
    #  * `excluded` is what the repair must NOT read from. It has to be generously larger than
    #    the detected core, because the detector fires on a blemish's centre while the blemish
    #    extends further. With too small a margin the donor region still contains the blemish
    #    and the repair reads it straight back in.
    #  * `alpha` is where the repair is applied, and stays tighter so nothing else moves.
    excluded = ops.dilate_mask(blobs, geometry.scaled(config.donor_dilate, 2))
    alpha_core = ops.dilate_mask(blobs, geometry.scaled(config.alpha_dilate, 2))
    feather = max(1, geometry.scaled(config.alpha_dilate * 0.5, 1))
    alpha = np.clip(ops.blur_mask(alpha_core, feather) * skin, 0.0, 1.0)

    heal_radius = geometry.scaled(config.heal_radius, 2)
    correction = config.max_correction * strength

    # Skin outside the blemishes, which is where the repair reads from.
    donor = np.clip(skin * (1.0 - excluded), 0.0, 1.0)
    fill_radius = max(geometry.scaled(config.donor_dilate * 2.5, 4), heal_radius * 2)

    channels = []
    for index in range(3):
        channel = lab[..., index]
        # `surface` = everything above the pore scale, so the whole blemish is inside it.
        # `channel - surface` is the fine texture, which is never modified.
        surface = ops.guided_filter(
            channel, lab[..., 0], heal_radius, config.heal_eps, subsample=1
        )
        repaired_surface = ops.normalized_lowpass(surface, donor, fill_radius)
        channels.append(channel + correction * (repaired_surface - surface))

    edited = lab_to_rgb(np.stack(channels, axis=-1))
    out = apply_masked(rgb, edited, alpha, strict=strict, name="skin_cleanup")

    # Measured over the whole skin mask, not over the blemish alpha. Inside a blemish the
    # high-frequency energy is *supposed* to change, since the blemish edge is what was removed;
    # measuring there would report a false failure. Over the skin it answers the real question:
    # did the tool damage the texture of the face?
    before_hf = ops.high_frequency_energy(lab[..., 0], skin)
    after_hf = ops.high_frequency_energy(rgb_to_lab(out)[..., 0], skin)
    stats: dict[str, Any] = dict(detect_stats)
    stats.update(
        {
            "edited_area_fraction": edited_area_fraction(alpha),
            "blemish_area_px": float(blobs.sum()),
        "excluded_area_px": float(excluded.sum()),
            "skin_area_px": float(skin.sum()),
            "blemish_fraction_of_skin": float(blobs.sum() / max(skin.sum(), 1.0)),
            "correction": correction,
            "heal_radius_px": heal_radius,
        "fill_radius_px": fill_radius,
            "texture_retention": float(after_hf / before_hf) if before_hf > 1e-6 else 1.0,
        }
    )

    debug_images: dict[str, np.ndarray] = {}
    if debug:
        debug_images["blemish_mask"] = blobs
        debug_images["blemish_alpha"] = alpha
    return ToolResult(
        image=out, tool="skin_cleanup", params=params, stats=stats, changed=True,
        debug=debug_images,
    )
