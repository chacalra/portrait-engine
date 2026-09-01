"""Nose reshaping.

Started here rather than with the jaw or the brows for a practical reason: the nose is
surrounded by skin on every side, so a warp has nothing awkward to resolve behind it. The jaw
borders hair, neck and background, and the brows are hair rather than skin. Validating the warp
machinery on the easy case first is cheaper than discovering its limits on the hard one.

The operation states its intent as a **measured target in landmark space** and then verifies it,
which is what keeps this module inside the project's calibration discipline even though "a better
nose" has no objective definition. "Reduce the alar width by 12 percent" is not a matter of taste
to implement or to check, whatever anyone thinks of the result.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from portrait_engine.core import ops
from portrait_engine.core.types import FaceAnalysis, ToolResult
from portrait_engine.perception import landmark_sets as ls
from portrait_engine.reshaping.base import register_reshape
from portrait_engine.reshaping.warp import (
    apply_warp,
    displacement_from_handles,
    measure_landmark_shift,
)
from portrait_engine.tools.base import no_op, resolve_image, validate_strength


@dataclass(frozen=True)
class NoseConfig:
    #: Fraction of the *current* alar width removed at strength 1.0. Relative rather than
    #: absolute so the same strength means the same thing on a wide and a narrow nose.
    max_alar_narrowing: float = 0.12
    #: Influence radius of each handle, as a fraction of the interocular distance. This is what
    #: makes the deformation read as anatomy: too small and the alae detach from the cheek, too
    #: large and the whole midface moves with them.
    influence_radius: float = 0.16
    #: How far past the nose the field is allowed to act. The surrounding skin has to deform
    #: with the nose or the result has a visible seam.
    region_margin: float = 0.22
    #: Hard rail on displacement, as a fraction of the interocular distance, enforced by
    #: `apply_warp`. At the default narrowing this leaves roughly 40 percent headroom, so the
    #: rail catches a mistake rather than trimming normal operation.
    max_displacement: float = 0.08


DEFAULT_NOSE_CONFIG = NoseConfig()


@register_reshape
def narrow_nose(
    image: Any,
    analysis: FaceAnalysis,
    strength: float = 0.20,
    *,
    config: NoseConfig | None = None,
    strict: bool = True,
    debug: bool = False,
) -> ToolResult:
    """Narrow the nose by moving the nostril wings inward along the face's own horizontal axis,
    deforming the surrounding skin smoothly so the alae stay attached to the cheeks.

    Args:
        strength: 0 leaves the image untouched. 1.0 removes 12 percent of the current alar
            width, which is a clear change and still within what a retoucher would call a
            correction rather than a different nose.

    Returns:
        A :class:`~portrait_engine.core.types.ToolResult`. ``stats`` reports the alar width
        before and after **as measured on the warped landmarks**, not as requested, plus the
        displacement and the minimum Jacobian of the field.
    """
    strength = validate_strength(strength)
    config = config or DEFAULT_NOSE_CONFIG
    params = {"strength": strength}

    rgb = resolve_image(image)
    if strength == 0.0:
        return no_op(rgb, "narrow_nose", params, "strength is zero")

    geometry = analysis.geometry
    landmarks = analysis.landmarks
    shape = rgb.shape[:2]

    # The face's own horizontal axis, from the eye centres. Using this rather than image columns
    # keeps the operation correct on a rolled head, the same reason the under-eye contour uses
    # the lid normal.
    left = np.asarray(geometry.eye_left_center, dtype=np.float32)
    right = np.asarray(geometry.eye_right_center, dtype=np.float32)
    axis = left - right
    norm = float(np.linalg.norm(axis))
    if norm < 1e-3:
        return no_op(rgb, "narrow_nose", params, "degenerate eye geometry")
    axis = axis / norm

    left_indices, right_indices = _alar_extremes(landmarks, axis)
    width_before = _alar_width(landmarks, left_indices, right_indices, axis)
    if width_before < 1e-3:
        return no_op(rgb, "narrow_nose", params, "could not measure the alar width")

    target_reduction = config.max_alar_narrowing * strength * width_before
    radius = geometry.scaled(config.influence_radius, 6)

    region = ops.blur_mask(
        ops.dilate_mask(analysis.masks["nose"], geometry.scaled(config.region_margin, 8)),
        geometry.scaled(0.06, 3),
    )

    # Build the field, measure what it actually achieved, then rescale. The scatter-and-smooth
    # interpolation under-shoots at the handles by design (see `displacement_from_handles`), and
    # by an amount that depends on how the handles cluster. Since the field is linear in the
    # offsets, one correction pass lands it exactly rather than approximately.
    handles = np.concatenate([landmarks[list(left_indices)], landmarks[list(right_indices)]])
    unit = np.concatenate([
        np.tile(-axis, (len(left_indices), 1)),   # subject's left ala moves toward the midline
        np.tile(axis, (len(right_indices), 1)),
    ]).astype(np.float32)

    achieved = 0.0
    dx = dy = None
    request = target_reduction / 2.0
    for _ in range(2):
        if request <= 0.0:
            break
        # The field is a *source* offset, so to move a wing toward the midline the field there
        # must point away from it. Hence the sign flip.
        offsets = -unit * request
        dx, dy = displacement_from_handles(shape, handles, offsets, radius)
        dx, dy = dx * region, dy * region
        achieved = width_before - _alar_width(
            landmarks, left_indices, right_indices, axis, dx=dx, dy=dy
        )
        if achieved <= 1e-4:
            break
        request *= target_reduction / achieved

    if dx is None or achieved <= 1e-4:
        return no_op(rgb, "narrow_nose", params, "the requested change is below one pixel")

    out, report = apply_warp(
        rgb, dx, dy, region,
        interocular_px=geometry.interocular_px,
        max_displacement=config.max_displacement,
        name="narrow_nose",
        strict=strict,
    )

    width_after = _alar_width(landmarks, left_indices, right_indices, axis, dx=dx, dy=dy)
    stats: dict[str, Any] = {
        "alar_width_before_px": round(float(width_before), 2),
        "alar_width_after_px": round(float(width_after), 2),
        "alar_width_before_iod": round(float(width_before / geometry.interocular_px), 4),
        "narrowing_fraction": round(float((width_before - width_after) / width_before), 4),
        "requested_fraction": round(float(config.max_alar_narrowing * strength), 4),
        "influence_radius_px": radius,
    }
    stats.update(report.as_dict())

    debug_images: dict[str, np.ndarray] = {}
    if debug:
        debug_images["warp_region"] = region
        magnitude = np.sqrt(dx * dx + dy * dy)
        debug_images["displacement"] = magnitude / max(float(magnitude.max()), 1e-6)
    return ToolResult(
        image=out, tool="narrow_nose", params=params, stats=stats, changed=True,
        debug=debug_images,
    )


def _alar_extremes(
    landmarks: np.ndarray, axis: np.ndarray
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """The nose-ring landmarks furthest out on each side.

    Found by projection onto the face's horizontal axis rather than by hardcoding two indices.
    The widest point of a nose is not the same landmark on every face, and on a semi-profile the
    pair that reads as "the width" changes.
    """
    ring = np.asarray(ls.NOSE, dtype=int)
    points = landmarks[ring].astype(np.float32)
    projection = (points - points.mean(axis=0)) @ axis
    order = np.argsort(projection)
    right = tuple(int(ring[i]) for i in order[:3])           # most negative along the axis
    left = tuple(int(ring[i]) for i in order[-3:])
    return left, right


def _alar_width(
    landmarks: np.ndarray,
    left_indices: tuple[int, ...],
    right_indices: tuple[int, ...],
    axis: np.ndarray,
    *,
    dx: np.ndarray | None = None,
    dy: np.ndarray | None = None,
) -> float:
    """Alar width along the face's horizontal axis, optionally after a displacement field.

    One function for both the before and the after on purpose. The first version measured the
    baseline as the full extent of the nose ring and the result as the distance between the two
    handle-group means, which are different estimators. The rescaling loop then never converged,
    because the difference it was correcting contained a constant offset as well as the actual
    narrowing: the measured reduction overshot the request by about 19 percent at every strength
    and the loop had no way to see it. Measuring both ends the same way is not a detail.
    """
    if dx is None or dy is None:
        left = landmarks[list(left_indices)].astype(np.float32)
        right = landmarks[list(right_indices)].astype(np.float32)
    else:
        left = measure_landmark_shift(landmarks, dx, dy, left_indices)
        right = measure_landmark_shift(landmarks, dx, dy, right_indices)
    return float(np.abs((left.mean(axis=0) - right.mean(axis=0)) @ axis))
