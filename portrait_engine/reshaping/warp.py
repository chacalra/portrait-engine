"""Bounded geometric warping: the one place in the engine where pixels are allowed to move.

This module exists deliberately outside ``portrait_engine.tools``. Everything in ``tools`` is
covered by AST tests that fail if a module so much as names ``cv2.remap``, and those tests still
hold. Reshaping is a **separate registry**, so a caller (and later the agent) has to ask for it
by name, and any run that used only the identity-preserving tools can still be proved to have
left the geometry alone. See ``docs/RESHAPING.md``.

How the warp is built
---------------------
Not a general free-form liquify. A displacement field is built from a small set of **handles**:
landmark positions paired with a target offset. Each handle contributes a Gaussian bump, and the
field is their sum, so the deformation decays smoothly to nothing away from the handles.

The mask primitives in ``core.ops`` are the wrong tool here and were tried first.
``normalized_lowpass`` divides by the smoothed weights, so a handful of impulses produce the
*average of the offsets* across the whole influence area: a plateau with no falloff at all, which
warps the surrounding face as hard as the nose. And ``blur_mask`` clips its output to ``[0, 1]``,
which silently truncated every displacement to one pixel. Both are correct for what they were
built for, masks and illumination, and both are wrong for a vector field.

A smooth displacement field is what makes a warp look like anatomy rather than like a smear. It
also makes the operation *measurable*, which is what let this module keep the project's
calibration discipline: an operation states its intent in landmark space ("reduce the alar width
by 4 percent of the interocular distance") and the result is verified in the same space.

Three guardrails, all enforced in :func:`apply_warp`
----------------------------------------------------
1. **Bounded displacement.** No pixel moves further than ``max_displacement`` times the
   interocular distance. This is the rail that keeps a strength of 1.0 an edit rather than a
   deformation.
2. **Containment.** The field is multiplied by a region mask, so an operation on the nose cannot
   move an eye. Verified after the fact, not just intended.
3. **No folding.** A displacement field whose Jacobian determinant goes non-positive has folded
   the image onto itself, which produces the melted look of a badly used liquify tool. This is
   checked and refused rather than shipped.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from portrait_engine.core.types import PortraitEngineError


class WarpContractError(PortraitEngineError):
    """A warp violated one of the three guardrails above."""


@dataclass(frozen=True)
class WarpReport:
    """What a warp actually did, in units that can be checked against what was asked."""

    max_displacement_px: float
    max_displacement_iod: float
    mean_displacement_px: float
    min_jacobian: float
    moved_area_fraction: float

    def as_dict(self) -> dict[str, float]:
        return {
            "max_displacement_px": round(self.max_displacement_px, 3),
            "max_displacement_iod": round(self.max_displacement_iod, 5),
            "mean_displacement_px": round(self.mean_displacement_px, 4),
            "min_jacobian": round(self.min_jacobian, 4),
            "moved_area_fraction": round(self.moved_area_fraction, 6),
        }


def displacement_from_handles(
    shape: tuple[int, int],
    handles: np.ndarray,
    offsets: np.ndarray,
    radius: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Scatter sparse handle offsets into a dense, smooth displacement field.

    Args:
        shape: ``(height, width)``.
        handles: ``(N, 2)`` float pixel coordinates, ``(x, y)``.
        offsets: ``(N, 2)`` float pixel offsets to apply at those handles.
        radius: influence radius in pixels. Each handle contributes a Gaussian bump of roughly
            this extent, so larger means a smoother, wider deformation.

    Returns:
        ``(dx, dy)``, each ``(H, W)`` float32. These are **source** offsets: the value at a
        destination pixel says where to read from, which is the convention ``cv2.remap`` wants.

    The field is a sum of Gaussians rather than an exact interpolant such as a thin-plate
    spline. A TPS reproduces the handle offsets exactly and oscillates between them; this
    under-shoots wherever handles are sparse and stays monotone, which is the safer failure mode
    on a face. Callers compensate by stating their intent as a measured target and rescaling
    until the measured landmark movement matches it, which converges in one pass because the
    field is linear in ``offsets``.
    """
    height, width = shape
    scatter = np.zeros((height, width, 2), dtype=np.float32)

    placed = 0
    for (hx, hy), (ox, oy) in zip(handles, offsets, strict=True):
        col, row = int(round(float(hx))), int(round(float(hy)))
        if not (0 <= row < height and 0 <= col < width):
            continue
        scatter[row, col, 0] += float(ox)
        scatter[row, col, 1] += float(oy)
        placed += 1

    if placed == 0:
        zero = np.zeros((height, width), dtype=np.float32)
        return zero, zero.copy()

    sigma = max(1.0, radius / 2.0)
    blurred = cv2.GaussianBlur(scatter, (0, 0), sigma, borderType=cv2.BORDER_REPLICATE)

    # A Gaussian kernel sums to one, so an impulse of amplitude A comes back with a peak of
    # A / (2 pi sigma^2). Measure that peak on a single impulse instead of using the closed
    # form, so kernel truncation and discretisation cannot bias the amplitude.
    probe_size = int(max(9, 6 * sigma + 1)) | 1
    probe = np.zeros((probe_size, probe_size), dtype=np.float32)
    probe[probe_size // 2, probe_size // 2] = 1.0
    peak = float(cv2.GaussianBlur(probe, (0, 0), sigma, borderType=cv2.BORDER_REPLICATE).max())

    field = blurred / max(peak, 1e-9)
    return (
        np.ascontiguousarray(field[..., 0], dtype=np.float32),
        np.ascontiguousarray(field[..., 1], dtype=np.float32),
    )


def jacobian_minimum(dx: np.ndarray, dy: np.ndarray, step: int = 4) -> float:
    """Smallest Jacobian determinant of the mapping ``p -> p + d(p)``.

    Below zero the warp has folded the image onto itself. Sampled on a stride because the field
    is smooth by construction and a full-resolution evaluation costs more than it tells you.
    """
    sx, sy = dx[::step, ::step], dy[::step, ::step]
    dxdy, dxdx = np.gradient(sx)
    dydy, dydx = np.gradient(sy)
    scale = float(step)
    determinant = (1.0 + dxdx / scale) * (1.0 + dydy / scale) - (dxdy / scale) * (dydx / scale)
    return float(determinant.min())


def apply_warp(
    rgb: np.ndarray,
    dx: np.ndarray,
    dy: np.ndarray,
    region: np.ndarray,
    *,
    interocular_px: float,
    max_displacement: float,
    name: str = "warp",
    strict: bool = True,
) -> tuple[np.ndarray, WarpReport]:
    """Resample ``rgb`` through a bounded, contained, non-folding displacement field.

    This is to reshaping what ``core.masking.apply_masked`` is to the photometric tools: the one
    function that writes the result, so the guardrails live in a single place and cannot be
    bypassed by a new operation forgetting to apply them.

    Raises:
        WarpContractError: if the field folds, or if ``strict`` and it exceeds the displacement
            bound or moves anything outside ``region``.
    """
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise WarpContractError(f"{name}: expected (H, W, 3), got {rgb.shape}")
    height, width = rgb.shape[:2]
    if dx.shape != (height, width) or dy.shape != (height, width):
        raise WarpContractError(f"{name}: displacement field does not match the image")

    # Containment first, so the bound is measured on the field that will actually be applied.
    contained = np.clip(region, 0.0, 1.0).astype(np.float32)
    dx = dx * contained
    dy = dy * contained

    magnitude = np.sqrt(dx * dx + dy * dy)
    max_px = float(magnitude.max())
    limit_px = max_displacement * interocular_px
    if strict and max_px > limit_px + 1e-3:
        raise WarpContractError(
            f"{name}: maximum displacement {max_px:.2f} px exceeds the bound "
            f"{limit_px:.2f} px ({max_displacement} of the interocular distance)"
        )

    min_jacobian = jacobian_minimum(dx, dy)
    if min_jacobian <= 0.0:
        raise WarpContractError(
            f"{name}: the displacement field folds (minimum Jacobian {min_jacobian:.3f}). "
            "A folded warp melts the image onto itself; reduce the strength or widen the radius."
        )

    grid_x, grid_y = np.meshgrid(
        np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32)
    )
    out = cv2.remap(
        rgb, grid_x + dx, grid_y + dy,
        interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE,
    )
    out = np.clip(out, 0.0, 1.0).astype(np.float32)

    # Outside the region the field is exactly zero, so the resample is the identity there. Copy
    # the original rather than trusting the interpolator to be bit-exact about it.
    untouched = magnitude <= 0.0
    out[untouched] = rgb[untouched]

    moved = float((magnitude > 0.05).mean())
    report = WarpReport(
        max_displacement_px=max_px,
        max_displacement_iod=max_px / max(interocular_px, 1e-6),
        mean_displacement_px=float(magnitude[magnitude > 0].mean()) if moved > 0 else 0.0,
        min_jacobian=min_jacobian,
        moved_area_fraction=moved,
    )
    return out, report


def measure_landmark_shift(
    landmarks: np.ndarray,
    dx: np.ndarray,
    dy: np.ndarray,
    indices: tuple[int, ...],
) -> np.ndarray:
    """Where the given landmarks end up under the field, in pixels.

    The field is a *source* offset, so a landmark at ``p`` appears in the output at ``p - d(p)``
    to first order. This is how an operation checks that it achieved what it stated, rather than
    trusting the handle offsets it asked for.
    """
    height, width = dx.shape
    moved = np.empty((len(indices), 2), dtype=np.float32)
    for slot, index in enumerate(indices):
        x, y = landmarks[index]
        col = int(np.clip(round(float(x)), 0, width - 1))
        row = int(np.clip(round(float(y)), 0, height - 1))
        moved[slot] = (float(x) - dx[row, col], float(y) - dy[row, col])
    return moved
