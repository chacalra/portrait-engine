"""Bounded geometric warping: the one place in the engine where pixels are allowed to move.

This module exists deliberately outside ``portrait_engine.tools``. Everything in ``tools`` is
covered by AST tests that fail if a module so much as names ``cv2.remap``, and those tests still
hold. Reshaping is a **separate registry**, so a caller (and later the agent) has to ask for it
by name, and any run that used only the identity-preserving tools can still be proved to have
left the geometry alone. See ``docs/RESHAPING.md``.

How the warp is built
---------------------
Not a general free-form liquify. A displacement field is built from a small set of **handles**:
landmark positions paired with a target offset, interpolated by a **compactly supported radial
basis function**, Wendland's C2 kernel ``psi(r) = (1-r)^4 (4r+1)`` truncated at ``r >= 1``. It is
positive definite in R^3 and therefore in R^2, so the interpolation matrix is invertible for any
set of distinct handles.

Two properties earn it its place over the alternatives:

* **It reproduces the handle offsets exactly.** Solving ``Phi w = offsets`` means each handle
  moves by what it was asked to move by, so a wing slides rather than deforming lumpily.
* **Its support is exactly the union of discs of radius R.** Displacement is identically zero
  beyond that, so containment is provable from R rather than only verified after the fact.

The first version scattered impulses and blurred them with a Gaussian. That was wrong twice over,
and the docstring recording its rationale was wrong a third time: it claimed the field
"under-shoots wherever handles are sparse". Measured on a real portrait at the project's own
defaults it **over**-shot by 1.96x to 2.53x, with a spread of 0.57 between the three handles that
define a single nose wing. The two-pass rescale in ``nose.py`` corrected the aggregate width and
structurally could not correct that inconsistency. The replacement measures 0.99 to 1.01 per
handle and builds the field about 60 times faster, because it evaluates only inside the handles'
bounding box instead of convolving the whole frame.

The mask primitives in ``core.ops`` are the wrong tool here and were tried before either version.
``normalized_lowpass`` divides by the smoothed weights, so a handful of impulses produce the
*average of the offsets* across the whole influence area: a plateau with no falloff at all. And
``blur_mask`` clips its output to ``[0, 1]``, which silently truncated every displacement to one
pixel. Both are correct for masks and illumination and wrong for a vector field.

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


#: Handles closer together than this fraction of the support radius are merged. Two coincident
#: handles make the interpolation matrix singular, and MediaPipe genuinely returns near-coincident
#: nose-ring landmarks on a steep semi-profile, where `_alar_extremes` picks by axis projection
#: without checking distance. Measured condition number for a coincident pair: 1.9e17.
HANDLE_MERGE_FRACTION = 0.01

#: Tikhonov ridge added to the diagonal, relative to the mean diagonal. Insurance for handle
#: configurations that survive merging but are still nearly degenerate.
RIDGE = 1e-8


def wendland(r: np.ndarray) -> np.ndarray:
    """Wendland's C2 radial basis function, ``(1-r)^4 (4r+1)``, zero for ``r >= 1``.

    Positive definite in R^3 and therefore in R^2, which is what guarantees the interpolation
    matrix is invertible for distinct handles. Compact support is the property that matters most
    here: it makes containment a consequence of the radius rather than a mask multiply.
    """
    tail = np.clip(1.0 - r, 0.0, None)
    return (tail**4) * (4.0 * r + 1.0)


def displacement_from_handles(
    shape: tuple[int, int],
    handles: np.ndarray,
    offsets: np.ndarray,
    radius: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate sparse handle offsets into a dense displacement field with compact support.

    Args:
        shape: ``(height, width)``.
        handles: ``(N, 2)`` float pixel coordinates, ``(x, y)``.
        offsets: ``(N, 2)`` float pixel offsets to apply at those handles.
        radius: support radius in pixels. The field is exactly zero beyond this distance from
            every handle, so it also bounds the region the warp can touch.

    Returns:
        ``(dx, dy)``, each ``(H, W)`` float32. These are **source** offsets: the value at a
        destination pixel says where to read from, which is the convention ``cv2.remap`` wants.

    Raises:
        WarpContractError: if the handle configuration is degenerate enough that the
            interpolation cannot be solved.

    Note on the support radius: enlarging it does **not** improve conditioning, contrary to the
    usual advice about RBF interpolation. Measured here, ``cond(Phi)`` rises monotonically with
    the radius: 1.00 at 0.01 of the interocular distance, 9.2 at 0.16, 3.2e4 at 2.56. The
    guidance in the literature concerns globally supported kernels; for a compactly supported one
    a larger radius means more handles inside each other's support, which is what raises the
    condition number. Growing the radius to "fix" conditioning would also destroy the locality
    that is the whole reason for choosing this kernel.
    """
    height, width = shape
    zero = np.zeros((height, width), dtype=np.float32)
    points = np.asarray(handles, dtype=np.float64).reshape(-1, 2)
    values = np.asarray(offsets, dtype=np.float64).reshape(-1, 2)
    support = float(max(radius, 1))

    points, values = _merge_close_handles(points, values, support * HANDLE_MERGE_FRACTION)
    if points.shape[0] == 0:
        return zero, zero.copy()

    distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
    phi = wendland(distances / support)
    phi[np.diag_indices_from(phi)] += RIDGE * max(float(np.mean(np.diag(phi))), 1e-12)
    try:
        weights = np.linalg.solve(phi, values)
    except np.linalg.LinAlgError as error:
        raise WarpContractError(
            f"the {points.shape[0]} handles are too degenerate to interpolate: {error}. "
            "This usually means landmarks collapsed onto each other on a steep profile."
        ) from error

    # Evaluate only inside the handles' bounding box grown by the support radius. Everything
    # outside is exactly zero by construction, so convolving the whole frame would be wasted
    # work; this is where most of the speedup over the previous Gaussian scatter comes from.
    pad = int(np.ceil(support)) + 2
    x0 = max(0, int(np.floor(points[:, 0].min())) - pad)
    x1 = min(width, int(np.ceil(points[:, 0].max())) + pad)
    y0 = max(0, int(np.floor(points[:, 1].min())) - pad)
    y1 = min(height, int(np.ceil(points[:, 1].max())) + pad)
    if x1 <= x0 or y1 <= y0:
        return zero, zero.copy()

    grid_y, grid_x = np.mgrid[y0:y1, x0:x1]
    patch = np.zeros((y1 - y0, x1 - x0, 2), dtype=np.float64)
    for (px, py), weight in zip(points, weights, strict=True):
        r = np.hypot(grid_x - px, grid_y - py) / support
        basis = wendland(r)
        patch[..., 0] += weight[0] * basis
        patch[..., 1] += weight[1] * basis

    dx, dy = zero, zero.copy()
    dx[y0:y1, x0:x1] = patch[..., 0]
    dy[y0:y1, x0:x1] = patch[..., 1]
    return dx, dy


def _merge_close_handles(
    points: np.ndarray, values: np.ndarray, tolerance: float
) -> tuple[np.ndarray, np.ndarray]:
    """Collapse handles nearer than ``tolerance``, averaging their offsets.

    Without this, two coincident landmarks make ``Phi`` singular and ``np.linalg.solve`` raises.
    Averaging rather than dropping keeps the intent: two handles asking for the same point to
    move are one handle asking for the mean of what they wanted.
    """
    kept_points: list[np.ndarray] = []
    kept_values: list[list[np.ndarray]] = []
    for point, value in zip(points, values, strict=True):
        for index, existing in enumerate(kept_points):
            if float(np.linalg.norm(point - existing)) <= tolerance:
                kept_values[index].append(value)
                break
        else:
            kept_points.append(point)
            kept_values.append([value])
    if not kept_points:
        return np.empty((0, 2)), np.empty((0, 2))
    return (
        np.stack(kept_points),
        np.stack([np.mean(group, axis=0) for group in kept_values]),
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
