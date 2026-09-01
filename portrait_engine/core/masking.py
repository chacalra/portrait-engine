"""The single place in the engine where original and edited pixels are combined.

Every tool must route its output through :func:`apply_masked`. That is what makes the
identity guardrails enforceable rather than aspirational:

* the edited image must have exactly the same shape as the input, so no tool can resize,
  crop or warp;
* pixels where ``alpha == 0`` are copied from the original bit for bit;
* in strict mode, any measurable change outside the mask raises.

This is a generalization of the compositing step in the OpenCV G-API face beautification
sample, which composes ``O = b*I + p*U + s*L`` from disjoint masks.
"""

from __future__ import annotations

import numpy as np

from portrait_engine.core.types import MaskContractError

# Tolerance for "no change outside the mask". One 8-bit level is 1/255 = 0.0039, so this is
# a quarter of a level: invisible, but tight enough to catch a real leak.
OUT_OF_MASK_TOLERANCE = 1e-3


def validate_alpha(alpha: np.ndarray, shape: tuple[int, int], name: str = "alpha") -> np.ndarray:
    """Check the mask contract and return a ``float32`` ``(H, W)`` array."""
    arr = np.asarray(alpha, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[..., 0]
    if arr.shape != shape:
        raise MaskContractError(f"{name} has shape {arr.shape}, expected {shape}")
    if not np.isfinite(arr).all():
        raise MaskContractError(f"{name} contains non-finite values")
    if arr.min() < -1e-4 or arr.max() > 1.0 + 1e-4:
        raise MaskContractError(f"{name} out of [0,1]: min={arr.min()} max={arr.max()}")
    return np.clip(arr, 0.0, 1.0)


def apply_masked(
    original: np.ndarray,
    edited: np.ndarray,
    alpha: np.ndarray,
    *,
    strict: bool = True,
    name: str = "tool",
) -> np.ndarray:
    """Composite ``edited`` over ``original`` using ``alpha``.

    Args:
        original: ``float32`` RGB ``(H, W, 3)`` in ``[0, 1]``.
        edited: same shape and range. Anything else is a bug in the caller.
        alpha: ``(H, W)`` in ``[0, 1]``.
        strict: verify that nothing changed where ``alpha`` is zero.
        name: used in error messages.

    Returns:
        A new ``float32`` array. Neither input is modified.
    """
    if original.shape != edited.shape:
        raise MaskContractError(
            f"{name}: geometry change detected, {original.shape} -> {edited.shape}. "
            "Tools must not resize, crop or warp the image."
        )
    if original.ndim != 3 or original.shape[2] != 3:
        raise MaskContractError(f"{name}: expected (H, W, 3), got {original.shape}")

    shape = (original.shape[0], original.shape[1])
    a = validate_alpha(alpha, shape, f"{name} alpha")[..., None]

    edited = np.clip(np.nan_to_num(edited, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
    out = original * (1.0 - a) + edited * a

    # Where alpha is exactly zero, copy the original so the result is bit-identical there
    # even after float rounding.
    zero = a[..., 0] <= 0.0
    if zero.any():
        out[zero] = original[zero]

    if strict:
        verify_no_change_outside(original, out, a[..., 0], name=name)
    return out.astype(np.float32, copy=False)


def verify_no_change_outside(
    original: np.ndarray,
    result: np.ndarray,
    alpha: np.ndarray,
    *,
    tolerance: float = OUT_OF_MASK_TOLERANCE,
    name: str = "tool",
) -> float:
    """Raise if ``result`` differs from ``original`` where the mask has no influence.

    "No influence" means ``alpha`` is below the tolerance, not exactly zero, because a
    feathered mask has a long tail of very small values that legitimately produce
    sub-quantization changes.

    Returns:
        The maximum absolute deviation found outside the mask.
    """
    outside = alpha <= tolerance
    if not outside.any():
        return 0.0
    delta = np.abs(result[outside] - original[outside])
    worst = float(delta.max()) if delta.size else 0.0
    if worst > tolerance:
        raise MaskContractError(
            f"{name} modified {int((delta > tolerance).sum())} pixel components outside its "
            f"mask (max delta {worst:.5f} > {tolerance}). This is an identity-preservation "
            "violation."
        )
    return worst


def subtract_protected(mask: np.ndarray, protected: np.ndarray) -> np.ndarray:
    """Remove protected regions from a skin mask. Never the other way around."""
    return np.clip(mask * (1.0 - protected), 0.0, 1.0).astype(np.float32)


def edited_area_fraction(alpha: np.ndarray) -> float:
    """Fraction of the frame effectively edited, for stats and sanity checks."""
    return float(alpha.sum() / max(1, alpha.size))
