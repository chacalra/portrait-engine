"""Reusable signal-processing primitives.

Everything the tools need is built from a small set of edge-aware primitives, so that
behaviour is consistent across tools and there is one place to test and to port later.

The central primitive is the **guided filter** (He, Sun, Tang, ECCV 2010), implemented from
the paper over ``cv2.boxFilter``. It is used for three different jobs:

1. edge-preserving smoothing (skin smoothing, illumination estimation),
2. guided feathering of masks (the "guided feathering" step described by FabSoften),
3. multi-band decomposition of an image into base / mid / fine detail.

See ``docs/ALGORITHMS.md`` for the formulas and ``docs/RESEARCH.md`` for why guided filtering
was chosen over bilateral, anisotropic diffusion and wavelets.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


def box_filter(src: np.ndarray, radius: int) -> np.ndarray:
    """Normalized box filter with a ``(2r+1)`` window and reflected borders."""
    k = 2 * int(radius) + 1
    return cv2.boxFilter(
        src, ddepth=-1, ksize=(k, k), normalize=True, borderType=cv2.BORDER_REFLECT101
    )


def guided_filter(
    src: np.ndarray, guide: np.ndarray, radius: int, eps: float, subsample: int = 1
) -> np.ndarray:
    """Grayscale-guided guided filter. ``src`` and ``guide`` are ``float32`` 2-D arrays.

    Args:
        src: signal to filter, shape ``(H, W)``.
        guide: guidance image, shape ``(H, W)``. Same units as ``src`` is not required.
        radius: window radius in pixels. Always derive it from face scale, never hardcode.
        eps: regularization. Larger means smoother. In the paper it is a variance, so it has
            the units of ``guide`` squared.
        subsample: fast guided filter factor (He and Sun, 2015). ``>1`` computes the
            coefficients on a downscaled image and upsamples them. Cheap and visually
            equivalent for large radii.

    Returns:
        ``float32`` array shaped like ``src``.
    """
    src = np.ascontiguousarray(src, dtype=np.float32)
    guide = np.ascontiguousarray(guide, dtype=np.float32)
    if src.shape != guide.shape:
        raise ValueError(f"guided_filter shape mismatch: {src.shape} vs {guide.shape}")
    radius = max(1, int(radius))

    if subsample > 1:
        h, w = src.shape
        sh, sw = max(2, h // subsample), max(2, w // subsample)
        small_src = cv2.resize(src, (sw, sh), interpolation=cv2.INTER_AREA)
        small_guide = cv2.resize(guide, (sw, sh), interpolation=cv2.INTER_AREA)
        r_small = max(1, radius // subsample)
        mean_g = box_filter(small_guide, r_small)
        mean_s = box_filter(small_src, r_small)
        cov = box_filter(small_guide * small_src, r_small) - mean_g * mean_s
        var = box_filter(small_guide * small_guide, r_small) - mean_g * mean_g
        a = cov / (var + eps)
        b = mean_s - a * mean_g
        a = cv2.resize(box_filter(a, r_small), (w, h), interpolation=cv2.INTER_LINEAR)
        b = cv2.resize(box_filter(b, r_small), (w, h), interpolation=cv2.INTER_LINEAR)
        return (a * guide + b).astype(np.float32)

    mean_g = box_filter(guide, radius)
    mean_s = box_filter(src, radius)
    cov = box_filter(guide * src, radius) - mean_g * mean_s
    var = box_filter(guide * guide, radius) - mean_g * mean_g
    a = cov / (var + eps)
    b = mean_s - a * mean_g
    return (box_filter(a, radius) * guide + box_filter(b, radius)).astype(np.float32)


def guided_filter_color(
    src: np.ndarray, guide_rgb: np.ndarray, radius: int, eps: float, subsample: int = 1
) -> np.ndarray:
    """Guided filter with a 3-channel guide, applied per channel of the guide's luma.

    A full color-guided guided filter needs a 3x3 matrix inverse per pixel. For mask
    feathering the luma guide is enough and much cheaper, so that is what this does: it
    reduces the guide to Rec.709 luma and delegates. Kept as a separate name so the call
    sites read correctly and a full color implementation can replace it later.
    """
    from portrait_engine.core.colorspace import linear_luma

    return guided_filter(src, linear_luma(guide_rgb), radius, eps, subsample=subsample)


@dataclass(frozen=True)
class Bands:
    """Three-band decomposition of a single channel. ``base + mid + fine`` equals the input.

    * ``base``: illumination and large-scale tone. Never touched by smoothing.
    * ``mid``: blotches, uneven tone, spots. This is what skin smoothing attenuates.
    * ``fine``: pores and true skin texture. Preserved so the result does not look plastic.
    """

    base: np.ndarray
    mid: np.ndarray
    fine: np.ndarray

    def recombine(self, mid_gain: float = 1.0, fine_gain: float = 1.0) -> np.ndarray:
        return (self.base + self.mid * mid_gain + self.fine * fine_gain).astype(np.float32)


def decompose_bands(
    channel: np.ndarray,
    guide: np.ndarray,
    radius_fine: int,
    radius_base: int,
    eps_fine: float,
    eps_base: float,
    subsample: int = 1,
) -> Bands:
    """Split ``channel`` into base / mid / fine using two guided filters.

    Both cutoffs are edge-aware, which is the reason this is used instead of a wavelet
    decomposition: reinjecting detail near an eyelid or an eyebrow does not ring.
    """
    fine_base = guided_filter(channel, guide, radius_fine, eps_fine, subsample=subsample)
    base = guided_filter(channel, guide, radius_base, eps_base, subsample=subsample)
    return Bands(base=base, mid=(fine_base - base), fine=(channel - fine_base))


def feather_mask(
    mask: np.ndarray,
    guide_rgb: np.ndarray,
    radius: int,
    eps: float = 1e-4,
    subsample: int = 1,
) -> np.ndarray:
    """Guided feathering: turn a hard mask into an alpha matte that follows image edges.

    This is what keeps a coarse polygon from bleeding onto an eyebrow. A Gaussian blur
    would not do this; it would just make the bleed soft.
    """
    alpha = guided_filter_color(
        mask.astype(np.float32, copy=False), guide_rgb, radius, eps, subsample=subsample
    )
    return np.clip(alpha, 0.0, 1.0).astype(np.float32)


def normalized_lowpass(
    channel: np.ndarray, weights: np.ndarray, radius: int
) -> np.ndarray:
    """Smooth low-pass of ``channel`` using only the pixels where ``weights`` is non-zero.

    Normalized convolution: ``blur(channel * w) / blur(w)``. Two properties matter here:

    * it is genuinely **smooth**, with no edge preservation, which is what an illumination
      estimate must be. An edge-aware estimate would follow the image's own edges, and a gain
      field derived from it would cancel local contrast rather than even out lighting;
    * content outside the weights (hair, background) cannot bleed in.

    Values where ``weights`` is zero everywhere within the window fall back to the global mean.
    """
    channel = channel.astype(np.float32, copy=False)
    weights = weights.astype(np.float32, copy=False)
    k = 2 * max(1, int(radius)) + 1
    numerator = cv2.GaussianBlur(channel * weights, (k, k), 0, borderType=cv2.BORDER_REFLECT101)
    denominator = cv2.GaussianBlur(weights, (k, k), 0, borderType=cv2.BORDER_REFLECT101)
    fallback = float((channel * weights).sum() / max(float(weights.sum()), 1e-6))
    out = np.where(denominator > 1e-4, numerator / np.maximum(denominator, 1e-6), fallback)
    return out.astype(np.float32)


def blur_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    """Plain Gaussian feathering, for masks whose boundary should be geometric, not
    image-aligned (for example the wide falloff of the illumination gain field)."""
    radius = max(1, int(radius))
    k = 2 * radius + 1
    return np.clip(
        cv2.GaussianBlur(mask.astype(np.float32, copy=False), (k, k), 0, borderType=cv2.BORDER_REFLECT101),
        0.0,
        1.0,
    ).astype(np.float32)


def dilate_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.astype(np.float32, copy=True)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * int(radius) + 1,) * 2)
    return cv2.dilate(mask.astype(np.float32, copy=False), k)


def erode_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.astype(np.float32, copy=True)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * int(radius) + 1,) * 2)
    return cv2.erode(mask.astype(np.float32, copy=False), k)


def high_frequency_energy(channel: np.ndarray, weights: np.ndarray | None = None) -> float:
    """Mean absolute Laplacian response, optionally weighted by a mask.

    Used as a texture metric: comparing this before and after a tool tells you whether skin
    texture survived. It is a diagnostic, not a filter.
    """
    lap = cv2.Laplacian(channel.astype(np.float32, copy=False), cv2.CV_32F, ksize=3)
    lap = np.abs(lap)
    if weights is None:
        return float(lap.mean())
    total = float(weights.sum())
    if total <= 1e-6:
        return 0.0
    return float((lap * weights).sum() / total)


def masked_percentile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    """Percentile of ``values`` restricted to ``weights > 0.5``. Returns ``nan`` if empty."""
    sel = values[weights > 0.5]
    if sel.size == 0:
        return float("nan")
    return float(np.percentile(sel, q))


def smoothstep(x: np.ndarray, edge0: float, edge1: float) -> np.ndarray:
    """Hermite smoothstep, clamped. Used everywhere a soft threshold is needed."""
    if edge1 == edge0:
        return (x >= edge1).astype(np.float32)
    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0).astype(np.float32)
    return (t * t * (3.0 - 2.0 * t)).astype(np.float32)
