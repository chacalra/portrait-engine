"""Color space conversions with explicit precision and range contracts.

The engine has one canonical in-memory representation, defined in
:mod:`portrait_engine.core.image`:

    ``float32``, shape ``(H, W, 3)``, channel order **RGB**, sRGB-encoded, range ``[0, 1]``.

Every function here documents the range it returns, because getting this wrong is the
most common source of bugs in portrait retouching code. Two rules:

* Perceptual work (blemish detection, chroma, teeth classification) happens in CIELAB.
* Photometric work (exposure, illumination gain) happens in **linear light**. A gain applied
  to sRGB-encoded values is not an exposure change and shifts saturation.
"""

from __future__ import annotations

import cv2
import numpy as np

# Rec.709 luma weights, for use on *linear* RGB.
LUMA_709 = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)

# CIELAB ranges as produced by OpenCV for float32 input.


def srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    """sRGB-encoded ``[0,1]`` to linear light ``[0,1]``. Exact IEC 61966-2-1 EOTF."""
    x = np.clip(rgb.astype(np.float32, copy=False), 0.0, 1.0)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4).astype(np.float32)


def linear_to_srgb(linear: np.ndarray) -> np.ndarray:
    """Linear light ``[0,1]`` to sRGB-encoded ``[0,1]``. Inverse of :func:`srgb_to_linear`."""
    x = np.clip(linear.astype(np.float32, copy=False), 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1.0 / 2.4) - 0.055).astype(
        np.float32
    )


def linear_luma(rgb_srgb: np.ndarray) -> np.ndarray:
    """Rec.709 luminance in linear light, from an sRGB-encoded RGB image. Returns ``[0,1]``."""
    return (srgb_to_linear(rgb_srgb) @ LUMA_709).astype(np.float32)


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """sRGB-encoded RGB ``[0,1]`` to CIELAB. ``L`` in ``[0,100]``, ``a``/``b`` in ``[-127,127]``."""
    return cv2.cvtColor(np.ascontiguousarray(rgb, dtype=np.float32), cv2.COLOR_RGB2Lab)


def lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    """CIELAB back to sRGB-encoded RGB, clipped to ``[0,1]``."""
    rgb = cv2.cvtColor(np.ascontiguousarray(lab, dtype=np.float32), cv2.COLOR_Lab2RGB)
    return np.clip(rgb, 0.0, 1.0)


def rgb_to_ycrcb(rgb: np.ndarray) -> np.ndarray:
    """sRGB-encoded RGB ``[0,1]`` to YCrCb, all channels in ``[0,1]`` (Cr/Cb centered on 0.5)."""
    return cv2.cvtColor(np.ascontiguousarray(rgb, dtype=np.float32), cv2.COLOR_RGB2YCrCb)



def soft_highlight_rolloff(x: np.ndarray, knee: float = 0.85) -> np.ndarray:
    """Compress values above ``knee`` into ``[knee, 1]`` instead of hard-clipping them.

    Used after any operation that can push linear values past 1.0. A hard clip flattens
    specular highlights into paper-white blobs; this keeps them ordered and monotonic.
    """
    x = x.astype(np.float32, copy=False)
    head = 1.0 - knee
    over = np.maximum(x - knee, 0.0)
    # Rational rolloff: f(0)=0, f'(0)=1, asymptote at `head`.
    compressed = head * over / (head + over)
    return np.where(x <= knee, x, knee + compressed).astype(np.float32)
