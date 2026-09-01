"""Debug renderers. Nothing here is used by the retouching path.

The point of this module is that no claim about perception has to be taken on faith: every
mask, every landmark and every stage delta can be looked at.
"""

from __future__ import annotations

import cv2
import numpy as np

from portrait_engine.core.image import PortraitImage, _to_u8
from portrait_engine.core.types import FaceAnalysis

#: Distinct overlay colors (RGB) for stacked mask views.
PALETTE: tuple[tuple[int, int, int], ...] = (
    (232, 74, 95), (61, 169, 252), (255, 191, 0), (120, 224, 143),
    (186, 104, 245), (255, 138, 76), (0, 208, 197), (244, 114, 182),
)


def _u8(rgb: np.ndarray) -> np.ndarray:
    return _to_u8(np.clip(rgb, 0.0, 1.0))


def draw_bbox(image: PortraitImage, analysis: FaceAnalysis) -> np.ndarray:
    """Face box with its pixel size annotated."""
    canvas = _u8(image.rgb).copy()
    box = analysis.bbox
    thickness = max(2, int(round(analysis.geometry.interocular_px * 0.012)))
    cv2.rectangle(canvas, (box.x, box.y), (box.x2, box.y2), (61, 169, 252), thickness)
    label = f"{box.width}x{box.height}px  IOD={analysis.geometry.interocular_px:.0f}px"
    scale = max(0.4, analysis.geometry.interocular_px / 420.0)
    cv2.putText(
        canvas, label, (box.x, max(18, box.y - 8)), cv2.FONT_HERSHEY_SIMPLEX, scale,
        (61, 169, 252), max(1, thickness // 2), cv2.LINE_AA,
    )
    return canvas


def draw_landmarks(image: PortraitImage, analysis: FaceAnalysis, rings: bool = True) -> np.ndarray:
    """All 478 points, with the named rings drawn as closed polylines."""
    from portrait_engine.perception import landmark_sets as ls

    canvas = _u8(image.rgb).copy()
    points = np.rint(analysis.landmarks).astype(np.int32)
    radius = max(1, int(round(analysis.geometry.interocular_px * 0.004)))
    for x, y in points:
        cv2.circle(canvas, (int(x), int(y)), radius, (120, 224, 143), -1, cv2.LINE_AA)
    if rings:
        thickness = max(1, int(round(analysis.geometry.interocular_px * 0.006)))
        for index, ring in enumerate(ring for _, ring in sorted(ls.ALL_RINGS.items())):
            color = PALETTE[index % len(PALETTE)]
            cv2.polylines(canvas, [points[list(ring)]], True, color, thickness, cv2.LINE_AA)
    return canvas


def mask_preview(mask: np.ndarray) -> np.ndarray:
    """The alpha mask on its own, as an 8-bit grayscale image."""
    return _to_u8(np.clip(mask, 0.0, 1.0))


def mask_overlay(
    image: PortraitImage, mask: np.ndarray, color: tuple[int, int, int] = (232, 74, 95),
    opacity: float = 0.55,
) -> np.ndarray:
    """A mask tinted over the photo, so you can see what it covers and where it feathers."""
    base = _u8(image.rgb).astype(np.float32)
    tint = np.array(color, dtype=np.float32)
    alpha = (np.clip(mask, 0.0, 1.0) * opacity)[..., None]
    return np.clip(base * (1.0 - alpha) + tint * alpha, 0, 255).astype(np.uint8)


def mask_stack(image: PortraitImage, masks: dict[str, np.ndarray], opacity: float = 0.5) -> np.ndarray:
    """Several masks at once, each in its own color, with a legend."""
    base = _u8(image.rgb).astype(np.float32)
    out = base.copy()
    for index, mask in enumerate(masks.values()):
        color = np.array(PALETTE[index % len(PALETTE)], dtype=np.float32)
        alpha = (np.clip(mask, 0.0, 1.0) * opacity)[..., None]
        out = out * (1.0 - alpha) + color * alpha
    out = np.clip(out, 0, 255).astype(np.uint8)

    scale = max(0.45, out.shape[1] / 1600.0)
    step = int(28 * scale)
    for index, name in enumerate(masks):
        color = PALETTE[index % len(PALETTE)]
        y = step * (index + 1)
        cv2.putText(out, name, (12, y), cv2.FONT_HERSHEY_SIMPLEX, scale * 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, name, (12, y), cv2.FONT_HERSHEY_SIMPLEX, scale * 0.6, color, 1, cv2.LINE_AA)
    return out


def difference_map(before: np.ndarray, after: np.ndarray, gain: float = 12.0) -> np.ndarray:
    """Amplified absolute difference as a heat map, plus the numeric peak.

    This is the primary tool for checking that nothing was edited outside the permitted
    regions: anything lit up outside a mask is a bug.
    """
    delta = np.abs(after.astype(np.float32) - before.astype(np.float32)).mean(axis=2)
    peak = float(delta.max())
    amplified = np.clip(delta * gain, 0.0, 1.0)
    heat = cv2.applyColorMap(_to_u8(amplified), cv2.COLORMAP_INFERNO)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)

    scale = max(0.45, heat.shape[1] / 1600.0)
    label = f"max delta {peak * 255:.1f}/255   mean {delta.mean() * 255:.2f}/255   gain x{gain:g}"
    cv2.putText(heat, label, (12, int(30 * scale)), cv2.FONT_HERSHEY_SIMPLEX, scale * 0.6, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(heat, label, (12, int(30 * scale)), cv2.FONT_HERSHEY_SIMPLEX, scale * 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return heat


def signed_difference_map(before: np.ndarray, after: np.ndarray, gain: float = 12.0) -> np.ndarray:
    """Signed luminance difference: red where the image got darker, blue where it got lighter."""
    delta = (after.astype(np.float32) - before.astype(np.float32)).mean(axis=2)
    scaled = np.clip(delta * gain, -1.0, 1.0)
    out = np.zeros((*delta.shape, 3), dtype=np.float32)
    out[..., 0] = np.maximum(-scaled, 0.0)
    out[..., 2] = np.maximum(scaled, 0.0)
    out[..., 1] = 0.15 * np.abs(scaled)
    return _to_u8(out)


def before_after(
    before: np.ndarray, after: np.ndarray, labels: tuple[str, str] = ("before", "after"),
    divider: int = 6,
) -> np.ndarray:
    """Side-by-side comparison with a divider and labels."""
    left, right = _u8(before), _u8(after)
    height = left.shape[0]
    gap = np.full((height, divider, 3), 24, dtype=np.uint8)
    out = np.hstack([left, gap, right])

    scale = max(0.5, out.shape[1] / 2400.0)
    for text, x in ((labels[0], 14), (labels[1], left.shape[1] + divider + 14)):
        pos = (x, int(38 * scale))
        cv2.putText(out, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(out, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def crop_zoom(
    image: np.ndarray, analysis: FaceAnalysis, region: str = "face", zoom: float = 1.0
) -> np.ndarray:
    """Crop to a region of interest, for judging texture at 100%.

    ``region`` is one of ``face``, ``left_eye``, ``right_eye``, ``mouth``, ``cheek``.
    """
    from portrait_engine.perception import landmark_sets as ls

    height, width = image.shape[:2]
    points = analysis.landmarks
    iod = analysis.geometry.interocular_px
    if region == "face":
        box = analysis.bbox.expanded(1.15, (height, width))
        x0, y0, x1, y1 = box.x, box.y, box.x2, box.y2
    else:
        anchors = {
            "left_eye": ls.LEFT_EYE, "right_eye": ls.RIGHT_EYE,
            "mouth": ls.LIPS_OUTER, "cheek": ls.RIGHT_CHEEK,
        }[region]
        sel = points[list(anchors)]
        cx, cy = sel[:, 0].mean(), sel[:, 1].mean()
        half = iod * 0.55
        x0, y0 = int(max(0, cx - half)), int(max(0, cy - half))
        x1, y1 = int(min(width, cx + half)), int(min(height, cy + half))
    crop = image[y0:y1, x0:x1]
    if zoom != 1.0 and crop.size:
        crop = cv2.resize(crop, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_NEAREST)
    return crop
