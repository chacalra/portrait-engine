"""``DebugRecorder``: writes inspectable artifacts to disk during a pipeline run.

Layout::

    <debug_dir>/
      perception/   00_bbox.jpg  01_landmarks.jpg  02_mask_overview.jpg
      masks/        skin.png  teeth.png  left_under_eye.png  ... skin_probability.png
      stages/       01_original.jpg  02_skin_cleanup.jpg  ...  07_final.jpg
      diffs/        02_skin_cleanup_diff.jpg  ...
      before_after.jpg
      difference_map.jpg
      difference_signed.jpg
      report.json
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from portrait_engine.core.image import PortraitImage
from portrait_engine.core.types import FaceAnalysis
from portrait_engine.debug import visualize as viz

#: Masks written out by default. The overview composite uses the first five.
DEFAULT_MASKS: tuple[str, ...] = (
    "skin", "teeth", "left_under_eye", "right_under_eye", "protected",
    "hair_like", "face_photometric", "mouth_interior", "nostrils",
    "eyebrows", "lips", "nose", "face_oval",
    "left_under_eye_reference", "right_under_eye_reference",
)


@dataclass
class DebugRecorder:
    """Writes debug artifacts. Every method is a no-op when ``enabled`` is False."""

    directory: str | None = None
    enabled: bool = False
    jpeg_quality: int = 92
    written: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.enabled and not self.directory:
            raise ValueError("DebugRecorder needs a directory when enabled")
        if self.enabled:
            for sub in ("perception", "masks", "stages", "diffs"):
                os.makedirs(os.path.join(self.directory, sub), exist_ok=True)  # type: ignore[arg-type]

    # ------------------------------------------------------------------ helpers
    def _path(self, *parts: str) -> str:
        return os.path.join(self.directory, *parts)  # type: ignore[arg-type]

    def _write_rgb(self, relative: str, rgb_u8: np.ndarray) -> None:
        path = self._path(relative)
        cv2.imwrite(path, cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR))
        self.written.append(path)

    def _write_gray(self, relative: str, gray_u8: np.ndarray) -> None:
        path = self._path(relative)
        cv2.imwrite(path, gray_u8)
        self.written.append(path)

    # ------------------------------------------------------------------ perception
    def record_perception(self, image: PortraitImage, analysis: FaceAnalysis) -> None:
        """Face box, landmarks, a colored overview of the main masks, and every mask alone."""
        if not self.enabled:
            return
        self._write_rgb("perception/00_bbox.jpg", viz.draw_bbox(image, analysis))
        self._write_rgb("perception/01_landmarks.jpg", viz.draw_landmarks(image, analysis))
        overview = {
            name: analysis.masks[name]
            for name in DEFAULT_MASKS[:5]
            if name in analysis.masks
        }
        self._write_rgb("perception/02_mask_overview.jpg", viz.mask_stack(image, overview))

        for name in DEFAULT_MASKS:
            if name in analysis.masks:
                self._write_gray(f"masks/{name}.png", viz.mask_preview(analysis.masks[name]))
        self._write_gray("masks/skin_probability.png", viz.mask_preview(analysis.skin_probability))

    # ------------------------------------------------------------------ stages
    def record_stage(
        self,
        index: int,
        name: str,
        before: np.ndarray,
        after: np.ndarray,
        extra: dict[str, np.ndarray] | None = None,
    ) -> None:
        """One pipeline stage: the result, its own difference map, and any tool debug layers."""
        if not self.enabled:
            return
        label = f"{index:02d}_{name}"
        self._write_rgb(f"stages/{label}.jpg", viz._u8(after))
        self._write_rgb(f"diffs/{label}_diff.jpg", viz.difference_map(before, after, gain=20.0))
        for key, layer in (extra or {}).items():
            if layer.ndim == 2:
                self._write_gray(f"diffs/{label}_{key}.png", viz.mask_preview(layer))

    def record_original(self, image: PortraitImage) -> None:
        if not self.enabled:
            return
        self._write_rgb("stages/01_original.jpg", viz._u8(image.rgb))

    def record_final(
        self, original: np.ndarray, final: np.ndarray, analysis: FaceAnalysis | None = None
    ) -> None:
        """Final image, the overall before/after, and both difference maps."""
        if not self.enabled:
            return
        self._write_rgb("stages/99_final.jpg", viz._u8(final))
        self._write_rgb("before_after.jpg", viz.before_after(original, final))
        self._write_rgb("difference_map.jpg", viz.difference_map(original, final, gain=12.0))
        self._write_rgb("difference_signed.jpg", viz.signed_difference_map(original, final, gain=12.0))
        if analysis is not None:
            crop_before = viz.crop_zoom(viz._u8(original), analysis, "face")
            crop_after = viz.crop_zoom(viz._u8(final), analysis, "face")
            self._write_rgb(
                "before_after_face.jpg",
                viz.before_after(crop_before / 255.0, crop_after / 255.0),
            )

    def record_report(self, report: dict[str, Any]) -> None:
        if not self.enabled:
            return
        path = self._path("report.json")
        with open(path, "w") as handle:
            json.dump(report, handle, indent=2, default=_json_default)
        self.written.append(path)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return f"<array shape={value.shape}>"
    return str(value)
