"""Core data types and errors. All perception output is immutable."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np


class PortraitEngineError(Exception):
    """Base class for every error this package raises on purpose."""


class InvalidImageError(PortraitEngineError):
    """The file could not be decoded, or the array is not a usable RGB image."""


class NoFaceDetectedError(PortraitEngineError):
    """No face was found. The MVP requires exactly one clearly visible face."""


class MultipleFacesError(PortraitEngineError):
    """More than one face was found. Out of scope for the MVP."""


class StrengthOutOfRangeError(PortraitEngineError, ValueError):
    """A strength value was outside ``[0, 1]``."""


class MaskContractError(PortraitEngineError):
    """A mask violated the mask contract (dtype, range, shape) or a tool wrote outside it."""


@dataclass(frozen=True)
class BBox:
    """Axis-aligned box in pixel coordinates. ``x``/``y`` are the top-left corner."""

    x: int
    y: int
    width: int
    height: int

    @property
    def x2(self) -> int:
        return self.x + self.width

    @property
    def y2(self) -> int:
        return self.y + self.height

    @property
    def center(self) -> tuple[float, float]:
        return (self.x + self.width / 2.0, self.y + self.height / 2.0)

    def clipped(self, shape: tuple[int, int]) -> BBox:
        h, w = shape
        x = max(0, min(self.x, w - 1))
        y = max(0, min(self.y, h - 1))
        return BBox(x, y, max(1, min(self.width, w - x)), max(1, min(self.height, h - y)))

    def expanded(self, factor: float, shape: tuple[int, int]) -> BBox:
        cx, cy = self.center
        nw, nh = self.width * factor, self.height * factor
        return BBox(int(round(cx - nw / 2)), int(round(cy - nh / 2)), int(round(nw)), int(round(nh))).clipped(shape)

    def as_dict(self) -> dict[str, int]:
        return {"x": self.x, "y": self.y, "width": self.width, "height": self.height}


@dataclass(frozen=True)
class FaceGeometry:
    """Scale and pose descriptors. Read-only: no tool may change facial geometry.

    ``interocular_px`` is the engine's unit of length. Every filter radius, blob size and
    offset is expressed as a fraction of it, which is what makes a given ``strength`` look
    the same on an 800 px and on a 6000 px photo.
    """

    interocular_px: float
    face_width_px: float
    face_height_px: float
    roll_degrees: float
    eye_left_center: tuple[float, float]
    eye_right_center: tuple[float, float]

    def scaled(self, fraction: float, minimum: int = 1) -> int:
        """A pixel length equal to ``fraction`` of the interocular distance."""
        return max(minimum, int(round(self.interocular_px * fraction)))

    def as_dict(self) -> dict[str, Any]:
        return {
            "interocular_px": round(self.interocular_px, 2),
            "face_width_px": round(self.face_width_px, 2),
            "face_height_px": round(self.face_height_px, 2),
            "roll_degrees": round(self.roll_degrees, 2),
        }


class MaskSet(Mapping[str, np.ndarray]):
    """Immutable mapping of name to alpha mask.

    Mask contract, enforced on construction:

    * ``float32``, shape ``(H, W)`` matching the full image,
    * values in ``[0, 1]``,
    * already feathered by the perception layer.

    Tools may narrow a mask (multiply) but never widen it.
    """

    __slots__ = ("_masks", "_shape")

    def __init__(self, masks: Mapping[str, np.ndarray], shape: tuple[int, int]):
        self._shape = shape
        checked: dict[str, np.ndarray] = {}
        for name, mask in masks.items():
            arr = np.asarray(mask)
            if arr.shape != shape:
                raise MaskContractError(f"mask {name!r} has shape {arr.shape}, expected {shape}")
            if arr.dtype != np.float32:
                arr = arr.astype(np.float32)
            if not np.isfinite(arr).all():
                raise MaskContractError(f"mask {name!r} contains non-finite values")
            lo, hi = float(arr.min()), float(arr.max())
            if lo < -1e-4 or hi > 1.0 + 1e-4:
                raise MaskContractError(f"mask {name!r} out of [0,1]: min={lo} max={hi}")
            arr = np.clip(arr, 0.0, 1.0)
            arr.setflags(write=False)
            checked[name] = arr
        self._masks = checked

    def __getitem__(self, key: str) -> np.ndarray:
        try:
            return self._masks[key]
        except KeyError as exc:
            raise MaskContractError(
                f"mask {key!r} not available; have {sorted(self._masks)}"
            ) from exc

    def __contains__(self, key: object) -> bool:
        """Explicit, because the inherited one does not work here.

        `Mapping.__contains__` is implemented by calling `__getitem__` and catching `KeyError`.
        This class raises `MaskContractError` instead, to give a useful message, so the
        inherited version let that propagate: `"absent" in masks` raised rather than returning
        False. Every defensive `if name in analysis.masks` in the codebase was silently broken,
        including the debug recorder's, which is exactly the place that needs it.
        """
        return key in self._masks

    def __iter__(self) -> Iterator[str]:
        return iter(self._masks)

    def __len__(self) -> int:
        return len(self._masks)

    @property
    def shape(self) -> tuple[int, int]:
        return self._shape

    def area(self, key: str) -> float:
        """Effective area in pixels (sum of alpha), a cheap way to detect empty masks."""
        return float(self[key].sum())

    def with_masks(self, extra: Mapping[str, np.ndarray]) -> MaskSet:
        return MaskSet({**self._masks, **dict(extra)}, self._shape)


@dataclass(frozen=True)
class FaceAnalysis:
    """Everything the perception layer knows about one face. Computed once per image."""

    image_shape: tuple[int, int]
    bbox: BBox
    landmarks: np.ndarray  # (N, 2) float32, pixel coordinates
    landmarks_3d: np.ndarray  # (N, 3) float32, MediaPipe normalized space
    masks: MaskSet
    skin_probability: np.ndarray  # (H, W) float32 in [0, 1]
    geometry: FaceGeometry

    def __post_init__(self) -> None:
        object.__setattr__(self, "landmarks", np.ascontiguousarray(self.landmarks, np.float32))
        object.__setattr__(self, "landmarks_3d", np.ascontiguousarray(self.landmarks_3d, np.float32))
        self.landmarks.setflags(write=False)
        self.landmarks_3d.setflags(write=False)
        prob = np.ascontiguousarray(self.skin_probability, np.float32)
        prob.setflags(write=False)
        object.__setattr__(self, "skin_probability", prob)

    def mask(self, name: str) -> np.ndarray:
        return self.masks[name]

    def summary(self) -> dict[str, Any]:
        return {
            "image_shape": list(self.image_shape),
            "bbox": self.bbox.as_dict(),
            "landmark_count": int(self.landmarks.shape[0]),
            "geometry": self.geometry.as_dict(),
            "masks": {name: round(self.masks.area(name), 1) for name in sorted(self.masks)},
        }


@dataclass
class ToolResult:
    """Output of a single tool.

    ``changed`` is ``False`` when the tool decided to do nothing (strength 0, or the region
    it needs was not found). Callers should treat that as a normal outcome, not an error.
    """

    image: np.ndarray
    tool: str
    params: dict[str, Any]
    stats: dict[str, Any] = field(default_factory=dict)
    changed: bool = True
    debug: dict[str, np.ndarray] = field(default_factory=dict)


@dataclass
class StageRecord:
    """One executed pipeline stage, for reporting and debug."""

    name: str
    strength: float
    changed: bool
    stats: dict[str, Any]


@dataclass
class RetouchResult:
    """Final output of :meth:`portrait_engine.PortraitEngine.process`.

    ``image`` is the canonical ``float32`` RGB buffer. ``portrait`` carries the same pixels
    together with the source file's metadata, and is what ``PortraitEngine.save`` writes so that
    EXIF and ICC survive the round trip.
    """

    image: np.ndarray  # float32 RGB [0,1]
    analysis: FaceAnalysis
    stages: list[StageRecord] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    debug_dir: str | None = None
    portrait: Any = None  # PortraitImage; typed loosely to keep core.types import-free

    def report(self) -> dict[str, Any]:
        return {
            "stages": [
                {"name": s.name, "strength": s.strength, "changed": s.changed, "stats": s.stats}
                for s in self.stages
            ],
            "diagnostics": self.diagnostics,
            "face": self.analysis.summary(),
        }
