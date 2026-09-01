"""Image IO and the canonical in-memory representation.

Canonical buffer: ``float32``, shape ``(H, W, 3)``, channel order **RGB**, sRGB-encoded,
range ``[0, 1]``.

Rationale (see ``docs/ARCHITECTURE.md`` section 3):

* RGB internally, BGR only inside this module, so the RGB/BGR mixup has exactly one place
  it could happen.
* ``float32`` end to end. Five tools chained over ``uint8`` accumulate quantization error
  that is visible as banding in skin gradients.
* Metadata (EXIF, ICC, original dtype) travels beside the array, never inside it.
"""

from __future__ import annotations

import io
import os
from dataclasses import dataclass, field, replace
from typing import Any

import cv2
import numpy as np

from portrait_engine.core.types import InvalidImageError

try:  # Pillow is used only for metadata-preserving IO. Optional by design.
    from PIL import Image as PILImage
    from PIL import ImageOps as PILImageOps

    _HAS_PIL = True
except ImportError:  # pragma: no cover
    _HAS_PIL = False

MAX_PIXELS = 80_000_000  # ~80 MP. Guards against decompression bombs.


@dataclass(frozen=True)
class ImageMeta:
    """Everything about the file that is not pixels."""

    path: str | None = None
    original_dtype: str = "uint8"
    original_size: tuple[int, int] | None = None  # (width, height) before any downscale
    exif: bytes | None = None
    icc_profile: bytes | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PortraitImage:
    """The canonical buffer plus its metadata."""

    rgb: np.ndarray
    meta: ImageMeta = field(default_factory=ImageMeta)

    def __post_init__(self) -> None:
        arr = self.rgb
        if not isinstance(arr, np.ndarray) or arr.ndim != 3 or arr.shape[2] != 3:
            raise InvalidImageError(
                f"expected an (H, W, 3) RGB array, got {getattr(arr, 'shape', type(arr))}"
            )
        object.__setattr__(self, "rgb", np.ascontiguousarray(arr, dtype=np.float32))

    @property
    def shape(self) -> tuple[int, int]:
        """``(height, width)``."""
        return (self.rgb.shape[0], self.rgb.shape[1])

    @property
    def height(self) -> int:
        return self.rgb.shape[0]

    @property
    def width(self) -> int:
        return self.rgb.shape[1]

    def with_rgb(self, rgb: np.ndarray) -> PortraitImage:
        """New image, same metadata. Used between pipeline stages."""
        return replace(self, rgb=np.ascontiguousarray(rgb, dtype=np.float32))

    def to_u8_rgb(self) -> np.ndarray:
        return _to_u8(self.rgb)

    def to_u8_bgr(self) -> np.ndarray:
        return cv2.cvtColor(_to_u8(self.rgb), cv2.COLOR_RGB2BGR)


def _to_u8(rgb: np.ndarray) -> np.ndarray:
    """Round-half-up quantization to ``uint8``. No dithering: it would break reproducibility
    tests for no visible benefit at 8 bits over smooth skin."""
    return np.clip(np.rint(rgb * 255.0), 0, 255).astype(np.uint8)


def from_array(array: np.ndarray, meta: ImageMeta | None = None) -> PortraitImage:
    """Wrap a NumPy array. Accepts ``uint8``/``uint16`` RGB, or ``float`` RGB in ``[0,1]``.

    Grayscale and RGBA inputs are converted. Alpha is dropped (portraits, not compositing).
    """
    arr = np.asarray(array)
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    if arr.ndim != 3:
        raise InvalidImageError(f"unsupported array shape {arr.shape}")
    if arr.shape[2] == 4:
        arr = arr[:, :, :3]
    if arr.shape[2] != 3:
        raise InvalidImageError(f"unsupported channel count {arr.shape[2]}")

    dtype = arr.dtype
    if dtype == np.uint8:
        rgb = arr.astype(np.float32) / 255.0
    elif dtype == np.uint16:
        rgb = arr.astype(np.float32) / 65535.0
    elif np.issubdtype(dtype, np.floating):
        rgb = arr.astype(np.float32)
        if float(np.nanmax(rgb)) > 1.5:  # someone handed us 0..255 floats
            rgb = rgb / 255.0
        rgb = np.clip(rgb, 0.0, 1.0)
    else:
        raise InvalidImageError(f"unsupported dtype {dtype}")

    base = meta or ImageMeta()
    return PortraitImage(
        rgb=rgb,
        meta=replace(
            base,
            original_dtype=str(dtype),
            original_size=base.original_size or (arr.shape[1], arr.shape[0]),
        ),
    )


def load_image(path: str | os.PathLike[str], max_side: int | None = None) -> PortraitImage:
    """Decode a file into the canonical buffer.

    Applies EXIF orientation, keeps EXIF and ICC bytes for later re-injection, and optionally
    downscales so the long side is at most ``max_side``.
    """
    path = os.fspath(path)
    if not os.path.isfile(path):
        raise InvalidImageError(f"file not found: {path}")
    if os.path.getsize(path) == 0:
        raise InvalidImageError(f"file is empty: {path}")

    exif: bytes | None = None
    icc: bytes | None = None
    array: np.ndarray | None = None

    if _HAS_PIL:
        try:
            with PILImage.open(path) as handle:
                handle.load()
                exif = handle.info.get("exif")
                icc = handle.info.get("icc_profile")
                oriented = PILImageOps.exif_transpose(handle)
                if oriented.mode not in ("RGB", "L", "RGBA", "I;16"):
                    oriented = oriented.convert("RGB")
                array = np.array(oriented)
        except Exception as exc:
            raise InvalidImageError(f"cannot decode {path}: {exc}") from exc
    else:  # pragma: no cover
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise InvalidImageError(f"cannot decode {path}")
        array = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    if array is None or array.size == 0:
        raise InvalidImageError(f"cannot decode {path}")
    if array.shape[0] * array.shape[1] > MAX_PIXELS:
        raise InvalidImageError(
            f"{path} has {array.shape[0] * array.shape[1]} pixels, over the {MAX_PIXELS} limit"
        )

    image = from_array(
        array,
        ImageMeta(path=path, exif=exif, icc_profile=icc, original_size=(array.shape[1], array.shape[0])),
    )
    if max_side is not None:
        image = resize_long_side(image, max_side)
    return image


def resize_long_side(image: PortraitImage, max_side: int) -> PortraitImage:
    """Downscale only. Upscaling a portrait before retouching invents detail."""
    h, w = image.shape
    longest = max(h, w)
    if longest <= max_side:
        return image
    scale = max_side / float(longest)
    new_size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
    resized = cv2.resize(image.rgb, new_size, interpolation=cv2.INTER_AREA)
    return image.with_rgb(resized)


def save_image(
    image: PortraitImage,
    path: str | os.PathLike[str],
    quality: int = 95,
    keep_metadata: bool = True,
) -> str:
    """Encode the canonical buffer to disk, re-injecting EXIF and ICC when available."""
    path = os.fspath(path)
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    rgb_u8 = image.to_u8_rgb()

    if _HAS_PIL:
        pil = PILImage.fromarray(rgb_u8, mode="RGB")
        params: dict[str, Any] = {}
        ext = os.path.splitext(path)[1].lower()
        if ext in (".jpg", ".jpeg"):
            params.update(quality=quality, subsampling=0, optimize=True)
        if keep_metadata:
            if image.meta.exif:
                params["exif"] = image.meta.exif
            if image.meta.icc_profile:
                params["icc_profile"] = image.meta.icc_profile
        pil.save(path, **params)
    else:  # pragma: no cover
        cv2.imwrite(path, cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR))
    return path


def encode_jpeg(image: PortraitImage, quality: int = 95) -> bytes:
    """In-memory JPEG, used by the FastAPI adapter."""
    if _HAS_PIL:
        buffer = io.BytesIO()
        PILImage.fromarray(image.to_u8_rgb(), mode="RGB").save(
            buffer, format="JPEG", quality=quality, subsampling=0
        )
        return buffer.getvalue()
    ok, encoded = cv2.imencode(".jpg", image.to_u8_bgr(), [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:  # pragma: no cover
        raise InvalidImageError("JPEG encoding failed")
    return encoded.tobytes()


def decode_bytes(payload: bytes) -> PortraitImage:
    """Decode an uploaded file. Used by the FastAPI adapter."""
    if not payload:
        raise InvalidImageError("empty payload")
    array = np.frombuffer(payload, dtype=np.uint8)
    bgr = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if bgr is None:
        raise InvalidImageError("payload is not a decodable image")
    return from_array(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
