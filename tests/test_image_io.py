"""Image loading, the canonical buffer contract, and metadata handling."""

from __future__ import annotations

import os

import numpy as np
import pytest

from portrait_engine import InvalidImageError, from_array, load_image, save_image
from portrait_engine.core.image import (
    PortraitImage,
    decode_bytes,
    encode_jpeg,
    resize_long_side,
)


def test_load_returns_canonical_buffer(portrait_path):
    image = load_image(portrait_path)
    assert image.rgb.dtype == np.float32
    assert image.rgb.ndim == 3 and image.rgb.shape[2] == 3
    assert float(image.rgb.min()) >= 0.0 and float(image.rgb.max()) <= 1.0
    assert image.rgb.flags["C_CONTIGUOUS"]
    assert image.shape == (image.height, image.width)
    assert image.meta.path == portrait_path


def test_load_channel_order_is_rgb(portrait_path):
    """Guards the single most common bug in this domain.

    OpenCV decodes BGR. If the loader ever forgets to convert, skin turns blue, so comparing
    against an explicit ``cv2.imread`` conversion catches it.
    """
    import cv2

    image = load_image(portrait_path)
    bgr = cv2.imread(portrait_path, cv2.IMREAD_COLOR)
    expected = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    assert np.abs(image.rgb - expected).mean() < 0.02


@pytest.mark.parametrize(
    "array",
    [
        np.zeros((16, 16), dtype=np.uint8),
        np.zeros((16, 16, 4), dtype=np.uint8),
        np.zeros((16, 16, 3), dtype=np.uint16),
        np.zeros((16, 16, 3), dtype=np.float64),
    ],
)
def test_from_array_accepts_common_shapes_and_dtypes(array):
    image = from_array(array)
    assert image.rgb.shape == (16, 16, 3)
    assert image.rgb.dtype == np.float32


def test_from_array_rescales_float_0_255():
    """Floats above 1.5 are interpreted as a 0..255 range, a common caller mistake."""
    image = from_array(np.full((8, 8, 3), 128.0, dtype=np.float32))
    assert 0.45 < float(image.rgb.mean()) < 0.55


def test_from_array_rejects_bad_shapes():
    with pytest.raises(InvalidImageError):
        from_array(np.zeros((16, 16, 5), dtype=np.uint8))
    with pytest.raises(InvalidImageError):
        from_array(np.zeros((16,), dtype=np.uint8))


def test_load_missing_file():
    with pytest.raises(InvalidImageError, match="file not found"):
        load_image("/nonexistent/portrait.jpg")


def test_load_empty_file(tmp_path):
    path = tmp_path / "empty.jpg"
    path.write_bytes(b"")
    with pytest.raises(InvalidImageError, match="empty"):
        load_image(str(path))


def test_load_garbage_file(tmp_path):
    path = tmp_path / "garbage.jpg"
    path.write_bytes(b"this is definitely not a JPEG" * 40)
    with pytest.raises(InvalidImageError):
        load_image(str(path))


def test_load_truncated_file(tmp_path, portrait_path):
    """A file that starts like a JPEG but stops mid-stream must raise, not return junk."""
    with open(portrait_path, "rb") as handle:
        payload = handle.read()
    path = tmp_path / "truncated.jpg"
    path.write_bytes(payload[: len(payload) // 6])
    with pytest.raises(InvalidImageError):
        load_image(str(path))


def test_decode_bytes_rejects_empty():
    with pytest.raises(InvalidImageError):
        decode_bytes(b"")


def test_resize_only_downscales(portrait):
    smaller = resize_long_side(portrait, 200)
    assert max(smaller.shape) == 200
    same = resize_long_side(smaller, 5000)
    assert same.shape == smaller.shape


def test_save_roundtrip_png_is_lossless(tmp_path, portrait):
    path = tmp_path / "out.png"
    save_image(portrait, str(path))
    reloaded = load_image(str(path))
    assert reloaded.shape == portrait.shape
    assert np.abs(reloaded.rgb - portrait.rgb).max() <= 1.5 / 255.0


def test_save_creates_parent_directories(tmp_path, portrait):
    path = tmp_path / "nested" / "deeper" / "out.jpg"
    save_image(portrait, str(path))
    assert os.path.isfile(path)


def test_encode_jpeg_produces_decodable_bytes(portrait):
    payload = encode_jpeg(portrait, quality=90)
    assert payload[:2] == b"\xff\xd8"
    assert decode_bytes(payload).shape == portrait.shape


def test_with_rgb_preserves_metadata(portrait):
    modified = portrait.with_rgb(portrait.rgb * 0.5)
    assert modified.meta == portrait.meta
    assert modified.shape == portrait.shape


def test_portrait_image_rejects_non_rgb():
    with pytest.raises(InvalidImageError):
        PortraitImage(rgb=np.zeros((4, 4), dtype=np.float32))


def test_quantization_is_deterministic(portrait):
    assert np.array_equal(portrait.to_u8_rgb(), portrait.to_u8_rgb())
