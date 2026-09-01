"""Color management and the signal-processing primitives.

These are the shared foundations. If any of this drifts, every tool drifts with it.
"""

from __future__ import annotations

import numpy as np
import pytest

from portrait_engine.core import ops
from portrait_engine.core.colorspace import (
    lab_to_rgb,
    linear_luma,
    linear_to_srgb,
    rgb_to_lab,
    rgb_to_ycrcb,
    soft_highlight_rolloff,
    srgb_to_linear,
)


@pytest.fixture
def ramp():
    values = np.linspace(0.0, 1.0, 64, dtype=np.float32)
    return np.repeat(values[None, :, None], 8, axis=0).repeat(3, axis=2)


def test_srgb_linear_roundtrip_is_exact(ramp):
    assert np.abs(linear_to_srgb(srgb_to_linear(ramp)) - ramp).max() < 1e-6


def test_srgb_transfer_function_matches_known_values():
    """Spot-check against the standard: mid-grey 0.5 sRGB is about 0.2140 linear."""
    linear = srgb_to_linear(np.array([[[0.0, 0.5, 1.0]]], dtype=np.float32))
    assert linear[0, 0, 0] == pytest.approx(0.0, abs=1e-6)
    assert linear[0, 0, 1] == pytest.approx(0.2140, abs=1e-3)
    assert linear[0, 0, 2] == pytest.approx(1.0, abs=1e-6)


def test_lab_ranges_and_roundtrip(ramp):
    lab = rgb_to_lab(ramp)
    assert lab[..., 0].min() >= 0.0 and lab[..., 0].max() <= 100.0 + 1e-3
    # OpenCV's float32 Lab conversion is not bit-exact. Documented tolerance: well under one
    # 8-bit level, which is why tools convert once and back once rather than repeatedly.
    assert np.abs(lab_to_rgb(lab) - ramp).max() < 1.0 / 255.0


def test_lab_neutral_grey_has_no_chroma():
    lab = rgb_to_lab(np.full((4, 4, 3), 0.5, dtype=np.float32))
    assert np.abs(lab[..., 1]).max() < 0.5
    assert np.abs(lab[..., 2]).max() < 0.5


def test_ycrcb_is_centered(ramp):
    ycrcb = rgb_to_ycrcb(ramp)
    assert ycrcb.min() >= 0.0 and ycrcb.max() <= 1.0 + 1e-3


def test_linear_luma_is_monotonic(ramp):
    luma = linear_luma(ramp)[0]
    assert np.all(np.diff(luma) >= -1e-6)


def test_soft_rolloff_is_monotonic_and_bounded():
    x = np.linspace(0.0, 8.0, 500, dtype=np.float32)
    y = soft_highlight_rolloff(x, knee=0.85)
    assert np.all(np.diff(y) >= -1e-7)
    assert y.max() < 1.0
    below = x <= 0.85
    assert np.allclose(y[below], x[below])


# --------------------------------------------------------------------------- guided filter


@pytest.fixture
def noisy_edge():
    rng = np.random.default_rng(0)
    image = np.zeros((128, 128), dtype=np.float32)
    image[:, 64:] = 1.0
    return image + rng.normal(0, 0.05, image.shape).astype(np.float32)


def test_guided_filter_reduces_noise_and_keeps_the_edge(noisy_edge):
    out = ops.guided_filter(noisy_edge, noisy_edge, radius=8, eps=1e-3)
    assert out[:, :40].std() < noisy_edge[:, :40].std() * 0.85
    step = float(out[:, 70].mean() - out[:, 55].mean())
    assert step > 0.85  # the 1.0 step survives


def test_guided_filter_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="shape mismatch"):
        ops.guided_filter(np.zeros((8, 8), np.float32), np.zeros((8, 9), np.float32), 2, 1e-3)


def test_guided_filter_is_identity_at_tiny_eps(noisy_edge):
    """With eps at zero and the guide equal to the input, the filter must return the input."""
    out = ops.guided_filter(noisy_edge, noisy_edge, radius=4, eps=0.0)
    assert np.abs(out - noisy_edge).max() < 1e-4


def test_guided_filter_subsampling_is_close_to_full(noisy_edge):
    full = ops.guided_filter(noisy_edge, noisy_edge, radius=12, eps=1e-3)
    fast = ops.guided_filter(noisy_edge, noisy_edge, radius=12, eps=1e-3, subsample=3)
    assert np.abs(full - fast).mean() < 0.02


def test_band_decomposition_is_exact(noisy_edge):
    bands = ops.decompose_bands(noisy_edge, noisy_edge, 2, 16, 1e-4, 1e-3)
    assert np.abs(bands.recombine() - noisy_edge).max() < 1e-5


def test_band_recombine_gains_do_what_they_say(noisy_edge):
    bands = ops.decompose_bands(noisy_edge, noisy_edge, 2, 16, 1e-4, 1e-3)
    zeroed = bands.recombine(mid_gain=0.0)
    assert np.abs(zeroed - (bands.base + bands.fine)).max() < 1e-6


def test_normalized_lowpass_ignores_masked_out_content():
    """The whole point: content where the weights are zero must not leak into the result."""
    channel = np.zeros((64, 64), dtype=np.float32)
    channel[28:36, 28:36] = 100.0  # a "blemish"
    weights = np.ones_like(channel)
    weights[24:40, 24:40] = 0.0  # generously excluded

    filled = ops.normalized_lowpass(channel, weights, radius=12)
    assert float(np.abs(filled[28:36, 28:36]).max()) < 1.0


def test_normalized_lowpass_falls_back_when_no_weights():
    channel = np.full((16, 16), 5.0, dtype=np.float32)
    filled = ops.normalized_lowpass(channel, np.zeros_like(channel), radius=3)
    assert np.isfinite(filled).all()


def test_smoothstep_bounds_and_monotonicity():
    x = np.linspace(-1.0, 2.0, 200, dtype=np.float32)
    y = ops.smoothstep(x, 0.0, 1.0)
    assert y.min() == 0.0 and y.max() == 1.0
    assert np.all(np.diff(y) >= -1e-7)
    assert ops.smoothstep(np.array([0.5], np.float32), 0.0, 1.0)[0] == pytest.approx(0.5)


def test_smoothstep_handles_degenerate_edges():
    y = ops.smoothstep(np.array([-1.0, 0.0, 1.0], np.float32), 0.0, 0.0)
    assert y.tolist() == [0.0, 1.0, 1.0]


def test_masked_percentile_returns_nan_when_empty():
    assert np.isnan(ops.masked_percentile(np.zeros((4, 4), np.float32), np.zeros((4, 4), np.float32), 50))


def test_high_frequency_energy_ranks_textures_correctly():
    rng = np.random.default_rng(1)
    smooth = np.zeros((64, 64), dtype=np.float32)
    rough = rng.normal(0, 1, (64, 64)).astype(np.float32)
    assert ops.high_frequency_energy(rough) > ops.high_frequency_energy(smooth)


def test_dilate_and_erode_are_inverse_in_direction():
    mask = np.zeros((64, 64), dtype=np.float32)
    mask[24:40, 24:40] = 1.0
    assert ops.dilate_mask(mask, 3).sum() > mask.sum()
    assert ops.erode_mask(mask, 3).sum() < mask.sum()
    assert np.array_equal(ops.dilate_mask(mask, 0), mask)
