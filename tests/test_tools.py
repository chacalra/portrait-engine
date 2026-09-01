"""Per-tool behaviour: the strength contract, the no-op paths, and the quality claims.

The quality tests here check the specific promises in the brief, in the form of measurements:
teeth do not go pure white and keep their inter-tooth shading, skin smoothing keeps texture,
dark circle reduction is a reduction and not a removal, lighting does not flatten, and cleanup
does not erase moles.
"""

from __future__ import annotations

import numpy as np
import pytest

from portrait_engine import (
    StrengthOutOfRangeError,
    balance_face_lighting,
    reduce_dark_circles,
    skin_cleanup,
    skin_smoothing,
    whiten_teeth,
)
from portrait_engine.core.colorspace import rgb_to_lab
from portrait_engine.tools import TOOL_REGISTRY

ALL_TOOLS = (
    skin_cleanup,
    skin_smoothing,
    reduce_dark_circles,
    whiten_teeth,
    balance_face_lighting,
)


# --------------------------------------------------------------------------- strength contract


@pytest.mark.parametrize("tool", ALL_TOOLS, ids=lambda t: t.__name__)
@pytest.mark.parametrize("bad", [-0.01, 1.01, 2.0, float("nan"), float("inf"), "0.5", None])
def test_strength_out_of_range_is_rejected(tool, bad, portrait, analysis):
    with pytest.raises(StrengthOutOfRangeError):
        tool(portrait.rgb, analysis, bad)


@pytest.mark.parametrize("tool", ALL_TOOLS, ids=lambda t: t.__name__)
def test_strength_zero_is_an_exact_no_op(tool, portrait, analysis):
    """Not "nearly identical": byte for byte identical, and reported as ``changed=False``."""
    result = tool(portrait.rgb, analysis, 0.0)
    assert result.changed is False
    assert np.array_equal(result.image, portrait.rgb)
    assert result.stats["edited_area_fraction"] == 0.0


@pytest.mark.parametrize("tool", ALL_TOOLS, ids=lambda t: t.__name__)
def test_shape_and_dtype_are_preserved(tool, portrait, analysis):
    result = tool(portrait.rgb, analysis, 0.5)
    assert result.image.shape == portrait.rgb.shape
    assert result.image.dtype == np.float32
    assert float(result.image.min()) >= 0.0 and float(result.image.max()) <= 1.0


@pytest.mark.parametrize("tool", ALL_TOOLS, ids=lambda t: t.__name__)
def test_tools_do_not_mutate_their_input(tool, portrait, analysis):
    original = portrait.rgb.copy()
    tool(portrait.rgb, analysis, 0.6)
    assert np.array_equal(portrait.rgb, original)


@pytest.mark.parametrize("tool", ALL_TOOLS, ids=lambda t: t.__name__)
def test_tools_are_deterministic(tool, portrait, analysis):
    first = tool(portrait.rgb, analysis, 0.4)
    second = tool(portrait.rgb, analysis, 0.4)
    assert np.array_equal(first.image, second.image)


@pytest.mark.parametrize("tool", ALL_TOOLS, ids=lambda t: t.__name__)
def test_effect_grows_with_strength(tool, portrait, analysis):
    """Monotonicity. A slider that does not respond monotonically is unusable, and a future
    agent could not reason about it."""
    magnitudes = []
    for strength in (0.2, 0.6, 1.0):
        result = tool(portrait.rgb, analysis, strength)
        magnitudes.append(float(np.abs(result.image - portrait.rgb).sum()))
    if magnitudes[-1] == 0.0:
        pytest.skip(f"{tool.__name__} is a no-op on this fixture")
    assert magnitudes[0] <= magnitudes[1] <= magnitudes[2]


@pytest.mark.parametrize("tool", ALL_TOOLS, ids=lambda t: t.__name__)
def test_tools_accept_a_portrait_image_or_a_raw_array(tool, portrait, analysis):
    from_object = tool(portrait, analysis, 0.3)
    from_array_ = tool(portrait.rgb, analysis, 0.3)
    assert np.array_equal(from_object.image, from_array_.image)


# --------------------------------------------------------------------------- teeth whitening


def test_teeth_whitening_reduces_yellow_without_reaching_white(portrait, analysis):
    if analysis.masks.area("teeth") < 1.0:
        pytest.skip("no teeth in this fixture")
    result = whiten_teeth(portrait.rgb, analysis, 1.0)
    stats = result.stats
    assert stats["mean_b_after"] < stats["mean_b_before"], "yellow was not reduced"
    assert stats["mean_L_after"] >= stats["mean_L_before"], "teeth got darker"
    assert stats["max_L_after"] <= 100.0


def test_teeth_whitening_preserves_inter_tooth_shading(portrait, analysis):
    """The "no flat white slab" requirement, as a number. The L* spread inside the teeth must
    survive; an earlier implementation compressed it by 35 percent at full strength."""
    if analysis.masks.area("teeth") < 1.0:
        pytest.skip("no teeth in this fixture")
    stats = whiten_teeth(portrait.rgb, analysis, 1.0).stats
    ratio = stats["inter_tooth_contrast_after"] / max(stats["inter_tooth_contrast_before"], 1e-6)
    assert ratio > 0.70, f"inter-tooth contrast fell to {ratio:.2f}"


def test_teeth_whitening_never_touches_lips_or_gums(portrait, analysis):
    if analysis.masks.area("teeth") < 1.0:
        pytest.skip("no teeth in this fixture")
    result = whiten_teeth(portrait.rgb, analysis, 1.0)
    lips = analysis.masks["lips"] > 0.6
    if lips.sum() > 32:
        assert np.abs(result.image[lips] - portrait.rgb[lips]).max() * 255.0 < 1.0


def test_teeth_whitening_is_a_no_op_without_teeth(portrait, analysis):
    """A closed mouth must produce a reported no-op, never a bleached tongue."""

    from portrait_engine.core.types import FaceAnalysis

    empty = analysis.masks.with_masks({"teeth": np.zeros(analysis.image_shape, np.float32)})
    stripped = FaceAnalysis(
        image_shape=analysis.image_shape, bbox=analysis.bbox, landmarks=analysis.landmarks,
        landmarks_3d=analysis.landmarks_3d, masks=empty,
        skin_probability=analysis.skin_probability, geometry=analysis.geometry,
    )
    result = whiten_teeth(portrait.rgb, stripped, 1.0)
    assert result.changed is False
    assert "teeth" in result.stats["reason"]
    assert np.array_equal(result.image, portrait.rgb)


# --------------------------------------------------------------------------- skin smoothing


def test_skin_smoothing_preserves_texture_at_full_strength(portrait, analysis):
    """The central quality claim: pores survive at any strength, because the fine band is
    reinjected at unity gain rather than being tuned to survive."""
    stats = skin_smoothing(portrait.rgb, analysis, 1.0).stats
    assert stats["texture_retention"] > 0.75, stats["texture_retention"]


def _blotch_energy(rgb: np.ndarray, analysis, mask: np.ndarray) -> float:
    """Mid-scale luminance energy, measured with a **fixed linear** band-pass.

    Deliberately not the tool's own guided-filter decomposition. Re-decomposing an image whose
    mid band was attenuated does not give back the attenuation, because the guided filter is
    data dependent and the energy redistributes between the bands. A linear difference of
    Gaussians is the same operator on both images and therefore a fair before/after comparison.
    """
    import cv2

    lightness = rgb_to_lab(rgb)[..., 0]
    small = 2 * analysis.geometry.scaled(0.012, 1) + 1
    large = 2 * analysis.geometry.scaled(0.130, 3) + 1
    band = cv2.GaussianBlur(lightness, (small, small), 0) - cv2.GaussianBlur(
        lightness, (large, large), 0
    )
    return float(np.abs(band)[mask > 0.5].mean())


def test_skin_smoothing_actually_smooths(portrait, analysis):
    """The opposite failure to a plastic result: an effect so conservative it is invisible.

    Blotch energy in the skin must drop measurably at full strength.
    """
    skin = analysis.masks["skin"]
    result = skin_smoothing(portrait.rgb, analysis, 1.0)
    before = _blotch_energy(portrait.rgb, analysis, skin)
    after = _blotch_energy(result.image, analysis, skin)
    assert after < 0.85 * before, f"blotch energy only went {before:.3f} -> {after:.3f}"


def test_skin_smoothing_response_is_graded(portrait, analysis):
    """Blotch reduction must increase with strength, monotonically."""
    skin = analysis.masks["skin"]
    energies = [
        _blotch_energy(skin_smoothing(portrait.rgb, analysis, s).image, analysis, skin)
        for s in (0.25, 0.5, 1.0)
    ]
    assert energies[0] > energies[1] > energies[2]


def test_skin_smoothing_adapts_its_radius_to_the_skin(portrait, analysis):
    stats = skin_smoothing(portrait.rgb, analysis, 0.5).stats
    assert 0.4 <= stats["roughness"] <= 2.5
    assert stats["radius_base_px"] > stats["radius_fine_px"]


# --------------------------------------------------------------------------- dark circles


def test_dark_circles_reduce_without_removing(portrait, analysis):
    """A residual deficit must remain at full strength: this is a reduction, by construction."""
    result = reduce_dark_circles(portrait.rgb, analysis, 1.0)
    if not result.changed:
        pytest.skip("no usable infraorbital region in this fixture")
    for side in ("left", "right"):
        key = f"{side}_mean_deficit_L"
        if key not in result.stats:
            continue
        deficit = result.stats[key]
        residual = result.stats[f"{side}_residual_deficit_L"]
        if deficit < 1.0:
            continue
        assert residual > 0.15 * deficit, f"{side} under-eye was flattened"
        assert result.stats[f"{side}_recovered_fraction"] <= 0.65


def test_dark_circles_preserve_texture(portrait, analysis):
    """Correcting only the low-frequency deficit is what keeps creases and the eye bag edge."""
    result = reduce_dark_circles(portrait.rgb, analysis, 1.0)
    if not result.changed:
        pytest.skip("no usable infraorbital region")
    assert result.stats["texture_retention"] > 0.9


def test_dark_circles_only_brighten(portrait, analysis):
    """The tool must never darken the under-eye region: that would deepen the circles."""
    result = reduce_dark_circles(portrait.rgb, analysis, 1.0)
    if not result.changed:
        pytest.skip("no usable infraorbital region")
    region = analysis.masks["under_eye"] > 0.5
    if region.sum() < 32:
        pytest.skip("region too small")
    before = rgb_to_lab(portrait.rgb)[..., 0][region]
    after = rgb_to_lab(result.image)[..., 0][region]
    assert float((after - before).mean()) > -0.05


def test_dark_circles_mask_reaches_full_strength(analysis):
    """The infraorbital alpha must actually reach 1.0 somewhere.

    Regression test for the defect that made this tool nearly inert. Two mask decisions each
    scaled the correction down invisibly: a feather radius expressed as a fraction of the
    interocular distance (about half the region's own height, which smoothed the alpha to a
    peak of 0.40) and a soft `skin_probability` gate that penalised the dark circle for being
    dark. Neither produced an error, a warning or a bad stat. They just quietly capped every
    correction at a fraction of its nominal value.
    """
    checked = 0
    for side in ("left", "right"):
        region = analysis.masks[f"{side}_under_eye"]
        if region.sum() < 32:
            continue
        checked += 1
        # Per side, not on the union: the union takes the max of the two, so one healthy eye
        # hides a collapsed one. With the previous guided feather this fixture measured 0.83
        # on the left and 1.00 on the right, and the union read 1.00.
        assert region.max() > 0.95, (
            f"{side} under-eye alpha peaks at {region.max():.2f}, so every correction there "
            "is silently scaled by that factor"
        )
    if checked == 0:
        pytest.skip("no usable infraorbital region in this fixture")


def test_dark_circles_deliver_their_stated_ceiling(portrait_paths, analyzer):
    """Where there is a real deficit, the delivered recovery must land near the stated ceiling.

    Regression test for a squared alpha: the tool multiplied the lift by the soft region and
    then composited with that same region, so the image received `lift * alpha**2`. On a
    portrait with a 17 L* deficit that delivered 2.6 percent of it while the stats, which read
    the pre-compositing field, reported 47 percent.

    Asserting against the *output image* is the point. A test that compares the stats to the
    image proves nothing now that the stats are themselves measured from the image, which is
    the correct design but makes that comparison tautological. The band is two-sided: too low
    means the correction is being scaled away somewhere, too high means removal rather than
    reduction.
    """
    from portrait_engine import load_image

    checked = 0
    for path in portrait_paths:
        image = load_image(path, max_side=900)
        try:
            analysis = analyzer.analyze(image)
        except Exception:
            continue
        result = reduce_dark_circles(image.rgb, analysis, 1.0)
        if not result.changed:
            continue
        for side in ("left", "right"):
            key = f"{side}_mean_deficit_L"
            if key not in result.stats or result.stats[key] < 4.0:
                continue
            # A region already near the top of the sRGB gamut is held back by the headroom
            # rail on purpose. That is the tool being correct, not weak, so it is excluded
            # here and covered by `test_dark_circles_stay_inside_the_gamut` instead.
            if result.stats.get(f"{side}_headroom_limited_fraction", 0.0) > 0.25:
                continue
            checked += 1
            fraction = result.stats[f"{side}_recovered_fraction"]
            assert 0.25 <= fraction <= 0.65, (
                f"{path}: {side} recovered {fraction:.3f} of a "
                f"{result.stats[key]:.1f} L* deficit, outside the usable band"
            )
    if checked == 0:
        pytest.skip("no fixture has a deficit deep enough to measure against the ceiling")


def test_dark_circles_alpha_carries_no_texture(portrait, analysis):
    """A lift that provably contains no texture must not change texture.

    The under-eye alpha used to be built with `feather_mask`, whose guided filter makes the
    alpha follow image edges. Compositing then imprints a copy of the skin texture scaled by
    the lift. The bound is **two-sided** because the sign flips with the subject: measured on a
    texture-free lift, guided feathering scored 0.90 on one portrait and 1.10 on another, while
    a plain blur stayed within half a percent of 1.0 on both. A one-sided test would have
    passed on half the cases for the wrong reason.
    """
    from portrait_engine.core import ops
    from portrait_engine.core.colorspace import lab_to_rgb, srgb_to_linear
    from portrait_engine.core.masking import apply_masked

    alpha = analysis.masks["under_eye"]
    if alpha.sum() < 32:
        pytest.skip("no usable infraorbital region")
    lab = rgb_to_lab(portrait.rgb)
    lightness = lab[..., 0]

    # The control lift has to be large enough to separate the two behaviours, but must not
    # clip: clipping destroys texture on its own and would confound this measurement. Size it
    # against the region's actual headroom (see `test_dark_circles_stay_inside_the_gamut`).
    selection = alpha > 0.5
    if selection.sum() < 32:
        pytest.skip("region too small")
    scale = 1.0 / np.maximum(srgb_to_linear(portrait.rgb).max(axis=-1), 1e-4)
    headroom = (lightness + 16.0) * (np.cbrt(scale) - 1.0)
    lift = min(9.0, 0.8 * float(np.percentile(headroom[selection], 5.0)))
    if lift < 4.0:
        pytest.skip("not enough gamut headroom to run a discriminating control")

    flat = lab_to_rgb(np.stack([lightness + lift, lab[..., 1], lab[..., 2]], axis=-1))
    out = apply_masked(portrait.rgb, flat, alpha, strict=True, name="control")
    ratio = ops.high_frequency_energy(rgb_to_lab(out)[..., 0], alpha) / ops.high_frequency_energy(
        lightness, alpha
    )
    assert 0.95 < ratio < 1.05, (
        f"a texture-free {lift:.1f} L* lift changed texture energy to {ratio:.3f} of the "
        "original: the alpha itself is carrying image structure"
    )


def test_dark_circles_stay_inside_the_gamut(portrait_paths, analyzer):
    """Brightening must not push skin out of the sRGB gamut.

    On fair skin under warm light the red channel under the eye is often already near 1.0. One
    fixture had 20 percent of the region above 0.98, where a 9 L* lift clipped 99 percent of it
    into a flat patch, measured at 0.76 high-frequency retention. Clipping is the exact plastic
    look the brief forbids, so the lift is limited by the remaining headroom.

    Runs over every fixture, not just one: only a subject who is already near the top of the
    gamut can exercise this at all.
    """
    from portrait_engine import load_image

    checked = 0
    for path in portrait_paths:
        image = load_image(path, max_side=900)
        try:
            analysis = analyzer.analyze(image)
        except Exception:
            continue
        result = reduce_dark_circles(image.rgb, analysis, 1.0)
        if not result.changed:
            continue
        region = analysis.masks["under_eye"] > 0.5
        if region.sum() < 32:
            continue
        checked += 1
        before = float((image.rgb[region].max(axis=-1) >= 0.999).mean())
        after = float((result.image[region].max(axis=-1) >= 0.999).mean())
        assert after <= before + 0.01, (
            f"{path}: clipped pixels went from {before:.1%} to {after:.1%} of the under-eye "
            "region"
        )
    if checked == 0:
        pytest.skip("no fixture has a usable infraorbital region")


# --------------------------------------------------------------------------- lighting


def test_lighting_reduces_illumination_spread(portrait, analysis):
    stats = balance_face_lighting(portrait.rgb, analysis, 1.0).stats
    assert stats["illumination_spread_after_stops"] < stats["illumination_spread_before_stops"]


def test_lighting_respects_its_gain_clamp(portrait, analysis):
    from portrait_engine.core.config import DEFAULT_CONFIG

    config = DEFAULT_CONFIG.face_lighting
    stats = balance_face_lighting(portrait.rgb, analysis, 1.0).stats
    assert stats["gain_min_used"] >= config.gain_min - 1e-4
    assert stats["gain_max_used"] <= config.gain_max + 1e-4


def test_lighting_does_not_flatten_or_clip(portrait, analysis):
    """The "no HDR, no grey skin" requirement, as two numbers."""
    stats = balance_face_lighting(portrait.rgb, analysis, 1.0).stats
    assert stats["local_contrast_retention"] > 0.85
    assert stats["clipped_pixels"] == 0


# --------------------------------------------------------------------------- skin cleanup


def test_skin_cleanup_respects_its_area_budget(portrait, analysis):
    from portrait_engine.core.config import DEFAULT_CONFIG

    result = skin_cleanup(portrait.rgb, analysis, 1.0)
    if not result.changed:
        pytest.skip("no blemishes in this fixture")
    budget = DEFAULT_CONFIG.skin_cleanup.max_area_fraction
    assert result.stats["blemish_fraction_of_skin"] <= budget + 1e-6


def test_skin_cleanup_preserves_overall_skin_texture(portrait, analysis):
    result = skin_cleanup(portrait.rgb, analysis, 1.0)
    if not result.changed:
        pytest.skip("no blemishes in this fixture")
    assert result.stats["texture_retention"] > 0.9


def _inject_spots(portrait, analysis, blemish_depth: float = 10.0, seed: int = 7):
    """Paint synthetic blemishes and moles onto real skin.

    Blemishes: 10 L* darker and slightly red, which is a typical spot. Moles: 26 L* darker and
    brown, well past the 17 L* threshold at which the tool treats a blob as permanent.
    """
    import cv2

    from portrait_engine.core.colorspace import lab_to_rgb
    from portrait_engine.core.image import PortraitImage

    interocular = analysis.geometry.interocular_px
    lab = rgb_to_lab(portrait.rgb).copy()
    ys, xs = np.where(analysis.masks["skin"] > 0.8)
    if len(ys) < 500:
        return None, []

    rng = np.random.default_rng(seed)
    picks = rng.choice(len(ys), 12, replace=False)
    spots: list[tuple[str, np.ndarray]] = []
    for index, pick in enumerate(picks):
        cy, cx = int(ys[pick]), int(xs[pick])
        radius = max(2, int(round(interocular * rng.uniform(0.010, 0.026) / 2)))
        blob = np.zeros(portrait.shape, dtype=np.float32)
        cv2.circle(blob, (cx, cy), radius, 1.0, -1, cv2.LINE_AA)
        blob = cv2.GaussianBlur(blob, (2 * radius + 1, 2 * radius + 1), 0)
        if index % 2 == 0:
            lab[..., 0] -= blemish_depth * blob
            lab[..., 1] += 4.0 * blob
            spots.append(("blemish", blob > 0.35))
        else:
            lab[..., 0] -= 26.0 * blob
            lab[..., 2] += 6.0 * blob
            spots.append(("mole", blob > 0.35))
    return PortraitImage(rgb=lab_to_rgb(lab), meta=portrait.meta), spots


@pytest.mark.slow
def test_skin_cleanup_keeps_moles_and_finds_spots(portrait_paths, analyzer):
    """Injects synthetic blemishes and synthetic moles into real skin, across every fixture.

    Two assertions, and both halves matter. A tool that removes everything would satisfy recall
    and fail mole preservation, and erasing moles is explicitly a failure in the brief.

    * **Moles must survive on every fixture.** Aggregated over all of them, at most a few
      percent of injected mole area may be touched at all.
    * **Recall varies with the skin, so it is asserted as a best case.** On smooth skin the tool
      finds every injected blemish; on heavily freckled or very dark low-contrast skin it finds
      few or none, because a mild spot genuinely is not unusual for that skin. The tool is
      normalized by local texture, so this is a property of the method, not a bug. Requiring a
      floor on every fixture would encode an accuracy the method does not have. See README.md.
    """
    from portrait_engine import load_image
    from portrait_engine.core.types import NoFaceDetectedError

    mole_area_total = 0.0
    mole_area_touched = 0.0
    best_recall = 0
    fixtures_checked = 0

    for path in portrait_paths:
        try:
            portrait = load_image(path, max_side=1400)
            analysis = analyzer.analyze(portrait)
        except NoFaceDetectedError:
            continue
        dirty, spots = _inject_spots(portrait, analysis)
        if dirty is None:
            continue
        fixtures_checked += 1

        result = skin_cleanup(dirty, analyzer.analyze(dirty), 1.0, debug=True)
        detected = (
            result.debug["blemish_mask"]
            if result.changed
            else np.zeros(portrait.shape, dtype=np.float32)
        )
        for kind, area in spots:
            if kind == "mole":
                mole_area_total += float(area.sum())
                mole_area_touched += float((detected[area] > 0.5).sum())
        recall = sum(
            1 for kind, area in spots if kind == "blemish" and (detected[area] > 0.5).any()
        )
        best_recall = max(best_recall, recall)

    assert fixtures_checked >= 2, "not enough usable fixtures"
    touched_fraction = mole_area_touched / max(mole_area_total, 1.0)
    assert touched_fraction < 0.05, f"{touched_fraction:.1%} of mole area was treated as blemish"
    assert best_recall >= 4, f"best blemish recall across fixtures was only {best_recall}/6"


# --------------------------------------------------------------------------- registry


def test_registry_contains_every_tool():
    assert set(TOOL_REGISTRY) == {
        "skin_cleanup", "skin_smoothing", "reduce_dark_circles",
        "whiten_teeth", "balance_face_lighting",
    }
