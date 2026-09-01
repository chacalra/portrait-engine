"""Perception: detection policy, mask contract, and the derived regions."""

from __future__ import annotations

import numpy as np
import pytest

from portrait_engine import (
    FaceAnalyzer,
    MaskContractError,
    MultipleFacesError,
    NoFaceDetectedError,
)
from portrait_engine.core.types import MaskSet
from portrait_engine.perception import landmark_sets as ls
from portrait_engine.perception.face_analyzer import AnalyzerConfig

REQUIRED_MASKS = (
    "skin", "face_oval", "protected",
    "left_eye", "right_eye", "eyes",
    "left_eyebrow", "right_eyebrow", "eyebrows",
    "lips", "mouth_interior", "teeth",
    "nose", "nostrils", "hair_like",
    "left_under_eye", "right_under_eye", "under_eye",
    "left_under_eye_reference", "right_under_eye_reference",
    "face_photometric",
)


# --------------------------------------------------------------------------- landmark sets


def test_landmark_indices_are_in_range():
    for name, ring in ls.ALL_RINGS.items():
        assert max(ring) < ls.LANDMARK_COUNT, name
        assert min(ring) >= 0, name


def test_landmark_rings_have_no_duplicates():
    for name, ring in ls.ALL_RINGS.items():
        assert len(set(ring)) == len(ring), f"{name} repeats an index"


def test_lip_rings_are_disjoint():
    assert not set(ls.LIPS_OUTER) & set(ls.LIPS_INNER)


def test_skin_seed_points_avoid_features():
    """Seeds must be skin. If one landed on an eye or a lip, the skin color model would be
    poisoned and every skin mask would be wrong."""
    feature_indices = set()
    for key in ("left_eye", "right_eye", "lips_outer", "lips_inner", "left_eyebrow", "right_eyebrow"):
        feature_indices |= set(ls.ALL_RINGS[key])
    assert not set(ls.SKIN_SEED_POINTS) & feature_indices


# --------------------------------------------------------------------------- detection policy


def test_analysis_shapes_and_geometry(analysis, portrait):
    assert analysis.image_shape == portrait.shape
    assert analysis.landmarks.shape == (ls.LANDMARK_COUNT, 2)
    assert analysis.landmarks_3d.shape == (ls.LANDMARK_COUNT, 3)
    assert analysis.geometry.interocular_px > 10
    assert 0 <= analysis.bbox.x < analysis.bbox.x2 <= portrait.width
    assert 0 <= analysis.bbox.y < analysis.bbox.y2 <= portrait.height


def test_landmarks_are_inside_the_frame(analysis, portrait):
    inside = (
        (analysis.landmarks[:, 0] >= -2)
        & (analysis.landmarks[:, 0] <= portrait.width + 2)
        & (analysis.landmarks[:, 1] >= -2)
        & (analysis.landmarks[:, 1] <= portrait.height + 2)
    )
    assert inside.mean() > 0.98


def test_analysis_is_immutable(analysis):
    with pytest.raises(ValueError):
        analysis.landmarks[0, 0] = 0.0
    with pytest.raises(ValueError):
        analysis.masks["skin"][0, 0] = 0.5


def test_no_face_raises_controlled_error(analyzer, synthetic_no_face):
    with pytest.raises(NoFaceDetectedError):
        analyzer.analyze(synthetic_no_face)


def test_multiple_faces_raise_in_the_mvp(analyzer, two_face_image):
    with pytest.raises(MultipleFacesError, match="crop to a single subject"):
        analyzer.analyze(two_face_image)


def test_multiple_faces_can_be_allowed_explicitly(two_face_image):
    permissive = FaceAnalyzer(AnalyzerConfig(allow_multiple_faces=True))
    result = permissive.analyze(two_face_image)
    assert result.landmarks.shape[0] == ls.LANDMARK_COUNT


def test_tiny_face_is_rejected(analyzer, portrait):
    """A face too small for reliable masks must produce a clear error, not bad masks."""
    from portrait_engine import from_array

    padded = np.full((portrait.height * 9, portrait.width * 9, 3), 0.5, dtype=np.float32)
    padded[: portrait.height, : portrait.width] = portrait.rgb
    with pytest.raises(NoFaceDetectedError):
        analyzer.analyze(from_array(padded))


def test_detection_is_deterministic(analyzer, portrait):
    first = analyzer.analyze(portrait)
    second = analyzer.analyze(portrait)
    assert np.array_equal(first.landmarks, second.landmarks)
    assert np.array_equal(first.masks["skin"], second.masks["skin"])


# --------------------------------------------------------------------------- mask contract


def test_all_required_masks_exist(analysis):
    missing = [name for name in REQUIRED_MASKS if name not in analysis.masks]
    assert not missing, f"missing masks: {missing}"


def test_masks_obey_the_contract(analysis, portrait):
    for name in analysis.masks:
        mask = analysis.masks[name]
        assert mask.shape == portrait.shape, name
        assert mask.dtype == np.float32, name
        assert float(mask.min()) >= 0.0, name
        assert float(mask.max()) <= 1.0, name
        assert np.isfinite(mask).all(), name


def test_skin_probability_obeys_the_contract(analysis, portrait):
    prob = analysis.skin_probability
    assert prob.shape == portrait.shape
    assert prob.dtype == np.float32
    assert float(prob.min()) >= 0.0 and float(prob.max()) <= 1.0


def test_skin_mask_is_substantial(analysis):
    skin = analysis.masks["skin"]
    assert skin.sum() > 0.1 * analysis.masks["face_oval"].sum(), "skin mask is implausibly small"


def test_skin_mask_is_contained_near_the_face_oval(analysis):
    """The mask is feathered, so some alpha immediately outside the oval polygon is correct and
    wanted. What must not happen is the mask reaching well past the face, onto the neck or the
    background, where smoothing would soften a jaw edge.

    The bound is therefore a *margin*: past a modest dilation of the oval, the mask must be
    essentially zero.
    """
    from portrait_engine.core import ops

    skin = analysis.masks["skin"]
    margin = ops.dilate_mask(
        analysis.masks["face_oval"], analysis.geometry.scaled(0.07, 2)
    )
    escaped = skin * (1.0 - margin)
    assert float(escaped.max()) < 0.05, "skin mask reaches well beyond the face oval"
    assert float(escaped.sum()) < 0.002 * float(skin.sum())


def test_skin_mask_excludes_protected_regions(analysis):
    """The core guardrail of the perception layer: skin never overlaps eyes, brows or lips."""
    skin = analysis.masks["skin"]
    for name in ("eyes", "eyebrows", "lips", "mouth_interior"):
        overlap = float((skin * analysis.masks[name]).max())
        assert overlap < 0.2, f"skin overlaps {name} at alpha {overlap:.2f}"


def test_teeth_stay_inside_the_mouth_opening(analysis):
    teeth = analysis.masks["teeth"]
    if teeth.sum() < 1.0:
        pytest.skip("no teeth detected in this fixture")
    outside = teeth * (1.0 - analysis.masks["mouth_interior"])
    assert float(outside.max()) < 0.05


def test_teeth_do_not_touch_the_lips(analysis):
    teeth = analysis.masks["teeth"]
    if teeth.sum() < 1.0:
        pytest.skip("no teeth detected in this fixture")
    assert float((teeth * analysis.masks["lips"]).max()) < 0.05


def test_under_eye_regions_are_below_the_eyes(analysis):
    """A sanity check on the derived infraorbital contour: its centroid must sit lower in the
    face than the eye it belongs to."""
    for side, eye in (("left", "left_eye"), ("right", "right_eye")):
        region = analysis.masks[f"{side}_under_eye"]
        if region.sum() < 32:
            continue
        ys = np.arange(region.shape[0], dtype=np.float32)[:, None]
        region_y = float((region * ys).sum() / region.sum())
        eye_mask = analysis.masks[eye]
        eye_y = float((eye_mask * ys).sum() / max(eye_mask.sum(), 1.0))
        assert region_y > eye_y, f"{side} under-eye region is not below the {side} eye"


def test_under_eye_does_not_overlap_the_eye(analysis):
    for side in ("left", "right"):
        overlap = float((analysis.masks[f"{side}_under_eye"] * analysis.masks["eyes"]).max())
        assert overlap < 0.15


def test_under_eye_reference_is_below_the_under_eye_region(analysis):
    ys = np.arange(analysis.image_shape[0], dtype=np.float32)[:, None]
    for side in ("left", "right"):
        region = analysis.masks[f"{side}_under_eye"]
        reference = analysis.masks[f"{side}_under_eye_reference"]
        if region.sum() < 32 or reference.sum() < 32:
            continue
        assert float((reference * ys).sum() / reference.sum()) > float(
            (region * ys).sum() / region.sum()
        )


def test_mask_set_rejects_contract_violations():
    shape = (8, 8)
    with pytest.raises(MaskContractError, match="shape"):
        MaskSet({"bad": np.zeros((4, 4), np.float32)}, shape)
    with pytest.raises(MaskContractError, match=r"out of \[0,1\]"):
        MaskSet({"bad": np.full(shape, 2.0, np.float32)}, shape)
    with pytest.raises(MaskContractError, match="non-finite"):
        MaskSet({"bad": np.full(shape, np.nan, np.float32)}, shape)


def test_mask_set_gives_a_helpful_message_for_unknown_names(analysis):
    with pytest.raises(MaskContractError, match="not available"):
        analysis.masks["definitely_not_a_mask"]


@pytest.mark.slow
def test_masks_hold_across_every_fixture(analyzer, portrait_paths):
    """The contract must hold on every available photo, not just the default one."""
    from portrait_engine import load_image

    checked = 0
    for path in portrait_paths:
        try:
            result = analyzer.analyze(load_image(path, max_side=900))
        except NoFaceDetectedError:
            continue
        checked += 1
        for name in REQUIRED_MASKS:
            mask = result.masks[name]
            assert mask.shape == result.image_shape
            assert float(mask.min()) >= 0.0 and float(mask.max()) <= 1.0
        assert result.masks.area("skin") > 100
    assert checked >= 2, "not enough usable fixtures to validate the contract"
