"""Pipeline behaviour: reproducibility, ordering, debug output, and error handling."""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from portrait_engine import (
    MultipleFacesError,
    NoFaceDetectedError,
    PortraitEngine,
    PortraitEngineError,
    StrengthOutOfRangeError,
    load_image,
)
from portrait_engine.core.config import DEFAULT_ORDER

NATURAL = {
    "skin_cleanup": 0.15,
    "skin_smoothing": 0.20,
    "dark_circles": 0.15,
    "teeth_whitening": 0.10,
    "face_lighting": 0.10,
}


def test_pipeline_is_bit_reproducible(engine, portrait):
    """Two runs of the same input must be byte-identical. Nothing in the engine may be random
    or order-dependent, because a retouching decision has to be repeatable."""
    first = engine.process(portrait, **NATURAL)
    second = engine.process(portrait, **NATURAL)
    assert np.array_equal(first.image, second.image)


def test_pipeline_is_reproducible_across_engine_instances(portrait):
    a = PortraitEngine().process(portrait, **NATURAL)
    b = PortraitEngine().process(portrait, **NATURAL)
    assert np.array_equal(a.image, b.image)


def test_all_strengths_zero_returns_the_input_unchanged(engine, portrait):
    result = engine.process(portrait, skin_smoothing=0.0, teeth_whitening=0.0)
    assert np.array_equal(result.image, portrait.rgb)
    assert result.stages == []


def test_pipeline_preserves_size_and_dtype(engine, portrait):
    result = engine.process(portrait, **NATURAL)
    assert result.image.shape == portrait.rgb.shape
    assert result.image.dtype == np.float32
    assert float(result.image.min()) >= 0.0 and float(result.image.max()) <= 1.0


def test_perception_runs_once_and_is_reusable(engine, portrait, analysis):
    """Passing an analysis in must give the same result as letting the engine compute it."""
    with_analysis = engine.process(portrait, analysis=analysis, **NATURAL)
    without = engine.process(portrait, **NATURAL)
    assert np.array_equal(with_analysis.image, without.image)


def test_stage_order_follows_the_configured_order(engine, portrait):
    result = engine.process(portrait, **NATURAL)
    names = [stage.name for stage in result.stages]
    positions = [DEFAULT_ORDER.index(name) for name in names]
    assert positions == sorted(positions)


def test_order_can_be_overridden(engine, portrait):
    reversed_order = tuple(reversed(DEFAULT_ORDER))
    result = engine.process(portrait, order=reversed_order, **NATURAL)
    assert [s.name for s in result.stages] == [
        name for name in reversed_order if name in {s.name for s in result.stages}
    ]


def test_order_matters(engine, portrait):
    """Not a correctness requirement, but worth pinning: the tools are not commutative, which is
    why the default order is documented and configurable rather than incidental."""
    forward = engine.process(portrait, **NATURAL)
    backward = engine.process(portrait, order=tuple(reversed(DEFAULT_ORDER)), **NATURAL)
    assert not np.array_equal(forward.image, backward.image)


def test_tool_aliases_and_canonical_names_agree(engine, portrait):
    short = engine.process(portrait, teeth_whitening=0.3, dark_circles=0.2)
    long = engine.process(portrait, whiten_teeth=0.3, reduce_dark_circles=0.2)
    assert np.array_equal(short.image, long.image)


def test_unknown_tool_name_is_rejected(engine, portrait):
    with pytest.raises(PortraitEngineError, match="unknown tool"):
        engine.process(portrait, make_me_beautiful=0.5)


def test_invalid_strength_is_rejected(engine, portrait):
    with pytest.raises(StrengthOutOfRangeError):
        engine.process(portrait, skin_smoothing=1.4)


def test_no_face_propagates_a_controlled_error(engine, synthetic_no_face):
    with pytest.raises(NoFaceDetectedError):
        engine.process(synthetic_no_face, skin_smoothing=0.2)


def test_multiple_faces_propagate_a_controlled_error(engine, two_face_image):
    with pytest.raises(MultipleFacesError):
        engine.process(two_face_image, skin_smoothing=0.2)


def test_invalid_file_propagates_a_controlled_error(engine, tmp_path):
    from portrait_engine import InvalidImageError

    path = tmp_path / "broken.jpg"
    path.write_bytes(b"nope" * 100)
    with pytest.raises(InvalidImageError):
        engine.process(str(path), skin_smoothing=0.2)


def test_report_is_json_serializable(engine, portrait):
    result = engine.process(portrait, **NATURAL)
    payload = json.loads(json.dumps(result.report(), default=str))
    assert payload["stages"] and payload["face"] and payload["diagnostics"]


def test_skipped_stages_are_reported_not_hidden(engine, portrait, analysis):
    """A tool that legitimately does nothing must say so, with a reason."""
    result = engine.process(portrait, analysis=analysis, **NATURAL)
    for stage in result.stages:
        if not stage.changed:
            assert stage.stats.get("reason"), f"{stage.name} skipped without a reason"


def test_max_side_only_applies_to_paths(engine, portrait_path):
    result = engine.process(portrait_path, max_side=400, skin_smoothing=0.2)
    assert max(result.image.shape[:2]) == 400


def test_save_writes_a_readable_file(engine, portrait, tmp_path):
    result = engine.process(portrait, **NATURAL)
    path = engine.save(result, str(tmp_path / "final.jpg"))
    assert os.path.isfile(path)
    assert load_image(path).shape == portrait.shape


def test_save_preserves_source_metadata(engine, portrait_path, tmp_path):
    source = load_image(portrait_path, max_side=600)
    if not source.meta.icc_profile and not source.meta.exif:
        pytest.skip("fixture carries no EXIF or ICC to preserve")
    result = engine.process(source, **NATURAL)
    path = engine.save(result, str(tmp_path / "final.jpg"))
    reloaded = load_image(path)
    assert bool(reloaded.meta.icc_profile) == bool(source.meta.icc_profile)


# --------------------------------------------------------------------------- debug mode


def test_debug_mode_writes_the_expected_artifacts(engine, portrait, tmp_path):
    debug_dir = tmp_path / "debug"
    result = engine.process(portrait, debug=True, debug_dir=str(debug_dir), **NATURAL)
    assert result.debug_dir == str(debug_dir)

    for relative in (
        "perception/00_bbox.jpg",
        "perception/01_landmarks.jpg",
        "perception/02_mask_overview.jpg",
        "masks/skin.png",
        "masks/skin_probability.png",
        "masks/left_under_eye.png",
        "masks/right_under_eye.png",
        "stages/01_original.jpg",
        "stages/99_final.jpg",
        "before_after.jpg",
        "difference_map.jpg",
        "report.json",
    ):
        assert (debug_dir / relative).is_file(), f"missing debug artifact {relative}"

    stages = sorted(os.listdir(debug_dir / "stages"))
    assert len(stages) == len(result.stages) + 2  # original and final


def test_debug_masks_are_valid_images(engine, portrait, tmp_path):
    import cv2

    debug_dir = tmp_path / "debug"
    engine.process(portrait, debug=True, debug_dir=str(debug_dir), **NATURAL)
    mask = cv2.imread(str(debug_dir / "masks" / "skin.png"), cv2.IMREAD_GRAYSCALE)
    assert mask is not None
    assert mask.shape == portrait.shape
    assert mask.max() > 0, "the written skin mask is blank"


def test_debug_report_matches_the_result(engine, portrait, tmp_path):
    debug_dir = tmp_path / "debug"
    result = engine.process(portrait, debug=True, debug_dir=str(debug_dir), **NATURAL)
    with open(debug_dir / "report.json") as handle:
        written = json.load(handle)
    assert [s["name"] for s in written["stages"]] == [s.name for s in result.stages]


def test_debug_off_writes_nothing(engine, portrait, tmp_path):
    engine.process(portrait, debug=False, debug_dir=str(tmp_path / "unused"), **NATURAL)
    assert not (tmp_path / "unused").exists()


# --------------------------------------------------------------------------- diagnostics


def test_diagnostics_can_be_skipped(engine, portrait):
    result = engine.process(portrait, run_diagnostics=False, **NATURAL)
    assert result.diagnostics == {}


def test_diagnostics_report_no_leak_and_no_warnings(engine, portrait):
    diag = engine.process(portrait, **NATURAL).diagnostics
    assert diag["out_of_region_leak"]["pixels_over_one_level"] == 0.0
    assert diag["texture_retention"] > 0.9
    assert diag["warnings"] == []


def test_permitted_region_covers_only_the_requested_tools(analysis):
    from portrait_engine.diagnostics import permitted_region

    teeth_only = permitted_region(analysis, ["whiten_teeth"])
    both = permitted_region(analysis, ["whiten_teeth", "skin_smoothing"])
    assert float(teeth_only.sum()) <= float(both.sum())
    assert np.allclose(teeth_only, np.minimum(teeth_only, both))


def test_landmark_drift_is_zero_for_an_identical_image(analyzer, portrait, analysis):
    """Establishes the noise floor the warning threshold is calibrated against."""
    from portrait_engine.diagnostics import landmark_drift

    drift = landmark_drift(portrait.rgb, portrait.rgb, analysis, analyzer.detector)
    assert drift["detected"] == 1.0
    assert drift["rms"] == pytest.approx(0.0, abs=1e-9)
