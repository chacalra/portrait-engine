"""Identity-preservation guardrails.

These tests are the point of the whole design. They check the properties the brief lists as
non-negotiable, structurally where possible and numerically otherwise.
"""

from __future__ import annotations

import ast
import os

import numpy as np
import pytest

from portrait_engine import MaskContractError, tool_schemas
from portrait_engine.core.masking import (
    apply_masked,
    edited_area_fraction,
    subtract_protected,
    validate_alpha,
    verify_no_change_outside,
)
from portrait_engine.tools import TOOL_REGISTRY

TOOLS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "portrait_engine", "tools"
)

#: Functions that move pixels around. No tool may reference any of them: that is what
#: "no geometric warping" means operationally.
FORBIDDEN_CALLS = frozenset(
    {
        "warpAffine", "warpPerspective", "remap", "getAffineTransform",
        "getPerspectiveTransform", "estimateAffine2D", "estimateAffinePartial2D",
        "seamlessClone", "inpaint", "resize", "pyrUp", "pyrDown", "flip", "transpose",
        "getRotationMatrix2D", "convertMaps", "initUndistortRectifyMap", "undistort",
    }
)


def _tool_modules() -> list[str]:
    return [
        os.path.join(TOOLS_DIR, name)
        for name in sorted(os.listdir(TOOLS_DIR))
        if name.endswith(".py") and name not in ("__init__.py", "base.py")
    ]


def test_no_tool_references_a_warping_function():
    """Static check, so it holds for code that has never been executed.

    A numeric test can only observe the paths a fixture happens to exercise. This one fails the
    moment anyone types ``cv2.warpAffine`` in a tool.
    """
    offenders: list[str] = []
    for path in _tool_modules():
        with open(path) as handle:
            tree = ast.parse(handle.read(), filename=path)
        for node in ast.walk(tree):
            name = None
            if isinstance(node, ast.Attribute):
                name = node.attr
            elif isinstance(node, ast.Name):
                name = node.id
            if name in FORBIDDEN_CALLS:
                offenders.append(f"{os.path.basename(path)}:{node.lineno} -> {name}")
    assert not offenders, "tools must not warp geometry: " + "; ".join(offenders)


#: Names that would let a tool build a per-pixel field from *coordinates* rather than from the
#: image. See `test_no_tool_can_build_a_geometric_profile` for why that is the line.
FORBIDDEN_COORDINATE_SOURCES = frozenset(
    {"landmarks", "landmarks_3d", "mgrid", "meshgrid", "indices", "fromfunction",
     "distanceTransform", "ogrid"}
)


def test_no_tool_can_build_a_geometric_profile():
    """Closes photometric reshaping: sculpting a face without moving a pixel.

    A retoucher can narrow a nose, define a jaw or reshape a brow purely with dodge and burn.
    No pixel moves, so every geometric guardrail in this file passes, and the perceived anatomy
    still changes. Three statistical detectors were built and measured against a real
    dodge-and-burn control, and all three failed; the numbers are in `docs/ALGORITHMS.md`. The
    short version is that a sculpt *is* a smooth low-frequency luminance change, which is
    exactly what legitimate retouching is, so no statistic separates them.

    What does separate them is where the modulation field comes from:

    * legitimate: derived from the **image** (a local mean, an illumination estimate, a measured
      deficit) or from a mask that perception built;
    * sculpting: derived from **landmark coordinates**, a profile imposed by anatomy regardless
      of what the pixels say.

    So the rule is on the tool's *inputs*, the same shape of rule as the geometry one. A tool may
    read `analysis.masks` and `analysis.geometry` (a scalar scale, for filter radii) and nothing
    else. You cannot express "lighten the midline, darken the sidewalls" from a mask and a
    radius; you need the coordinates.

    Honest about the limit: this is not airtight. Someone determined could recover a midline from
    a mask. It makes sculpting something you have to deliberately work around rather than
    something you can drift into, which is what a guardrail is for.
    """
    offenders: list[str] = []
    for path in _tool_modules():
        with open(path) as handle:
            tree = ast.parse(handle.read(), filename=path)
        for node in ast.walk(tree):
            name = node.attr if isinstance(node, ast.Attribute) else (
                node.id if isinstance(node, ast.Name) else None
            )
            if name in FORBIDDEN_COORDINATE_SOURCES:
                offenders.append(f"{os.path.basename(path)}:{node.lineno} -> {name}")
    assert not offenders, (
        "tools must not build a field from coordinates, only from the image and from "
        "perception's masks: " + "; ".join(offenders)
    )


def test_tools_read_only_masks_and_geometry_from_the_analysis():
    """The positive half of the rule above, so it cannot be satisfied by renaming an attribute.

    Whatever a tool reaches for on the `FaceAnalysis` is enumerated here. Adding a new one is a
    deliberate act that has to update this list, at which point someone has to think about
    whether it can be used to impose a shape.
    """
    allowed = {"masks", "geometry"}
    offenders: list[str] = []
    for path in _tool_modules():
        with open(path) as handle:
            tree = ast.parse(handle.read(), filename=path)
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == "analysis"
                and node.attr not in allowed
            ):
                offenders.append(f"{os.path.basename(path)}:{node.lineno} -> analysis.{node.attr}")
    assert not offenders, (
        f"tools may only read {sorted(allowed)} from the analysis: " + "; ".join(offenders)
    )


def test_every_tool_routes_through_apply_masked():
    """A tool that returns pixels without going through the compositor would bypass every
    guardrail, so require the call to be present in each tool module."""
    for path in _tool_modules():
        with open(path) as handle:
            source = handle.read()
        assert "apply_masked(" in source, f"{os.path.basename(path)} never calls apply_masked"


# --------------------------------------------------------------------------- apply_masked


@pytest.fixture
def pair():
    rng = np.random.default_rng(0)
    original = rng.random((32, 32, 3)).astype(np.float32)
    edited = np.clip(original + 0.25, 0.0, 1.0)
    return original, edited


def test_apply_masked_leaves_zero_alpha_bit_identical(pair):
    original, edited = pair
    alpha = np.zeros(original.shape[:2], dtype=np.float32)
    alpha[8:16, 8:16] = 1.0
    out = apply_masked(original, edited, alpha)
    outside = alpha == 0.0
    assert np.array_equal(out[outside], original[outside])
    assert np.allclose(out[8:16, 8:16], edited[8:16, 8:16])


def test_apply_masked_rejects_a_shape_change(pair):
    original, _ = pair
    with pytest.raises(MaskContractError, match="geometry change detected"):
        apply_masked(original, original[:16, :16], np.ones(original.shape[:2], np.float32))


def test_apply_masked_detects_a_leak_in_strict_mode(pair):
    """Simulates a buggy tool: it edits the whole frame but declares a small mask."""
    original, edited = pair
    alpha = np.zeros(original.shape[:2], dtype=np.float32)
    alpha[8:16, 8:16] = 1.0
    leaked = edited.copy()
    with pytest.raises(MaskContractError, match="identity-preservation violation"):
        verify_no_change_outside(original, leaked, alpha, name="buggy_tool")


def test_apply_masked_sanitizes_non_finite_input(pair):
    original, edited = pair
    edited = edited.copy()
    edited[0, 0] = np.nan
    edited[1, 1] = np.inf
    out = apply_masked(original, edited, np.ones(original.shape[:2], np.float32))
    assert np.isfinite(out).all()
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_validate_alpha_enforces_the_contract():
    with pytest.raises(MaskContractError, match="shape"):
        validate_alpha(np.zeros((4, 4), np.float32), (8, 8))
    with pytest.raises(MaskContractError, match=r"out of \[0,1\]"):
        validate_alpha(np.full((8, 8), -1.0, np.float32), (8, 8))
    assert validate_alpha(np.zeros((8, 8, 1), np.float32), (8, 8)).shape == (8, 8)


def test_subtract_protected_never_widens():
    mask = np.full((8, 8), 1.0, np.float32)
    protected = np.full((8, 8), 0.5, np.float32)
    result = subtract_protected(mask, protected)
    assert float(result.max()) <= 0.5 + 1e-6


def test_edited_area_fraction_is_a_fraction():
    assert edited_area_fraction(np.ones((4, 4), np.float32)) == pytest.approx(1.0)
    assert edited_area_fraction(np.zeros((4, 4), np.float32)) == 0.0


# --------------------------------------------------------------------------- end to end


MAX_STRENGTHS = {
    "skin_cleanup": 1.0,
    "skin_smoothing": 1.0,
    "dark_circles": 1.0,
    "teeth_whitening": 1.0,
    "face_lighting": 1.0,
}


@pytest.mark.slow
def test_full_strength_pipeline_does_not_leak_or_flatten(engine, portrait):
    """At the maximum safe strength: nothing outside the permitted regions moves, the skin keeps
    its texture, and the geometry does not move beyond detector noise."""
    result = engine.process(portrait, **MAX_STRENGTHS)
    diag = result.diagnostics
    assert diag["out_of_region_leak"]["max_levels"] < 1.0
    assert diag["out_of_region_leak"]["pixels_over_one_level"] == 0.0
    assert diag["texture_retention"] > 0.7
    assert diag["landmark_drift"]["detected"] == 1.0
    assert not diag["warnings"], diag["warnings"]


SKIN_TOOL_STRENGTHS = {"skin_cleanup": 1.0, "skin_smoothing": 1.0, "dark_circles": 1.0}


@pytest.mark.slow
def test_skin_tools_never_touch_eyes_brows_or_lips(engine, portrait, analysis):
    """The skin tools must leave eyes, eyebrows and lips bit-for-bit unchanged, even at full
    strength. This is the requirement behind "no smoothed eyebrows" and "no modified lips"."""
    result = engine.process(portrait, analysis=analysis, **SKIN_TOOL_STRENGTHS)
    for name in ("eyes", "eyebrows", "lips"):
        core = analysis.masks[name] > 0.9
        if core.sum() < 32:
            continue
        delta = np.abs(result.image[core] - portrait.rgb[core]).max() * 255.0
        assert delta < 1.0, f"{name} changed by {delta:.2f}/255"


@pytest.mark.slow
def test_lighting_changes_brow_exposure_but_not_brow_texture(engine, portrait, analysis):
    """``balance_face_lighting`` deliberately covers the whole face, eyebrows included.

    Excluding them would leave the brows at the original exposure while the surrounding skin
    moved, which reads as a retouching error. What it must not do is *smooth* them: the local
    contrast of the eyebrow must survive. Measured in the log domain, where a smooth
    multiplicative gain is an additive offset and therefore invisible to a Laplacian.
    """
    from portrait_engine.tools.face_lighting import _local_contrast_retention

    brows = analysis.masks["eyebrows"]
    if float(brows.sum()) < 64:
        pytest.skip("no eyebrow mask in this fixture")
    result = engine.process(portrait, analysis=analysis, face_lighting=1.0)
    retention = _local_contrast_retention(portrait.rgb, result.image, brows)
    assert retention > 0.85, f"eyebrow local contrast dropped to {retention:.2f}"


@pytest.mark.slow
def test_no_tool_changes_the_image_size(portrait, analysis):
    for name, spec in TOOL_REGISTRY.items():
        out = spec.fn(portrait.rgb, analysis, 0.5)
        assert out.image.shape == portrait.rgb.shape, name
        assert out.image.dtype == np.float32, name


# --------------------------------------------------------------------------- agent contract


def test_every_tool_has_a_usable_contract():
    schemas = {schema["name"]: schema for schema in tool_schemas()}
    assert set(schemas) == set(TOOL_REGISTRY)
    for name, schema in schemas.items():
        assert schema["description"], f"{name} has no description"
        strength = schema["parameters"]["properties"]["strength"]
        assert strength["type"] == "number"
        assert strength["minimum"] == 0.0
        assert strength["maximum"] == 1.0
        assert strength["description"], f"{name}.strength is undocumented"


def test_contract_is_json_serializable():
    import json

    assert json.loads(json.dumps(tool_schemas()))
