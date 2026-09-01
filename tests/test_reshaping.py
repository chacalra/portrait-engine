"""Reshaping: the separation from the retouching tools, and the warp's own guardrails.

The first test in this file is the one that matters most. Everything else in this engine rests
on a claim that can be checked rather than trusted, and adding a module that moves pixels is
exactly the change that could quietly dissolve it.
"""

from __future__ import annotations

import numpy as np
import pytest

from portrait_engine.reshaping import (
    RESHAPE_REGISTRY,
    WarpContractError,
    apply_warp,
    displacement_from_handles,
    jacobian_minimum,
    narrow_nose,
    reshape_schemas,
)
from portrait_engine.tools import TOOL_REGISTRY, tool_schemas

# --------------------------------------------------------------------------- the separation


def test_the_two_registries_never_mix():
    """The architectural guarantee, in one assertion.

    A run that used only ``TOOL_REGISTRY`` provably did not move a pixel: those modules are
    covered by an AST test over 18 warping function names. That statement is only worth anything
    while the reshaping operations stay out of that registry. If they ever merge, the claim
    becomes unverifiable and the future agent is one tool call away from reshaping a face nobody
    asked to reshape.
    """
    assert RESHAPE_REGISTRY, "no reshaping operations are registered"
    overlap = set(TOOL_REGISTRY) & set(RESHAPE_REGISTRY)
    assert not overlap, f"these appear in both registries: {sorted(overlap)}"

    tool_names = {schema["name"] for schema in tool_schemas()}
    reshape_names = {schema["name"] for schema in reshape_schemas()}
    assert not (tool_names & reshape_names)
    # Importing the reshaping package must not add anything to the retouching contracts. An
    # agent handed only `tool_schemas()` cannot reach a warp even by accident.
    assert "narrow_nose" not in tool_names


def test_reshaping_contracts_are_agent_ready():
    for schema in reshape_schemas():
        assert schema["name"]
        assert schema["description"]
        strength = schema["parameters"]["properties"]["strength"]
        assert strength["minimum"] == 0.0
        assert strength["maximum"] == 1.0


# --------------------------------------------------------------------------- warp guardrails


@pytest.fixture
def flat():
    rng = np.random.default_rng(0)
    return rng.random((160, 160, 3)).astype(np.float32)


def _field(shape, offset, radius=20):
    handles = np.array([[80.0, 80.0]], dtype=np.float32)
    offsets = np.array([offset], dtype=np.float32)
    return displacement_from_handles(shape, handles, offsets, radius)


def test_warp_refuses_to_exceed_the_displacement_bound(flat):
    dx, dy = _field(flat.shape[:2], (30.0, 0.0))
    region = np.ones(flat.shape[:2], dtype=np.float32)
    with pytest.raises(WarpContractError, match="exceeds the bound"):
        apply_warp(flat, dx, dy, region, interocular_px=100.0, max_displacement=0.05)


def test_warp_refuses_a_folding_field(flat):
    """A field steep enough to fold melts the image onto itself. Refused, not shipped."""
    dx, dy = _field(flat.shape[:2], (60.0, 0.0), radius=3)
    region = np.ones(flat.shape[:2], dtype=np.float32)
    with pytest.raises(WarpContractError, match="folds"):
        apply_warp(flat, dx, dy, region, interocular_px=1000.0, max_displacement=1.0)


def test_warp_changes_nothing_outside_its_region(flat):
    """Containment, verified on the output rather than assumed from the intent."""
    dx, dy = _field(flat.shape[:2], (4.0, 0.0))
    region = np.zeros(flat.shape[:2], dtype=np.float32)
    region[60:100, 60:100] = 1.0
    out, _ = apply_warp(flat, dx, dy, region, interocular_px=100.0, max_displacement=0.5)
    outside = region <= 0.0
    assert np.array_equal(out[outside], flat[outside])


def test_jacobian_is_one_for_the_identity():
    zero = np.zeros((64, 64), dtype=np.float32)
    assert abs(jacobian_minimum(zero, zero) - 1.0) < 1e-4


# --------------------------------------------------------------------------- narrow_nose


def test_narrow_nose_is_a_no_op_at_zero(portrait, analysis):
    result = narrow_nose(portrait.rgb, analysis, 0.0)
    assert not result.changed
    assert np.array_equal(result.image, portrait.rgb)


def test_narrow_nose_achieves_the_width_it_states(portrait, analysis):
    """The calibration property, and the reason this module still fits the project's method.

    "A better nose" has no objective definition, but "remove 12 percent of the alar width" does,
    and it is checkable in landmark space. The operation measures the field it built rather than
    trusting the offsets it asked for, and rescales until the measurement matches.

    This test caught a real bug: the baseline width was measured as the full extent of the nose
    ring while the result was measured between two handle-group means. Two different estimators,
    so the rescaling loop was correcting a quantity that contained a constant offset, and the
    achieved narrowing overshot the request by about 19 percent at every strength.
    """
    for strength in (0.3, 0.6, 1.0):
        result = narrow_nose(portrait.rgb, analysis, strength)
        if not result.changed:
            pytest.skip(f"no usable nose region: {result.stats.get('reason')}")
        achieved = result.stats["narrowing_fraction"]
        requested = result.stats["requested_fraction"]
        assert abs(achieved - requested) < 0.01, (
            f"strength {strength}: asked to remove {requested:.3f} of the alar width, "
            f"removed {achieved:.3f}"
        )


def test_narrow_nose_stays_inside_its_bounds(portrait, analysis):
    result = narrow_nose(portrait.rgb, analysis, 1.0)
    if not result.changed:
        pytest.skip("no usable nose region")
    assert result.stats["min_jacobian"] > 0.0, "the warp folded"
    assert result.stats["max_displacement_iod"] <= 0.08 + 1e-6
    # A nose operation must not be moving a large part of the frame.
    assert result.stats["moved_area_fraction"] < 0.10


def test_narrow_nose_leaves_the_eyes_and_mouth_alone(portrait, analysis):
    """The region bound, checked against the features a nose warp must never touch."""
    result = narrow_nose(portrait.rgb, analysis, 1.0)
    if not result.changed:
        pytest.skip("no usable nose region")
    difference = np.abs(result.image - portrait.rgb).max(axis=2)
    for name in ("eyes", "lips", "teeth"):
        mask = analysis.masks[name] > 0.5
        if mask.sum() < 32:
            continue
        worst = float(difference[mask].max()) * 255.0
        assert worst < 1.0, f"{name} moved by up to {worst:.2f} of an 8-bit level"


def test_narrow_nose_scales_monotonically(portrait, analysis):
    widths = []
    for strength in (0.25, 0.5, 1.0):
        result = narrow_nose(portrait.rgb, analysis, strength)
        if not result.changed:
            pytest.skip("no usable nose region")
        widths.append(result.stats["alar_width_after_px"])
    assert widths[0] > widths[1] > widths[2], f"not monotonic: {widths}"
