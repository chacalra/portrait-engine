"""Identity-preservation diagnostics.

These are measurements, not filters. They exist so that "identity is preserved" is a number
someone can check rather than a claim in a README. All three are cheap: no extra models beyond
the landmarker that perception already loaded.

* :func:`landmark_drift` re-runs the landmarker on the output and measures how far the facial
  landmarks moved, normalized by the interocular distance. Any tool that warped geometry would
  show up here. Note that the converse does not hold: the landmarker re-estimates from pixels,
  so a small nonzero drift is expected from any legitimate edit. See
  :data:`LANDMARK_DRIFT_WARNING` for the measured noise floor.
* :func:`texture_retention` compares high-frequency energy in the skin before and after. This
  is the "plastic skin" detector.
* :func:`out_of_region_leak` measures the largest change anywhere outside the union of the
  regions the tools were allowed to touch.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

from portrait_engine.core import ops
from portrait_engine.core.colorspace import rgb_to_lab
from portrait_engine.core.types import FaceAnalysis

#: Landmark drift above this fraction of the interocular distance is reported as a warning.
#:
#: Calibrated with control experiments, not guessed. Re-running the landmarker on the *identical*
#: image gives exactly 0.0, so the detector is deterministic. But re-running it after a change
#: that provably cannot move geometry (a uniform 5 percent exposure shift, or 1/255 of Gaussian
#: noise) already gives an RMS drift of 0.004 to 0.013 of the interocular distance. That is the
#: detector's re-estimation sensitivity, and it fully accounts for the drift the pipeline
#: measures at maximum strength (0.002 to 0.011). The threshold is set at roughly twice the
#: worst control value.
#:
#: Read this metric as a smoke alarm, not as the guarantee. The actual guarantee that geometry
#: is preserved is structural: `apply_masked` only ever blends per pixel, and a test asserts
#: that no tool module imports a warping function.
LANDMARK_DRIFT_WARNING = 0.025

#: Texture retention below this is the threshold for "looks plastic", from the brief's quality
#: criteria. Calibrated against the tools at their maximum safe strength, which stay above 0.9.
TEXTURE_RETENTION_WARNING = 0.55

#: Anything above one 8-bit level outside the permitted regions is a real leak.
LEAK_WARNING = 1.0 / 255.0


def landmark_drift(
    before: np.ndarray, after: np.ndarray, analysis: FaceAnalysis, detector: Any
) -> dict[str, float]:
    """RMS and maximum landmark displacement between input and output, in interocular units.

    Returns zeros with ``detected: 0`` if the landmarker cannot find the face in the output,
    which would itself be a serious red flag.
    """
    faces = detector.detect(after)
    if not faces:
        return {"detected": 0.0, "rms": float("nan"), "max": float("nan")}
    face = max(faces, key=lambda f: f.bbox.width * f.bbox.height)
    delta = face.landmarks - analysis.landmarks
    distance = np.linalg.norm(delta, axis=1)
    scale = max(analysis.geometry.interocular_px, 1.0)
    return {
        "detected": 1.0,
        "rms": float(np.sqrt((distance**2).mean()) / scale),
        "max": float(distance.max() / scale),
    }


def texture_retention(before: np.ndarray, after: np.ndarray, mask: np.ndarray) -> float:
    """High-frequency energy of ``L*`` after over before, inside ``mask``."""
    energy_before = ops.high_frequency_energy(rgb_to_lab(before)[..., 0], mask)
    energy_after = ops.high_frequency_energy(rgb_to_lab(after)[..., 0], mask)
    if energy_before <= 1e-6:
        return 1.0
    return float(energy_after / energy_before)


def out_of_region_leak(
    before: np.ndarray, after: np.ndarray, permitted: np.ndarray
) -> dict[str, float]:
    """Largest and mean change outside the permitted region, in 8-bit levels."""
    outside = permitted <= 1e-3
    if not outside.any():
        return {"max_levels": 0.0, "mean_levels": 0.0, "pixels_over_one_level": 0.0}
    delta = np.abs(after[outside] - before[outside]) * 255.0
    return {
        "max_levels": float(delta.max()),
        "mean_levels": float(delta.mean()),
        "pixels_over_one_level": float((delta > 1.0).sum()),
    }


def permitted_region(analysis: FaceAnalysis, tools: Iterable[str]) -> np.ndarray:
    """Union of the masks the requested tools are allowed to write to.

    Kept explicit and in one place, so the leak check cannot silently widen when a tool changes.
    """
    per_tool = {
        "skin_cleanup": ("skin",),
        "skin_smoothing": ("skin",),
        "equalize_skin_tone": ("skin",),
        "reduce_dark_circles": ("left_under_eye", "right_under_eye"),
        "whiten_teeth": ("teeth",),
        "balance_face_lighting": ("face_photometric",),
    }
    union = np.zeros(analysis.image_shape, dtype=np.float32)
    for tool in tools:
        for name in per_tool.get(tool, ()):
            union = np.maximum(union, analysis.masks[name])
    return union


def summarize(
    before: np.ndarray,
    after: np.ndarray,
    analysis: FaceAnalysis,
    tools: Iterable[str],
    detector: Any | None = None,
) -> dict[str, Any]:
    """Run every diagnostic and attach warnings for anything out of bounds."""
    tools = list(tools)
    permitted = permitted_region(analysis, tools)
    leak = out_of_region_leak(before, after, permitted)
    retention = texture_retention(before, after, analysis.masks["skin"])

    report: dict[str, Any] = {
        "texture_retention": round(retention, 4),
        "out_of_region_leak": {k: round(v, 4) for k, v in leak.items()},
        "permitted_area_fraction": round(float(permitted.mean()), 5),
        "changed_pixel_fraction": round(
            float((np.abs(after - before).max(axis=2) > LEAK_WARNING).mean()), 5
        ),
    }
    if detector is not None:
        drift = landmark_drift(before, after, analysis, detector)
        report["landmark_drift"] = {k: (round(v, 5) if np.isfinite(v) else None) for k, v in drift.items()}

    warnings: list[str] = []
    if retention < TEXTURE_RETENTION_WARNING:
        warnings.append(
            f"texture retention {retention:.2f} is below {TEXTURE_RETENTION_WARNING}: "
            "the skin is likely to read as plastic"
        )
    if leak["max_levels"] > 1.0:
        warnings.append(
            f"{int(leak['pixels_over_one_level'])} pixel components changed outside the "
            f"permitted regions, up to {leak['max_levels']:.1f}/255"
        )
    drift = report.get("landmark_drift")
    if drift and drift.get("detected") == 0.0:
        warnings.append("the landmarker no longer finds the face in the output")
    elif drift and drift.get("rms") is not None and drift["rms"] > LANDMARK_DRIFT_WARNING:
        warnings.append(
            f"landmark RMS drift {drift['rms']:.4f} of the interocular distance exceeds "
            f"{LANDMARK_DRIFT_WARNING}: facial geometry may have moved"
        )
    report["warnings"] = warnings
    return report
