"""``PortraitEngine``: runs perception once and applies the tools in order.

The engine is a thin orchestrator on purpose. It does not contain image processing. Everything
it does is: load, analyze once, call independent tools with shared masks, record debug output,
measure diagnostics, return.

The tools stay usable on their own. Nothing in a tool knows the engine exists.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

import numpy as np

from portrait_engine import diagnostics
from portrait_engine.core.config import DEFAULT_CONFIG, EngineConfig
from portrait_engine.core.image import PortraitImage, load_image, save_image
from portrait_engine.core.types import (
    FaceAnalysis,
    PortraitEngineError,
    RetouchResult,
    StageRecord,
)
from portrait_engine.debug.recorder import DebugRecorder
from portrait_engine.perception.face_analyzer import AnalyzerConfig, FaceAnalyzer
from portrait_engine.tools import TOOL_REGISTRY
from portrait_engine.tools.base import validate_strength

#: Maps the friendly pipeline keyword arguments to registered tool names.
STRENGTH_ALIASES: dict[str, str] = {
    "skin_cleanup": "skin_cleanup",
    "skin_smoothing": "skin_smoothing",
    "dark_circles": "reduce_dark_circles",
    "reduce_dark_circles": "reduce_dark_circles",
    "teeth_whitening": "whiten_teeth",
    "whiten_teeth": "whiten_teeth",
    "face_lighting": "balance_face_lighting",
    "balance_face_lighting": "balance_face_lighting",
}


class PortraitEngine:
    """Apply a set of retouching tools to a portrait.

    Example::

        engine = PortraitEngine()
        result = engine.process("portrait.jpg", skin_smoothing=0.2, teeth_whitening=0.1)
        engine.save(result, "out.jpg")
    """

    def __init__(
        self,
        config: EngineConfig | None = None,
        analyzer: FaceAnalyzer | None = None,
        analyzer_config: AnalyzerConfig | None = None,
    ):
        self.config = config or DEFAULT_CONFIG
        self.analyzer = analyzer or FaceAnalyzer(analyzer_config)

    # ------------------------------------------------------------------ public API
    def analyze(self, image: PortraitImage | np.ndarray | str) -> tuple[PortraitImage, FaceAnalysis]:
        """Load if needed, then run perception once. Useful on its own for the API layer."""
        portrait = self._coerce(image)
        return portrait, self.analyzer.analyze(portrait)

    def process(
        self,
        image: PortraitImage | np.ndarray | str,
        *,
        analysis: FaceAnalysis | None = None,
        debug: bool = False,
        debug_dir: str | None = None,
        max_side: int | None = None,
        strict: bool = True,
        order: tuple[str, ...] | None = None,
        run_diagnostics: bool = True,
        **strengths: float,
    ) -> RetouchResult:
        """Run the requested tools and return the retouched image plus a full report.

        Args:
            image: a path, a NumPy array, or a ``PortraitImage``.
            analysis: reuse a previous analysis instead of recomputing it.
            debug: write inspectable artifacts. Requires ``debug_dir``, or defaults to
                ``./output/debug``.
            max_side: downscale the long side on load. Only applies when ``image`` is a path.
            strict: enforce the "no change outside the mask" guardrail on every tool.
            order: override the execution order.
            run_diagnostics: measure identity preservation. Adds one landmarker inference.
            **strengths: one keyword per tool, in ``[0, 1]``. Accepts either the tool name
                (``whiten_teeth``) or its short alias (``teeth_whitening``). Omitted or zero
                means the tool does not run.

        Raises:
            PortraitEngineError: unknown tool name.
            NoFaceDetectedError, MultipleFacesError: from the perception layer.
            StrengthOutOfRangeError: a strength outside [0, 1].
        """
        requested = self._resolve_strengths(strengths)
        portrait = self._coerce(image, max_side=max_side)
        original = portrait.rgb.copy()

        recorder = DebugRecorder(
            directory=debug_dir or (os.path.join("output", "debug") if debug else None),
            enabled=debug,
        )

        if analysis is None:
            analysis = self.analyzer.analyze(portrait)
        recorder.record_original(portrait)
        recorder.record_perception(portrait, analysis)

        sequence = order or self.config.order
        unknown = set(requested) - set(sequence)
        if unknown:
            raise PortraitEngineError(
                f"tools not in the execution order: {sorted(unknown)}. Known: {sorted(sequence)}"
            )

        current = portrait.rgb
        stages: list[StageRecord] = []
        index = 1
        for tool_name in sequence:
            strength = requested.get(tool_name, 0.0)
            if strength <= 0.0:
                continue
            index += 1
            before = current
            result = self._run_tool(tool_name, current, analysis, strength, strict, debug)
            current = result.image
            stages.append(
                StageRecord(
                    name=tool_name,
                    strength=strength,
                    changed=result.changed,
                    stats=result.stats,
                )
            )
            recorder.record_stage(index, tool_name, before, current, result.debug)

        final = portrait.with_rgb(current)
        recorder.record_final(original, current, analysis)

        report_diagnostics: dict[str, Any] = {}
        if run_diagnostics and stages:
            report_diagnostics = diagnostics.summarize(
                original,
                current,
                analysis,
                [s.name for s in stages],
                detector=self.analyzer.detector,
            )

        result = RetouchResult(
            image=final.rgb,
            analysis=analysis,
            stages=stages,
            diagnostics=report_diagnostics,
            debug_dir=recorder.directory if debug else None,
            portrait=final,
        )
        recorder.record_report(result.report())
        return result

    def save(self, result: RetouchResult, path: str, quality: int = 95) -> str:
        """Write the retouched image, preserving EXIF and ICC from the source when available."""
        portrait = result.portrait or PortraitImage(rgb=result.image)
        return save_image(portrait, path, quality=quality)

    # ------------------------------------------------------------------ internals
    def _coerce(
        self, image: PortraitImage | np.ndarray | str, max_side: int | None = None
    ) -> PortraitImage:
        if isinstance(image, PortraitImage):
            return image
        if isinstance(image, (str, os.PathLike)):
            return load_image(image, max_side=max_side)
        from portrait_engine.core.image import from_array

        return from_array(image)

    def _resolve_strengths(self, strengths: Mapping[str, float]) -> dict[str, float]:
        resolved: dict[str, float] = {}
        for key, value in strengths.items():
            tool_name = STRENGTH_ALIASES.get(key)
            if tool_name is None:
                raise PortraitEngineError(
                    f"unknown tool {key!r}. Available: {sorted(set(STRENGTH_ALIASES))}"
                )
            strength = validate_strength(value, name=key)
            resolved[tool_name] = max(resolved.get(tool_name, 0.0), strength)
        return resolved

    def _run_tool(
        self,
        name: str,
        rgb: np.ndarray,
        analysis: FaceAnalysis,
        strength: float,
        strict: bool,
        debug: bool,
    ):
        spec = TOOL_REGISTRY.get(name)
        if spec is None:
            raise PortraitEngineError(f"tool {name!r} is not registered")
        tool_config = getattr(self.config, _CONFIG_ATTR[name], None)
        kwargs: dict[str, Any] = {"strict": strict, "debug": debug}
        if tool_config is not None:
            kwargs["config"] = tool_config
        if name != "whiten_teeth":  # the only tool with no guided-filter subsampling
            kwargs["subsample"] = self.config.subsample
        return spec.fn(rgb, analysis, strength, **kwargs)


#: Which `EngineConfig` field belongs to which tool.
_CONFIG_ATTR: dict[str, str] = {
    "skin_cleanup": "skin_cleanup",
    "skin_smoothing": "skin_smoothing",
    "reduce_dark_circles": "dark_circles",
    "whiten_teeth": "teeth_whitening",
    "balance_face_lighting": "face_lighting",
}
