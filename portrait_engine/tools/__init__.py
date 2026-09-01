"""Retouching tools.

Every tool is an independent pure function. Importing this package registers all of them, so
``TOOL_REGISTRY`` and ``tool_schemas()`` are complete after ``import portrait_engine.tools``.
"""

from portrait_engine.tools.base import TOOL_REGISTRY, ToolSpec, tool_schemas, validate_strength
from portrait_engine.tools.dark_circles import reduce_dark_circles
from portrait_engine.tools.face_lighting import balance_face_lighting
from portrait_engine.tools.skin_cleanup import skin_cleanup
from portrait_engine.tools.skin_smoothing import skin_smoothing
from portrait_engine.tools.skin_tone import equalize_skin_tone
from portrait_engine.tools.teeth_whitening import whiten_teeth

__all__ = [
    "TOOL_REGISTRY",
    "ToolSpec",
    "balance_face_lighting",
    "equalize_skin_tone",
    "reduce_dark_circles",
    "skin_cleanup",
    "skin_smoothing",
    "tool_schemas",
    "validate_strength",
    "whiten_teeth",
]
