"""Geometric reshaping: the one part of the engine allowed to move pixels.

Kept in its own package with its own registry so that the claim made by
``portrait_engine.tools`` stays provable. Those six tools cannot move a pixel, and AST tests
fail if one of them names a warping function; nothing here weakens that. A caller has to reach
for ``RESHAPE_REGISTRY`` or ``reshape_schemas()`` deliberately.

See ``docs/RESHAPING.md`` for why the separation is drawn here and what it buys.
"""

from portrait_engine.reshaping.base import RESHAPE_REGISTRY, register_reshape, reshape_schemas
from portrait_engine.reshaping.nose import NoseConfig, narrow_nose
from portrait_engine.reshaping.warp import (
    WarpContractError,
    WarpReport,
    apply_warp,
    displacement_from_handles,
    jacobian_minimum,
)

__all__ = [
    "RESHAPE_REGISTRY",
    "NoseConfig",
    "WarpContractError",
    "WarpReport",
    "apply_warp",
    "displacement_from_handles",
    "jacobian_minimum",
    "narrow_nose",
    "register_reshape",
    "reshape_schemas",
]
