"""A registry for reshaping operations, deliberately separate from ``TOOL_REGISTRY``.

The separation is the whole point and it is not bureaucracy. The six tools in
``portrait_engine.tools`` carry a provable claim: they cannot move a pixel, and AST tests fail if
one of them so much as names a warping function. Reshaping breaks that claim by design, so it
gets its own registry and its own contracts.

The consequence a caller can rely on: a run that used only ``TOOL_REGISTRY`` is still provably
identity-preserving. If both lived in one registry that statement would be unverifiable, and the
future agent would be one tool call away from silently reshaping a face nobody asked to reshape.

``reshape_schemas()`` mirrors ``tool_schemas()`` so the agent side is the same shape of work, but
they are two separate lists and the caller has to ask for the second one on purpose.
"""

from __future__ import annotations

import inspect
from typing import Any

from portrait_engine.tools.base import (
    ParameterSpec,
    ToolFn,
    ToolSpec,
    _param_docs,
    _summary_from_doc,
)

#: Registered reshaping operations, keyed by function name. Separate from ``TOOL_REGISTRY``.
RESHAPE_REGISTRY: dict[str, ToolSpec] = {}

_HIDDEN = {"image", "analysis", "config", "strict", "debug", "subsample"}


def register_reshape(fn: ToolFn) -> ToolFn:
    """Register a reshaping operation and derive its JSON contract from its signature."""
    signature = inspect.signature(fn)
    docs = _param_docs(fn.__doc__)
    specs: list[ParameterSpec] = []

    for name, parameter in signature.parameters.items():
        if name in _HIDDEN or parameter.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        annotation = parameter.annotation
        json_type = "boolean" if annotation is bool else "integer" if annotation is int else "number"
        specs.append(
            ParameterSpec(
                name=name,
                type=json_type,
                default=parameter.default,
                minimum=0.0 if "strength" in name else None,
                maximum=1.0 if "strength" in name else None,
                description=docs.get(name, ""),
            )
        )

    RESHAPE_REGISTRY[fn.__name__] = ToolSpec(
        name=fn.__name__,
        fn=fn,
        summary=_summary_from_doc(fn.__doc__),
        parameters=tuple(specs),
    )
    return fn


def reshape_schemas() -> list[dict[str, Any]]:
    """Every reshaping operation as a JSON contract.

    Kept separate from :func:`portrait_engine.tools.tool_schemas` on purpose. An agent that is
    only allowed to retouch is handed that list; an agent that is also allowed to reshape is
    handed both, as a deliberate act by whoever wires it up.
    """
    return [RESHAPE_REGISTRY[name].as_contract() for name in sorted(RESHAPE_REGISTRY)]
