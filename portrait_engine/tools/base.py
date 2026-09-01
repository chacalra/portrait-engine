"""Tool contract, registry and shared guardrails.

Every tool is a pure function with the same shape::

    tool(image, analysis, strength=..., **params) -> ToolResult

It receives a ``FaceAnalysis`` and never runs a model. It writes pixels only through
:func:`portrait_engine.core.masking.apply_masked`.

The ``@register`` decorator also builds the JSON contract that the future agent will use for
tool calling. The contract is derived from the signature and the docstring, so it cannot drift
away from the code.
"""

from __future__ import annotations

import inspect
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from portrait_engine.core.types import StrengthOutOfRangeError

ToolFn = Callable[..., Any]


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    type: str
    default: Any
    minimum: float | None = None
    maximum: float | None = None
    description: str = ""

    def as_json_schema(self) -> dict[str, Any]:
        schema: dict[str, Any] = {"type": self.type}
        if self.minimum is not None:
            schema["minimum"] = self.minimum
        if self.maximum is not None:
            schema["maximum"] = self.maximum
        if self.default is not inspect.Parameter.empty:
            schema["default"] = self.default
        if self.description:
            schema["description"] = self.description
        return schema


@dataclass(frozen=True)
class ToolSpec:
    """A registered tool and its machine-readable contract."""

    name: str
    fn: ToolFn
    summary: str
    parameters: tuple[ParameterSpec, ...] = field(default_factory=tuple)

    def as_contract(self) -> dict[str, Any]:
        """The JSON contract a tool-calling agent needs. No agent code lives in this repo."""
        return {
            "name": self.name,
            "description": self.summary,
            "parameters": {
                "type": "object",
                "properties": {p.name: p.as_json_schema() for p in self.parameters},
                "required": [
                    p.name for p in self.parameters if p.default is inspect.Parameter.empty
                ],
                "additionalProperties": False,
            },
        }


TOOL_REGISTRY: dict[str, ToolSpec] = {}

#: Parameters that are plumbing, not part of the agent-facing contract.
_HIDDEN_PARAMS = frozenset({"image", "analysis", "config", "debug", "strict"})

_PARAM_DOC = re.compile(r"^\s{4,8}(\w+)\s*:\s*(.+)$")


def _summary_from_doc(doc: str | None) -> str:
    """First paragraph of the docstring, collapsed to one line."""
    if not doc:
        return ""
    lines: list[str] = []
    for line in doc.strip().splitlines():
        stripped = line.strip()
        if not stripped:
            break
        lines.append(stripped)
    return " ".join(lines)


def _param_docs(doc: str | None) -> dict[str, str]:
    """Parse an ``Args:`` block into ``{name: description}``."""
    if not doc or "Args:" not in doc:
        return {}
    block = doc.split("Args:", 1)[1]
    for terminator in ("Returns:", "Raises:", "Notes:", "Example:"):
        block = block.split(terminator, 1)[0]
    out: dict[str, str] = {}
    current: str | None = None
    for line in block.splitlines():
        match = _PARAM_DOC.match(line)
        if match:
            current = match.group(1)
            out[current] = match.group(2).strip()
        elif current and line.strip():
            out[current] = f"{out[current]} {line.strip()}"
    return out


def register(fn: ToolFn) -> ToolFn:
    """Register a tool and derive its JSON contract from its signature and docstring."""
    signature = inspect.signature(fn)
    docs = _param_docs(fn.__doc__)
    specs: list[ParameterSpec] = []

    for name, parameter in signature.parameters.items():
        if name in _HIDDEN_PARAMS or parameter.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        annotation = parameter.annotation
        if annotation is bool:
            json_type = "boolean"
        elif annotation is int:
            json_type = "integer"
        else:
            json_type = "number"
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

    TOOL_REGISTRY[fn.__name__] = ToolSpec(
        name=fn.__name__,
        fn=fn,
        summary=_summary_from_doc(fn.__doc__),
        parameters=tuple(specs),
    )
    return fn


def tool_schemas() -> list[dict[str, Any]]:
    """Every registered tool as a JSON contract, ready for an LLM tool-calling loop."""
    return [TOOL_REGISTRY[name].as_contract() for name in sorted(TOOL_REGISTRY)]


def validate_strength(value: float, name: str = "strength") -> float:
    """Type- and range-check a strength. Raises rather than coercing or clamping silently.

    Strings are rejected even when they parse, and so is ``bool``. Accepting ``"0.5"`` would
    mean a caller could also pass ``"0,5"`` or ``"50%"`` and get a confusing failure much later;
    a future agent emitting JSON should send a number, not a string.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float, np.floating, np.integer)):
        raise StrengthOutOfRangeError(
            f"{name} must be a number, got {type(value).__name__} {value!r}"
        )
    try:
        strength = float(value)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
        raise StrengthOutOfRangeError(f"{name} must be a number, got {value!r}") from exc
    if not np.isfinite(strength):
        raise StrengthOutOfRangeError(f"{name} must be finite, got {strength}")
    if strength < 0.0 or strength > 1.0:
        raise StrengthOutOfRangeError(
            f"{name} must be in [0, 1], got {strength}. 1.0 already means the maximum "
            "intensity this engine considers natural."
        )
    return strength


def resolve_image(image: Any) -> np.ndarray:
    """Accept a ``PortraitImage`` or a raw array and return the canonical float32 RGB buffer."""
    from portrait_engine.core.image import PortraitImage, from_array

    if isinstance(image, PortraitImage):
        return image.rgb
    array = np.asarray(image)
    if array.dtype == np.float32 and array.ndim == 3 and array.shape[2] == 3:
        return array
    return from_array(array).rgb


def no_op(image: Any, tool: str, params: dict[str, Any], reason: str) -> Any:
    """A tool that decided to do nothing. Not an error: a normal, reported outcome."""
    from portrait_engine.core.types import ToolResult

    return ToolResult(
        image=resolve_image(image).copy(),
        tool=tool,
        params=params,
        stats={"reason": reason, "edited_area_fraction": 0.0},
        changed=False,
    )
