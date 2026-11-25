"""Tools the model can call, and the one place where calls are checked and run.

A tool is a Python function plus a JSON Schema for its arguments (the same format OpenAI-compatible APIs and
MCP use). `Registry.call` is the only path from a model's tool call to code, so every rule lives here:
1. the tool must exist and the arguments must match the schema (errors go back to the model, which fixes them);
2. an **outward** tool (one that changes something outside the agent: writes a file, commits, pushes, posts)
   runs only after the human approves that exact action. This is enforced in code, so no prompt, web page or
   model mistake can skip it;
3. output is capped, so one huge result can't blow the token budget.
"""

from __future__ import annotations

import inspect
import json
from dataclasses import dataclass, field
from typing import Any, Callable

from .approval import Action, Approver, DenyAll

MAX_OUTPUT_CHARS = 8000


class ToolError(Exception):
    """An expected failure (bad input, not found...). The message is shown to the model so it can recover."""


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict                      # JSON Schema of the arguments (an "object" schema)
    fn: Callable[..., Any]                # sync or async; called with the validated arguments as keywords
    outward: bool = False                 # needs human approval before it runs
    preview: Callable[[dict], Action] | None = None    # how to show the action to the human (outward tools)

    def schema(self) -> dict:
        return {"type": "function", "function": {"name": self.name, "description": self.description,
                                                 "parameters": self.parameters}}


@dataclass
class Outcome:
    text: str                             # what the model sees
    is_error: bool = False
    approved: bool | None = None          # None: not an outward tool
    data: Any = None                      # the raw return value (for traces and tests)


_TYPES = {"string": str, "integer": int, "number": (int, float), "boolean": bool, "array": list, "object": dict}


def validate(schema: dict, args: dict) -> dict:
    """A small JSON Schema check (types, required, enum, bounds, no unknown keys); fills defaults."""
    if not isinstance(args, dict):
        raise ToolError("arguments must be a JSON object")
    props = schema.get("properties", {})
    unknown = set(args) - set(props)
    if unknown:
        raise ToolError(f"unknown argument(s) {sorted(unknown)}; allowed: {sorted(props)}")
    missing = [k for k in schema.get("required", []) if k not in args]
    if missing:
        raise ToolError(f"missing required argument(s): {missing}")
    out = {}
    for key, spec in props.items():
        if key not in args:
            if "default" in spec:
                out[key] = spec["default"]
            continue
        value, kind = args[key], spec.get("type")
        if kind == "number" and isinstance(value, bool) or kind == "integer" and isinstance(value, bool):
            raise ToolError(f"{key} must be a {kind}")
        if kind == "integer" and isinstance(value, float) and value.is_integer():
            value = int(value)
        if kind in _TYPES and not isinstance(value, _TYPES[kind]):
            raise ToolError(f"{key} must be a {kind}, got {type(value).__name__}")
        if "enum" in spec and value not in spec["enum"]:
            raise ToolError(f"{key} must be one of {spec['enum']}")
        if "minimum" in spec and value < spec["minimum"] or "maximum" in spec and value > spec["maximum"]:
            raise ToolError(f"{key} must be between {spec.get('minimum')} and {spec.get('maximum')}")
        if kind == "string" and spec.get("minLength") and len(value.strip()) < spec["minLength"]:
            raise ToolError(f"{key} must not be empty")
        out[key] = value
    return out


def render(value: Any) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=1, default=str)
    if len(text) > MAX_OUTPUT_CHARS:
        text = text[:MAX_OUTPUT_CHARS] + f"\n...[truncated: {len(text) - MAX_OUTPUT_CHARS} more characters]"
    return text


@dataclass
class Registry:
    approver: Approver = field(default_factory=DenyAll)
    tools: dict[str, Tool] = field(default_factory=dict)

    def add(self, *tools: Tool) -> "Registry":
        for t in tools:
            if t.name in self.tools:
                raise ValueError(f"duplicate tool name {t.name!r}")
            self.tools[t.name] = t
        return self

    def schemas(self) -> list[dict]:
        return [t.schema() for t in self.tools.values()]

    async def call(self, name: str, raw_args: str | dict) -> Outcome:
        tool = self.tools.get(name)
        if tool is None:
            return Outcome(f"error: no tool named {name!r}. Available: {', '.join(self.tools)}", is_error=True)
        try:
            args = json.loads(raw_args or "{}") if isinstance(raw_args, str) else raw_args
            args = validate(tool.parameters, args)
        except json.JSONDecodeError as e:
            return Outcome(f"error: arguments are not valid JSON ({e})", is_error=True)
        except ToolError as e:
            return Outcome(f"error: {e}", is_error=True)

        approved = None
        if tool.outward:
            action = tool.preview(args) if tool.preview else Action(tool.name, f"{tool.name}({args})")
            approved = await self.approver.approve(action)
            if not approved:
                return Outcome("the user declined this action. Don't retry it; continue without it or explain what "
                               "you would have done.", approved=False)
        try:
            value = tool.fn(**args)
            if inspect.isawaitable(value):
                value = await value
        except ToolError as e:
            return Outcome(f"error: {e}", is_error=True, approved=approved)
        except Exception as e:                       # an unexpected bug: report it, don't crash the whole task
            return Outcome(f"error: {name} failed unexpectedly ({type(e).__name__}: {e})", is_error=True,
                           approved=approved)
        return Outcome(render(value), approved=approved, data=value)
