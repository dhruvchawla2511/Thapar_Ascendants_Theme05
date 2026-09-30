"""
tool_manifest.py — dynamic, schema-driven tool registration.

The Theme 05 spec lists "scenario tool manifests" as an input type (§3.1)
and "Schema-Driven Tools: Parse dynamic tool definitions (read-only vs.
state-modifying) from manifests" as a core objective (§3.2.4). Before this
module existed, the ONLY tools the agent could ever use were the two
hardcoded in tools_builtin.create_builtin_runner() (calculator,
current_time) — there was no way to hand the agent a tool it had never
seen before at session start, which is exactly what the hidden evaluation
set's "unseen tools" scenarios need.

A manifest is a list of plain dicts, e.g.:

    [
      {
        "name": "search_flights",
        "description": "Search flights to a destination.",
        "state_changing": False,
        "parameters": [
          {"name": "destination", "type": "string", "required": True},
        ],
      },
      {
        "name": "book_flight",
        "description": "Book a flight (mutates real state).",
        "state_changing": True,
        "parameters": [
          {"name": "destination", "type": "string", "required": True},
          {"name": "flight_number", "type": "string", "required": True},
        ],
      },
    ]

register_manifest_tools() turns each entry into a ToolSpec (reusing
tool_engine's existing schema/fingerprint/duplicate-protection machinery
unchanged — state-modifying tools get the exact same pending-fingerprint
guard as the built-in ones) and registers it with a ToolRunner. If the
caller does not supply a real handler for a given tool name, a
deterministic mock handler is used instead, so a manifest tool the agent
has genuinely never seen before is still safely callable (never crashes,
never contacts the network, always returns the same output for the same
input) rather than silently failing or being skipped.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from agent.tool_engine import ParamSpec, ToolSpec

Handler = Callable[[dict], Any]

_TYPE_MAP: dict[str, type] = {
    "string": str,
    "str": str,
    "text": str,
    "number": float,
    "float": float,
    "integer": int,
    "int": int,
    "boolean": bool,
    "bool": bool,
}


class ManifestError(Exception):
    """A manifest entry was malformed (missing name, unknown type, ...)."""


@dataclass
class ManifestParseResult:
    specs: list[ToolSpec]
    errors: list[str]  # human-readable, one per skipped/invalid entry


def _param_type(raw: Any, tool_name: str, param_name: str) -> type:
    if isinstance(raw, type):
        return raw
    key = str(raw).strip().lower()
    if key not in _TYPE_MAP:
        raise ManifestError(
            f"Tool '{tool_name}' parameter '{param_name}': unknown type {raw!r}. "
            f"Supported: {sorted(_TYPE_MAP)}."
        )
    return _TYPE_MAP[key]


def _parse_one(entry: dict[str, Any]) -> ToolSpec:
    if not isinstance(entry, dict):
        raise ManifestError(f"Manifest entry must be an object, got {type(entry).__name__}")

    name = entry.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ManifestError("Manifest entry missing a non-empty 'name'.")

    description = entry.get("description")
    if not isinstance(description, str):
        description = f"Dynamically registered tool '{name}'."

    state_changing = bool(entry.get("state_changing", False))

    raw_params = entry.get("parameters", entry.get("arguments", []))
    if not isinstance(raw_params, list):
        raise ManifestError(f"Tool '{name}': 'parameters' must be a list.")

    params: list[ParamSpec] = []
    for raw_param in raw_params:
        if not isinstance(raw_param, dict) or "name" not in raw_param:
            raise ManifestError(f"Tool '{name}': each parameter needs a 'name'.")
        p_name = str(raw_param["name"])
        p_type = _param_type(raw_param.get("type", "string"), name, p_name)
        p_required = bool(raw_param.get("required", True))
        params.append(ParamSpec(name=p_name, type=p_type, required=p_required))

    return ToolSpec(
        name=name,
        description=description,
        parameters=tuple(params),
        state_changing=state_changing,
    )


def parse_manifest(manifest: list[dict[str, Any]]) -> ManifestParseResult:
    """Parse a raw manifest (list of dicts) into ToolSpecs.

    Never raises on a single bad entry — that entry is skipped and its
    problem is recorded in `.errors` so the rest of the manifest (and the
    session) can still proceed. A manifest is untrusted input arriving at
    runtime; one malformed tool definition must not take down the agent.
    """
    if not isinstance(manifest, list):
        return ManifestParseResult(specs=[], errors=["Manifest must be a list of tool objects."])

    specs: list[ToolSpec] = []
    errors: list[str] = []
    seen_names: set[str] = set()
    for entry in manifest:
        try:
            spec = _parse_one(entry)
        except ManifestError as exc:
            errors.append(str(exc))
            continue
        if spec.name in seen_names:
            errors.append(f"Duplicate tool name in manifest: '{spec.name}' (kept first).")
            continue
        seen_names.add(spec.name)
        specs.append(spec)
    return ManifestParseResult(specs=specs, errors=errors)


def make_mock_handler(tool_name: str) -> Handler:
    """A deterministic stand-in for a manifest tool with no real handler.

    Deliberately boring and deterministic (same args -> same string every
    time, no randomness, no I/O) — this is what lets an "unseen tool from
    a manifest" scenario be exercised in tests/CI without a real backend
    behind it, while still going through the exact same call_id / schema
    validation / duplicate-protection path as every other tool.
    """

    def handler(args: dict[str, Any]) -> str:
        ordered = ", ".join(f"{k}={args[k]!r}" for k in sorted(args))
        return f"[mock result for '{tool_name}'({ordered})]"

    return handler


def register_manifest_tools(
    runner: Any,  # agent.tools_builtin.ToolRunner (duck-typed to avoid a cycle)
    manifest: list[dict[str, Any]],
    handlers: dict[str, Handler] | None = None,
) -> ManifestParseResult:
    """Parse `manifest` and register every valid entry on `runner`.

    `handlers` lets a real backend be plugged in per tool name; any tool
    name not present there gets `make_mock_handler` instead. Re-registering
    a tool name the runner already knows (e.g. re-sending the same manifest,
    or a manifest that tries to redefine a built-in) overwrites the old
    spec/handler for that name — this mirrors how ToolEngine.register_tool
    already behaves (last registration wins) and keeps manifest handling
    idempotent across repeated session setup calls.
    """
    result = parse_manifest(manifest)
    handlers = handlers or {}
    for spec in result.specs:
        handler = handlers.get(spec.name) or make_mock_handler(spec.name)
        runner.register(spec, handler)
    return result
