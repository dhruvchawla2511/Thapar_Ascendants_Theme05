"""
action_schema.py — Protocol Compliance (spec §3.2.6): "Emit well-formed
JSON payloads with valid snapshots and identifiers."

Before this module existed, every action dict (filler, tool_call, cancel,
clarification_request, final) was assembled ad hoc at each call site in
event_loop.py and orchestrator.py, with nothing checking that the shape
was actually consistent, that call_ids were real non-empty identifiers,
that a final action's state_snapshot had the fields the spec requires, or
that the payload could even be serialized to JSON at all. Safety &
Protocol is 10% of the scenario score and is graded "strictly on trace
logs" per the spec's scoring table — a malformed action there is a silent
deduction, not a crash we'd notice.

validate_action() is the single source of truth for what each action type
must contain. It's called from both EventProcessor.emit_action() (used by
the fast path and the controller's own interruption/clarification/manifest
paths) and Orchestrator's own action emission (tool_call/cancel/
clarification_request/final from the reasoning stack) — every action this
agent ever emits passes through it before reaching the output queue.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Dict

# The five output action types the spec's interface contract (§3.1) lists:
# "Spoken fillers, non-blocking tool calls (with explicit call_id),
# cancellations, clarification requests, and final responses carrying
# structured State Snapshots."
FILLER = "filler"
TOOL_CALL = "tool_call"
CANCEL = "cancel"
CLARIFICATION_REQUEST = "clarification_request"
FINAL = "final"

KNOWN_ACTIONS = frozenset({FILLER, TOOL_CALL, CANCEL, CLARIFICATION_REQUEST, FINAL})


class ActionSchemaError(ValueError):
    """An action dict does not conform to its type's canonical schema, or
    is not a well-formed JSON payload (spec §3.2.6: "Protocol Compliance").
    """


def _is_nonempty_str(value: Any) -> bool:
    return isinstance(value, str) and len(value) > 0


def _is_str(value: Any) -> bool:
    return isinstance(value, str)


def _is_dict(value: Any) -> bool:
    return isinstance(value, dict)


def _is_number(value: Any) -> bool:
    # bool is a subclass of int in Python; a timestamp/number field must
    # not silently accept True/False.
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_optional_str(value: Any) -> bool:
    return value is None or isinstance(value, str)


# Each action type's required fields and how to validate each one. Extra
# fields beyond these (e.g. "error": "timeout" on a final action, or
# "tool_trace") are always allowed — this validates the CONTRACT every
# consumer can rely on, not a closed/exhaustive shape.
_REQUIRED_FIELDS: Dict[str, Dict[str, Callable[[Any], bool]]] = {
    FILLER: {
        "call_id": _is_nonempty_str,
        "text": _is_str,
    },
    TOOL_CALL: {
        "call_id": _is_nonempty_str,
        "tool_name": _is_nonempty_str,
        "args": _is_dict,
    },
    CANCEL: {
        "call_id": _is_nonempty_str,
    },
    CLARIFICATION_REQUEST: {
        "reason": _is_str,
    },
    FINAL: {
        "call_id": _is_nonempty_str,
        "response": _is_str,
        "state_snapshot": _is_dict,
    },
}

# The spec explicitly names what a State Snapshot carries: "intent and
# slot values". last_updated is this codebase's own addition (a
# freshness/ordering signal) but is required here too since every
# state_snapshot this agent has ever emitted includes it — dropping it
# silently would be a regression worth catching.
_STATE_SNAPSHOT_FIELDS: Dict[str, Callable[[Any], bool]] = {
    "intent": _is_optional_str,
    "slots": _is_dict,
    "last_updated": _is_number,
}


def _validate_state_snapshot(snapshot: Any, action_name: str) -> None:
    if not isinstance(snapshot, dict):
        raise ActionSchemaError(f"{action_name}: state_snapshot must be an object")
    for field, checker in _STATE_SNAPSHOT_FIELDS.items():
        if field not in snapshot:
            raise ActionSchemaError(f"{action_name}: state_snapshot missing '{field}'")
        if not checker(snapshot[field]):
            raise ActionSchemaError(
                f"{action_name}: state_snapshot['{field}'] failed validation"
            )


def validate_action(action: Dict[str, Any]) -> None:
    """Raise ActionSchemaError if `action` violates its type's schema, is
    not JSON-serializable, or doesn't round-trip through JSON unchanged.
    Returns None (never mutates the input) on success.
    """
    if not isinstance(action, dict):
        raise ActionSchemaError(f"action must be a dict, got {type(action).__name__}")

    name = action.get("action")
    if not _is_nonempty_str(name):
        raise ActionSchemaError("action dict missing a non-empty 'action' field")
    if name not in KNOWN_ACTIONS:
        raise ActionSchemaError(f"unknown action type: {name!r}")

    for field, checker in _REQUIRED_FIELDS[name].items():
        if field not in action:
            raise ActionSchemaError(f"{name}: missing required field '{field}'")
        if not checker(action[field]):
            raise ActionSchemaError(f"{name}: field '{field}' failed validation")

    if name == FINAL:
        _validate_state_snapshot(action["state_snapshot"], name)

    # "Well-formed JSON payloads" (spec §3.2.6), taken literally: the
    # WHOLE action must actually serialize (allow_nan=False rejects
    # NaN/Infinity, which json.dumps accepts by default but not every
    # JSON parser understands), and must round-trip through
    # json.loads(json.dumps(...)) back to an identical structure.
    try:
        encoded = json.dumps(action, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ActionSchemaError(f"{name}: action is not JSON-serializable: {exc}") from exc
    if json.loads(encoded) != action:
        raise ActionSchemaError(f"{name}: action does not round-trip through JSON unchanged")


def is_valid_action(action: Dict[str, Any]) -> bool:
    """Non-raising convenience wrapper around validate_action()."""
    try:
        validate_action(action)
    except ActionSchemaError:
        return False
    return True
