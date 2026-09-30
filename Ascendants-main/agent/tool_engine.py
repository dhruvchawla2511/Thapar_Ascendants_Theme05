"""
tool_engine.py — dynamic tool registration and safe call tracking.

This is the piece that turns "the agent wants to search for flights" into
a concrete, trackable, cancellable unit of work with a unique call_id.

Key responsibilities (per the hackathon spec):
- schema-driven tool registration (name, description, parameters,
  read-only vs state-changing)
- every call gets a unique call_id
- argument validation against the tool's declared schema
- tracking call lifecycle: pending -> completed / cancelled / stale
- preventing accidental duplicate state-changing operations

This module is deliberately deterministic and has no external
dependencies — it does not actually call any real APIs. Tool *execution*
(the async work of actually calling an API) belongs to the slow path /
controller; this module models the call and its safety rules.
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


class ToolEngineError(Exception):
    """Base class for tool engine errors."""


class UnknownToolError(ToolEngineError):
    pass


class InvalidArgumentsError(ToolEngineError):
    pass


class DuplicateStateChangingCallError(ToolEngineError):
    """Raised when a state-changing call is requested while an identical
    (tool_name, args) call is already pending — this is the guard against
    e.g. accidentally booking the same flight twice.
    """


class CallStatus(str, Enum):
    PENDING = "pending"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    STALE = "stale"  # completed, but superseded by a newer call — ignore it


@dataclass(frozen=True)
class ParamSpec:
    """Describes one parameter a tool accepts."""

    name: str
    type: type
    required: bool = True


@dataclass(frozen=True)
class ToolSpec:
    """Schema for a single tool — what the tool engine needs to know to
    validate calls to it and to reason about its safety properties.
    """

    name: str
    description: str
    parameters: tuple[ParamSpec, ...]
    state_changing: bool  # False = read-only (e.g. search), True = mutates
    # real callable is optional; None means "not wired up yet" which is
    # fine for now since we're modeling call tracking, not execution.
    handler: Callable[..., Any] | None = None
    # Optional: given (validated args, handler result), return a dict of
    # session slots to merge into StateManager after a successful, still-
    # current completion. Lets a tool's effect on session state be declared
    # data (checked by the caller against staleness) instead of a tool
    # reaching into StateManager itself. None = this tool doesn't touch
    # session slots (e.g. calculator).
    slot_updates: Callable[[dict[str, Any], Any], dict[str, Any]] | None = None

    def validate(self, args: dict[str, Any]) -> None:
        for param in self.parameters:
            if param.required and param.name not in args:
                raise InvalidArgumentsError(
                    f"Missing required argument '{param.name}' for tool '{self.name}'"
                )
            if param.name in args and not isinstance(args[param.name], param.type):
                raise InvalidArgumentsError(
                    f"Argument '{param.name}' for tool '{self.name}' must be "
                    f"{param.type.__name__}, got {type(args[param.name]).__name__}"
                )
        unknown = set(args) - {p.name for p in self.parameters}
        if unknown:
            raise InvalidArgumentsError(
                f"Unknown arguments for tool '{self.name}': {sorted(unknown)}"
            )


@dataclass
class ToolCall:
    """A single tracked invocation of a tool."""

    call_id: str
    tool_name: str
    args: dict[str, Any]
    status: CallStatus = CallStatus.PENDING
    created_at: float = field(default_factory=time.monotonic)
    result: Any = None

    def to_dict(self) -> dict[str, Any]:
        """Matches the wire format shown in the spec, e.g.:
        {"action": "tool_call", "call_id": ..., "tool_name": ..., "args": {...}}
        """
        return {
            "action": "tool_call",
            "call_id": self.call_id,
            "tool_name": self.tool_name,
            "args": self.args,
        }


def _fingerprint(tool_name: str, args: dict[str, Any]) -> tuple:
    return (tool_name, tuple(sorted(args.items())))


class ToolEngine:
    """Registers tools and tracks calls to them safely."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}
        self._calls: dict[str, ToolCall] = {}
        self._id_counter = itertools.count(1)
        # fingerprints of state-changing calls that are still pending
        self._pending_state_changing_fingerprints: set[tuple] = set()

    # ---- registration ----------------------------------------------

    def register_tool(self, spec: ToolSpec) -> None:
        self._tools[spec.name] = spec

    def get_tool(self, tool_name: str) -> ToolSpec:
        if tool_name not in self._tools:
            raise UnknownToolError(f"No such tool: '{tool_name}'")
        return self._tools[tool_name]

    def is_state_changing(self, tool_name: str) -> bool:
        return self.get_tool(tool_name).state_changing

    # ---- call lifecycle ----------------------------------------------

    def create_call(self, tool_name: str, args: dict[str, Any]) -> ToolCall:
        """Validate args and create a new tracked call with a unique
        call_id. Raises DuplicateStateChangingCallError if an identical
        state-changing call is already pending (e.g. calling
        book_flight(destination='Mumbai') twice before the first finishes).
        """
        spec = self.get_tool(tool_name)
        spec.validate(args)

        if spec.state_changing:
            fp = _fingerprint(tool_name, args)
            if fp in self._pending_state_changing_fingerprints:
                raise DuplicateStateChangingCallError(
                    f"A state-changing call to '{tool_name}' with the same "
                    f"arguments is already pending."
                )
            self._pending_state_changing_fingerprints.add(fp)

        call_id = f"call_{next(self._id_counter)}"
        call = ToolCall(call_id=call_id, tool_name=tool_name, args=dict(args))
        self._calls[call_id] = call
        return call

    def cancel_call(self, call_id: str) -> None:
        """Mark a call as cancelled — used when the user interrupts and
        the old call's eventual result must not be applied.
        """
        call = self._require_call(call_id)
        if call.status == CallStatus.PENDING:
            call.status = CallStatus.CANCELLED
        self._release_fingerprint(call)

    def complete_call(self, call_id: str, result: Any) -> ToolCall:
        """Record a result for a call. If the call was already cancelled,
        the result is stored but the status stays CANCELLED (i.e. this is
        exactly how a stale result gets ignored: the controller should
        check `call.status` before acting on `result`).
        """
        call = self._require_call(call_id)
        if call.status == CallStatus.CANCELLED:
            # Stale result arriving after cancellation — keep it for
            # inspection/logging, but do not resurrect the call.
            call.result = result
            return call
        call.status = CallStatus.COMPLETED
        call.result = result
        self._release_fingerprint(call)
        return call

    def mark_stale(self, call_id: str) -> None:
        """Explicitly mark a completed call's result as stale (superseded
        by a newer call) so the controller knows not to use it, even
        though the call itself finished successfully.
        """
        call = self._require_call(call_id)
        call.status = CallStatus.STALE
        self._release_fingerprint(call)

    def get_call(self, call_id: str) -> ToolCall:
        return self._require_call(call_id)

    def pending_calls(self) -> list[ToolCall]:
        return [c for c in self._calls.values() if c.status == CallStatus.PENDING]

    # ---- internals ----------------------------------------------

    def _require_call(self, call_id: str) -> ToolCall:
        if call_id not in self._calls:
            raise ToolEngineError(f"No such call_id: '{call_id}'")
        return self._calls[call_id]

    def _release_fingerprint(self, call: ToolCall) -> None:
        spec = self._tools.get(call.tool_name)
        if spec and spec.state_changing:
            fp = _fingerprint(call.tool_name, call.args)
            self._pending_state_changing_fingerprints.discard(fp)
