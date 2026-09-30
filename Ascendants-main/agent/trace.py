"""
trace.py — structured event/action trace logging (spec §6).

A flat, append-only, in-memory log of everything that happened in a
session: every input event handled, every action emitted, every state
change, every cancellation/staleness decision, with enough fields to
reconstruct "what happened and why" without re-running anything. This is
what lets a scenario be replayed/debugged after the fact, independent of
the evaluation harness's own trace logs.

Deliberately dependency-free (stdlib only) and synchronous — recording a
trace entry must never itself be a source of latency or a reason a real
event gets dropped.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from agent.clock import REAL_CLOCK, Clock


@dataclass(frozen=True)
class TraceEvent:
    timestamp: float
    session_id: Optional[str]
    kind: str  # "input" | "action" | "state" | "decision"
    call_id: Optional[str] = None
    event_type: Optional[str] = None  # e.g. "text_chunk", "interruption"
    action: Optional[str] = None  # e.g. "tool_call", "cancel", "final"
    status: Optional[str] = None  # e.g. "pending", "cancelled", "completed"
    reason: Optional[str] = None  # human-readable, e.g. "superseded by turn_3"
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TraceRecorder:
    """Session-scoped trace log. One instance per session — never shared
    across sessions, matching StateManager's own no-global-cache rule.
    """

    def __init__(self, session_id: Optional[str] = None, clock: Clock = REAL_CLOCK) -> None:
        self.session_id = session_id
        self._clock = clock
        self._events: list[TraceEvent] = []

    def record(
        self,
        kind: str,
        *,
        call_id: Optional[str] = None,
        event_type: Optional[str] = None,
        action: Optional[str] = None,
        status: Optional[str] = None,
        reason: Optional[str] = None,
        **detail: Any,
    ) -> TraceEvent:
        entry = TraceEvent(
            timestamp=self._clock.now(),
            session_id=self.session_id,
            kind=kind,
            call_id=call_id,
            event_type=event_type,
            action=action,
            status=status,
            reason=reason,
            detail=detail,
        )
        self._events.append(entry)
        return entry

    # ---- convenience wrappers, one per `kind` ----------------------

    def input(self, event_type: str, **detail: Any) -> TraceEvent:
        return self.record("input", event_type=event_type, **detail)

    def action_emitted(self, action: str, call_id: Optional[str] = None, **detail: Any) -> TraceEvent:
        return self.record("action", action=action, call_id=call_id, **detail)

    def state_change(self, **detail: Any) -> TraceEvent:
        return self.record("state", **detail)

    def decision(self, reason: str, call_id: Optional[str] = None, status: Optional[str] = None, **detail: Any) -> TraceEvent:
        """A staleness/cancellation/duplicate-guard decision — the entries
        that answer "why did/didn't this result get used".
        """
        return self.record("decision", call_id=call_id, status=status, reason=reason, **detail)

    # ---- inspection / export ----------------------------------------

    def events(self) -> list[TraceEvent]:
        return list(self._events)

    def for_call(self, call_id: str) -> list[TraceEvent]:
        return [e for e in self._events if e.call_id == call_id]

    def to_jsonl(self) -> str:
        return "\n".join(json.dumps(e.to_dict(), default=str) for e in self._events)

    def clear(self) -> None:
        self._events.clear()
