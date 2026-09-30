"""
orchestrator.py — the Coordination Layer required by the Theme 05 spec.

Before this file existed, the repo had two working but disconnected
halves:

  - agent/event_loop.py + state_manager.py + interrupt_handler.py: a real
    async controller that can receive events and cancel in-flight work,
    but never called the LLM.
  - agent/reasoner.py + llm_client.py + tools_builtin.py: a real,
    tested Ollama-backed reasoning loop (qwen2.5:14b by default), but only
    ever driven synchronously from demo/ollama_smoke.py, with no
    connection to interruption handling at all.

Orchestrator is the missing coordination layer. It:

  1. turns a completed user turn into a cancellable background
     asyncio.Task (registered with InterruptHandler under its own
     `turn_id`), running AsyncLLMReasoner — the async reasoning loop whose
     LLM call goes through `asyncio.to_thread` so a slow qwen2.5:14b call
     never blocks the event loop, and whose every tool call is ITSELF a
     separately cancellable, separately call_id'd unit of work registered
     with the same InterruptHandler (see agent/async_reasoner.py and
     agent/tools_builtin.ToolRunner.run_cancellable);
  2. emits a `tool_call` action to the output queue the instant a tool
     call is created — not after the whole turn finishes — satisfying
     "non-blocking tool calls (with explicit call_id)" as a first-class
     output type;
  3. emits `clarification_request` instead of `final` when the reasoner
     needed to ask the user something (e.g. a missing required argument)
     rather than answer;
  4. emits `final` with an attached StateSnapshot when the turn actually
     completes;
  5. guarantees a turn superseded by a later interruption never produces
     ANY of the above — dropped structurally, not tagged-and-trusted;
  6. optionally records every step to a TraceRecorder for replay/debugging
     (spec §6), and optionally registers dynamic tool manifests (spec
     §"Schema-Driven Tools") onto the same ToolRunner the reasoner uses.
"""

from __future__ import annotations

import asyncio
import itertools
import logging
from typing import Any, Dict, Optional

from agent.action_schema import ActionSchemaError, validate_action
from agent.async_reasoner import AsyncLLMReasoner
from agent.reasoner import ConversationHistory
from agent.tool_engine import ToolCall
from agent.tool_manifest import ManifestParseResult, register_manifest_tools
from agent.trace import TraceRecorder

logger = logging.getLogger(__name__)

_turn_ids = itertools.count(1)


def _next_turn_id() -> str:
    return f"turn_{next(_turn_ids)}"


class Orchestrator:
    """Bridges completed user turns to the LLM reasoning stack.

    Implements agent/event_loop.py's TurnHandlerProtocol
    (`async def start_turn(self, user_text: str) -> str`) and
    ManifestHandlerProtocol (`def register_manifest(self, manifest) -> ManifestParseResult`).
    """

    def __init__(
        self,
        reasoner: AsyncLLMReasoner,
        state_manager: Any,
        interrupt_handler: Any,
        output_queue: "asyncio.Queue[Dict[str, Any]]",
        history: Optional[ConversationHistory] = None,
        trace: Optional[TraceRecorder] = None,
        turn_timeout_seconds: float = 110.0,
    ) -> None:
        self.reasoner = reasoner
        self.state_manager = state_manager
        self.interrupt_handler = interrupt_handler
        self.output_queue = output_queue
        self.history = history or ConversationHistory()
        self.trace = trace
        # Spec §6: "120s wall-clock cap per scenario." The harness itself
        # presumably enforces that cap externally, but a reasoning turn
        # (an LLM call plus however many chained tool calls) has no
        # internal bound of its own otherwise — a hung Ollama connection
        # or a runaway tool-call loop would just block that scenario
        # forever from OUR side too. 110s leaves a 10s margin under the
        # spec's cap for our own cleanup + the final action to be emitted.
        self.turn_timeout_seconds = turn_timeout_seconds

    # ---- TurnHandlerProtocol -------------------------------------------

    async def start_turn(self, user_text: str) -> str:
        """Kick off reasoning for a completed user turn as a cancellable
        background task, and return immediately with its call_id.

        Registering with the InterruptHandler under this call_id is what
        makes the NEXT interruption (a barge-in, a correction) cancel this
        turn outright: InterruptHandler.cancel_all_tasks() will find it in
        `active_tasks` and cancel it just like any tool call.
        """
        call_id = _next_turn_id()
        self.history.add("user", user_text)
        if self.trace:
            self.trace.input("text_turn", call_id=call_id, text=user_text)
        task = asyncio.create_task(self._run_turn(call_id))
        self.interrupt_handler.register_task(call_id, task)
        return call_id

    # ---- ManifestHandlerProtocol ---------------------------------------

    def register_manifest(self, manifest: list[dict]) -> ManifestParseResult:
        """Parse+register a dynamic tool manifest onto the SAME ToolRunner
        the reasoner already uses, so a tool the agent has never seen
        before (an "unseen tool" scenario) becomes usable on the very next
        turn without any code change or restart.
        """
        result = register_manifest_tools(self.reasoner.runner, manifest)
        if self.trace:
            self.trace.record(
                "input",
                event_type="tool_manifest",
                registered=[s.name for s in result.specs],
                errors=result.errors,
            )
        if result.errors:
            logger.warning(
                "Tool manifest had %d invalid entries: %s", len(result.errors), result.errors
            )
        return result

    # ---- internals -------------------------------------------------

    async def _emit(self, action: Dict[str, Any]) -> None:
        """Validate against the canonical action schema (Protocol
        Compliance, spec §3.2.6) before an action ever reaches the output
        queue. A validation failure here is always a bug in OUR OWN code
        (we constructed the action ourselves) — logged loudly and the
        action is DROPPED rather than letting a malformed payload reach
        the wire, since Safety & Protocol is graded strictly from what's
        actually on the wire/trace.
        """
        try:
            validate_action(action)
        except ActionSchemaError:
            logger.exception("Refusing to emit malformed action: %r", action)
            if self.trace:
                self.trace.decision(
                    "action failed schema validation",
                    call_id=action.get("call_id"),
                    status="invalid",
                )
            return
        await self.output_queue.put(action)

    async def _on_tool_call(self, call: ToolCall) -> None:
        """Fired by AsyncLLMReasoner the instant a tool call is created —
        this is what makes tool_call a live, non-blocking output instead
        of something only visible after the whole turn finishes.
        """
        if self.trace:
            self.trace.action_emitted("tool_call", call_id=call.call_id, tool_name=call.tool_name)
        await self._emit(call.to_dict())

    async def _run_turn(self, turn_id: str) -> None:
        messages = self.history.snapshot()

        def is_current() -> bool:
            # AsyncLLMReasoner checks this before every LLM round-trip and
            # every tool call. As soon as an interruption clears this turn
            # out of active_tasks, the reasoner bails out at its own next
            # check point instead of finishing a now-pointless call.
            return turn_id in self.interrupt_handler.active_tasks

        try:
            result = await asyncio.wait_for(
                self.reasoner.respond(
                    messages, is_current=is_current, on_tool_call=self._on_tool_call
                ),
                timeout=self.turn_timeout_seconds,
            )
        except asyncio.TimeoutError:
            # asyncio.wait_for() has already cancelled the reasoner.respond()
            # coroutine internally (which cascades into cancelling whatever
            # tool call it was awaiting, via the same mechanism a real
            # interruption uses) — we just need to clean up OUR OWN
            # bookkeeping and tell the user something concrete instead of
            # leaving them hanging forever.
            self.interrupt_handler.unregister_task(turn_id)
            logger.warning("Turn %s exceeded %.2fs wall-clock timeout", turn_id, self.turn_timeout_seconds)
            if self.trace:
                self.trace.decision(
                    "exceeded wall-clock timeout", call_id=turn_id, status="timeout"
                )
            await self._emit(
                {
                    "action": "final",
                    "call_id": turn_id,
                    "response": "Sorry, that took too long to process. Please try again.",
                    "tool_trace": [],
                    "error": "timeout",
                    "state_snapshot": self.state_manager.get_snapshot().model_dump(),
                }
            )
            return
        except asyncio.CancelledError:
            if self.trace:
                self.trace.decision("interrupted mid-reasoning", call_id=turn_id, status="cancelled")
            raise
        except Exception:  # noqa: BLE001 - a reasoning bug must not crash the agent
            logger.exception("Reasoning turn %s failed", turn_id)
            self.interrupt_handler.unregister_task(turn_id)
            return

        # Check currency BEFORE unregistering — unregistering first would
        # make is_current() (which checks membership in active_tasks)
        # always report False, treating every legitimate result as stale.
        still_current = is_current()
        self.interrupt_handler.unregister_task(turn_id)

        if result.cancelled or not still_current:
            # Superseded by a later interruption. This is the structural
            # fix for "stale re-runs": we never even emit the result,
            # rather than emitting it tagged stale and trusting every
            # downstream consumer to check the tag.
            logger.info("Turn %s cancelled/superseded, dropping result", turn_id)
            if self.trace:
                self.trace.decision(
                    "superseded by later interruption", call_id=turn_id, status="cancelled"
                )
            return

        if result.clarification is not None:
            if self.trace:
                self.trace.action_emitted(
                    "clarification_request", call_id=turn_id, reason=result.clarification
                )
            await self._emit(
                {
                    "action": "clarification_request",
                    "call_id": turn_id,
                    "reason": result.clarification,
                }
            )
            return

        if result.error is None:
            self.history.add("assistant", result.text)

        if self.trace:
            self.trace.action_emitted("final", call_id=turn_id)

        await self._emit(
            {
                "action": "final",
                "call_id": turn_id,
                "response": result.text,
                "tool_trace": result.tool_trace,
                # Satisfies "final responses carrying structured State
                # Snapshots (intent and slot values)" from the spec.
                "state_snapshot": self.state_manager.get_snapshot().model_dump(),
            }
        )
