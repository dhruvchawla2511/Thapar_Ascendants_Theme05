"""
async_reasoner.py — native-async counterpart to reasoner.LLMReasoner.

reasoner.LLMReasoner is fully synchronous (it blocks on both the LLM HTTP
call and on ToolRunner.run()'s handler execution) and is kept exactly as
it was — demo/ollama_smoke.py and its whole test suite depend on that, and
nothing about it was wrong; it just cannot give an individual TOOL CALL
its own cancellable identity, because the whole respond() call is one
opaque blocking unit handed to a single worker thread.

AsyncLLMReasoner exists for that one reason: making a tool call — not just
the whole reasoning turn — a first-class, separately call_id'd, separately
cancellable, stale-result-protected unit of work (spec §3.2.2: "a tool
call must have explicit call_id, lifecycle state, cancellation support,
stale-result protection").

It reuses the exact same system prompt / JSON action grammar / parsing
(`reasoner.SYSTEM_PROMPT_TEMPLATE`, `reasoner.parse_action`, `reasoner.
RETRY_NUDGE`) and the exact same ToolEngine fingerprint/duplicate-
protection (via `tools_builtin.ToolRunner.run_cancellable`) — only the
control flow around the LLM call and tool execution is different: both
now run as real `await`s on the event loop (the LLM HTTP call via
`asyncio.to_thread`, each tool handler via its own cancellable
asyncio.Task) instead of one big blocking function on a single thread.

Known limitation (also noted in tools_builtin.ToolRunner.run_cancellable
and in AGENTS.md): because a tool call is awaited synchronously within its
turn's coroutine, cancelling ONE tool call's call_id while its parent turn
keeps reasoning is not supported today — cancelling a tool call cancels
its containing turn too, same as cancelling the turn directly. What IS
delivered and tested: explicit per-call lifecycle distinct from the turn's
own call_id, immediate `tool_call`/`cancel` action emission with the
correct call_id, and stale-result protection — a handler result that
finishes computing after cancellation is inspected against
ToolEngine's CallStatus before anything (including StateManager slots)
is allowed to use it.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional

from agent.llm_client import LLMClient, LLMError
from agent.reasoner import RETRY_NUDGE, SYSTEM_PROMPT_TEMPLATE, parse_action
from agent.tool_engine import CallStatus, ToolCall
from agent.tools_builtin import ToolRunner

logger = logging.getLogger(__name__)

IsCurrent = Callable[[], bool]
OnToolCall = Callable[[ToolCall], Awaitable[None]]


@dataclass
class AsyncReasonResult:
    text: str = ""
    tool_trace: list[dict] = field(default_factory=list)
    error: Optional[str] = None
    cancelled: bool = False
    # Set (non-None) instead of `text` when the agent should ask the user
    # something rather than answer — e.g. a tool call was missing a
    # required argument. The caller (Orchestrator) turns this into a
    # `clarification_request` action instead of a `final` one.
    clarification: Optional[str] = None


class AsyncLLMReasoner:
    def __init__(
        self,
        llm: LLMClient,
        runner: ToolRunner,
        interrupt_handler: Any,
        state_manager: Any,
        max_tool_iterations: int = 4,
    ) -> None:
        self.llm = llm
        self.runner = runner
        self.interrupt_handler = interrupt_handler
        self.state_manager = state_manager
        self.max_tool_iterations = max_tool_iterations

    def system_prompt(self) -> str:
        return SYSTEM_PROMPT_TEMPLATE.replace("<<TOOLS>>", self.runner.describe())

    async def respond(
        self,
        messages: list[dict],
        is_current: Optional[IsCurrent] = None,
        on_tool_call: Optional[OnToolCall] = None,
    ) -> AsyncReasonResult:
        """`messages` = conversation so far, ending with the current user
        message. Mirrors LLMReasoner.respond()'s contract and control flow
        exactly, just natively async.
        """
        current = is_current or (lambda: True)
        convo: list[dict] = [{"role": "system", "content": self.system_prompt()}, *messages]
        trace: list[dict] = []
        tools_used = 0
        retried = False

        while True:
            if not current():
                return AsyncReasonResult(tool_trace=trace, cancelled=True)
            try:
                raw = await asyncio.to_thread(self.llm.chat, convo, json_mode=True)
            except LLMError as exc:
                return AsyncReasonResult(
                    text=f"Sorry, I couldn't get an answer from the language model: {exc}",
                    tool_trace=trace,
                    error=str(exc),
                )
            if not current():
                return AsyncReasonResult(tool_trace=trace, cancelled=True)

            action = parse_action(raw)
            if action is None:
                if retried:
                    return AsyncReasonResult(
                        text=(raw or "").strip() or "Sorry, I have no answer.", tool_trace=trace
                    )
                retried = True
                convo += [
                    {"role": "assistant", "content": raw},
                    {"role": "user", "content": RETRY_NUDGE},
                ]
                continue

            if action.kind == "final":
                return AsyncReasonResult(text=action.response, tool_trace=trace)

            # ---- tool request ----
            if tools_used >= self.max_tool_iterations:
                return AsyncReasonResult(
                    text="Sorry, I couldn't finish that within the allowed number of tool steps.",
                    tool_trace=trace,
                    error="max tool iterations reached",
                )
            if not current():
                return AsyncReasonResult(tool_trace=trace, cancelled=True)

            outcome = await self.runner.run_cancellable(
                action.tool,
                action.arguments,
                self.interrupt_handler,
                on_call_created=on_tool_call,
            )
            tools_used += 1
            trace.append(
                {
                    "tool": action.tool,
                    "arguments": action.arguments,
                    "ok": outcome.ok,
                    "result": outcome.text,
                    "call_id": outcome.call_id,
                }
            )

            if outcome.needs_clarification:
                # A required argument was missing/invalid. Ask the user
                # instead of letting the model keep guessing at arguments.
                return AsyncReasonResult(
                    tool_trace=trace,
                    clarification=(
                        f"I need a bit more information to use '{action.tool}': {outcome.text}"
                    ),
                )

            if not current():
                if outcome.call_id:
                    self.runner.mark_stale(outcome.call_id)
                return AsyncReasonResult(tool_trace=trace, cancelled=True)

            # Stale-result protection + state application: only a call
            # that is genuinely COMPLETED (never CANCELLED/STALE) may
            # touch session state, checked against the engine's own
            # lifecycle status rather than trusting the local `outcome.ok`
            # flag alone.
            if outcome.ok and outcome.call_id:
                call = self.runner.engine.get_call(outcome.call_id)
                if call.status == CallStatus.COMPLETED:
                    spec = self.runner.get_spec(action.tool)
                    if spec is not None and spec.slot_updates is not None:
                        try:
                            slots = spec.slot_updates(call.args, call.result)
                        except Exception:  # noqa: BLE001 - a bad slot_updates fn must not crash
                            logger.exception("slot_updates failed for tool %s", action.tool)
                            slots = None
                        if slots:
                            self.state_manager.update_slots(slots)

            label = "TOOL_RESULT" if outcome.ok else "TOOL_ERROR"
            convo += [
                {"role": "assistant", "content": raw},
                {
                    "role": "user",
                    "content": f"{label} {action.tool}: {outcome.text}\n"
                    "Now reply with a final JSON action using this result"
                    + ("." if outcome.ok else ", or fix the arguments and call the tool again."),
                },
            ]
