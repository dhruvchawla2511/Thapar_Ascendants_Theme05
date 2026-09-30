"""
main.py

Main execution entry point for the agent engine. Wires together
StateManager, InterruptHandler, EventProcessor, and the Coordination
Layer (Orchestrator + AsyncLLMReasoner + ToolRunner) behind an
AgentRunner, and provides a setup() warm-up hook for pre-loading
models/schemas before the engine starts serving traffic.

Two ways to run the LLM side:
  - use_llm=True  (--llm on the command line): talks to a real Ollama
    server running qwen2.5:14b (agent/llm_client.py's default model).
  - use_llm=False (the default): uses DeterministicDemoLLM, a small
    scripted stand-in for the model ONLY — every other component
    (EventProcessor, InterruptHandler, StateManager, Orchestrator,
    AsyncLLMReasoner, ToolRunner/ToolEngine) is the exact same real code
    that runs in production. This is what lets the full interruption /
    cancellation / stale-result pipeline be demonstrated end-to-end
    without a network connection or a running Ollama server.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from typing import Any, Dict

from agent.async_reasoner import AsyncLLMReasoner
from agent.clock import REAL_CLOCK, Clock
from agent.event_loop import EventProcessor
from agent.fault_injection import FaultPlan, apply_fault_plan
from agent.interrupt_handler import InterruptHandler
from agent.llm_client import OllamaClient
from agent.orchestrator import Orchestrator
from agent.state_manager import StateManager
from agent.tool_engine import ParamSpec, ToolSpec
from agent.tools_builtin import ToolExecutionError, create_builtin_runner
from agent.trace import TraceRecorder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_WARMUP_SECONDS = 300.0


async def setup(warmup_seconds: float = DEFAULT_WARMUP_SECONDS) -> None:
    """
    Warm-up hook, run once before the engine begins serving.

    In production this is where you'd pre-load models, tool schemas,
    embedding indexes, etc. so the first real request doesn't pay that
    latency. Here it's represented as a bounded wait; replace the body
    with real loading calls (and keep it non-blocking / async) as those
    components come online.
    """
    logger.info("Starting warm-up (%.0fs)...", warmup_seconds)
    start = time.monotonic()

    # Placeholder for real warm-up work, e.g.:
    #   await load_models()
    #   await preload_tool_schemas()
    await asyncio.sleep(warmup_seconds)

    elapsed = time.monotonic() - start
    logger.info("Warm-up complete in %.1fs", elapsed)


class _FastPathAdapter:
    """Adapter between Dhruv's FastPath API and the controller protocol.

    FastPath produces the user-facing acknowledgement with `acknowledge()`,
    while EventProcessor consumes a generic `handle_text_chunk()` interface.
    """

    def __init__(self) -> None:
        from agent.fast_path import FastPath
        self.fast_path = FastPath()

    async def handle_text_chunk(
        self, text: str, end_of_turn: bool, **kwargs: Any
    ) -> str | None:
        message = self.fast_path.acknowledge(text)
        logger.info("[fast_path] %s", message)
        return message


class DeterministicDemoLLM:
    """A scripted, fully offline stand-in for the real Ollama client.

    Implements the exact same `chat(messages, *, json_mode=False) -> str`
    protocol as `agent.llm_client.LLMClient`/`OllamaClient`, so it plugs
    into AsyncLLMReasoner without AsyncLLMReasoner (or anything else in
    the controller) knowing or caring that it isn't talking to Ollama.
    This is a stand-in for the MODEL only — the reasoning loop, tool
    execution, cancellation, and state application are all the real code.

    Behavior: looks at the latest genuine user message (skipping
    TOOL_RESULT/TOOL_ERROR follow-ups and the retry nudge) for a
    destination city and a booking intent; if found, calls book_flight;
    once a TOOL_RESULT/TOOL_ERROR is in the conversation, answers with a
    final response summarizing it. Deterministic — same input always
    produces the same output, no randomness, no I/O.
    """

    def __init__(self) -> None:
        self.calls = 0

    def chat(self, messages: list[dict], *, json_mode: bool = False) -> str:
        self.calls += 1
        last_tool_result = None
        last_user_text = ""
        for m in messages:
            if m.get("role") != "user":
                continue
            content = m.get("content", "")
            if content.startswith("TOOL_RESULT") or content.startswith("TOOL_ERROR"):
                last_tool_result = content
            elif "valid JSON action" not in content:  # skip the retry nudge
                last_user_text = content

        if last_tool_result is not None:
            if "lookup_manual:" in last_tool_result:
                # Chain: after looking up the manual, escalate to a
                # support ticket — demonstrates chained tool calls AND
                # exercises both new mock-environment tool categories in
                # one flow (spec §4: "...ticket creation, and frame-
                # grounded manual lookups").
                return json.dumps(
                    {
                        "action": "tool",
                        "tool": "create_ticket",
                        "arguments": {
                            "subject": "Washer error E4 - drainage blockage",
                            "priority": "normal",
                        },
                    }
                )
            summary = last_tool_result.split(":", 1)[-1].split("\n")[0].strip()
            return json.dumps({"action": "final", "response": f"Done — {summary}"})

        text = last_user_text.lower()
        if "washer" in text and ("e4" in text or "error" in text):
            return json.dumps(
                {
                    "action": "tool",
                    "tool": "lookup_manual",
                    "arguments": {"device": "washer", "query": "E4"},
                }
            )
        destination = None
        for city in ("delhi", "mumbai", "goa", "pune"):
            if city in text:
                destination = city.title()
                break

        if destination and any(k in text for k in ("book", "flight", "actually", "correction")):
            return json.dumps(
                {"action": "tool", "tool": "book_flight", "arguments": {"destination": destination}}
            )
        if "weather" in text:
            # Deliberately references a tool this LLM was never told about
            # in its examples — exercises a manifest-registered unseen tool.
            city = destination or "Delhi"
            return json.dumps(
                {"action": "tool", "tool": "current_weather", "arguments": {"city": city}}
            )
        return json.dumps({"action": "final", "response": "Okay!"})


DEMO_TOOL_DELAY_SECONDS = 0.15


class AgentRunner:
    """
    Owns the engine's core components and drives the event loop.

    Usage:
        runner = AgentRunner()
        runner.start()           # non-blocking: schedules the loop task
        await runner.submit(event)
        ...
        await runner.stop()
    """

    def __init__(
        self,
        use_llm: bool = True,
        session_id: str | None = None,
        slow_demo_tools: bool = False,
        turn_timeout_seconds: float = 110.0,
        clock: Clock = REAL_CLOCK,
    ) -> None:
        self.session_id = session_id or f"session_{uuid.uuid4().hex[:8]}"
        self.input_queue: "asyncio.Queue[Dict[str, Any]]" = asyncio.Queue()
        self.output_queue: "asyncio.Queue[Dict[str, Any]]" = asyncio.Queue()

        # A single Clock shared by every session-scoped component below
        # (StateManager, TraceRecorder, EventProcessor's perception
        # timestamps). Defaults to RealClock (wall-clock time, identical
        # to calling time.time() directly); pass a VirtualClock to run
        # this whole session against a deterministic, controllable clock
        # instead — e.g. for a replay harness (spec §4: "Virtual Clock
        # Streaming Harness") or a fully sleep-free test.
        self.clock = clock

        # StateManager/TraceRecorder are explicitly per-session objects —
        # never a module-level global — so two AgentRunners never share
        # state (spec §6: "State Scope: Session-scoped memory only, no
        # cross-session caching").
        self.state_manager = StateManager(session_id=self.session_id, clock=self.clock)
        self.interrupt_handler = InterruptHandler()
        self.fast_path = _FastPathAdapter()
        self.trace = TraceRecorder(session_id=self.session_id, clock=self.clock)

        llm = OllamaClient.from_env() if use_llm else DeterministicDemoLLM()

        # All four of the spec's named Mock Environment tool categories
        # (§4: "flight search, booking, ticket creation, and frame-
        # grounded manual lookups").
        tool_runner = create_builtin_runner(
            include_flight_demo_tools=True, include_troubleshooting_demo_tools=True
        )
        if slow_demo_tools:
            # Deterministic latency injection (spec §4), via the reusable
            # wrapper instead of a one-off hand-rolled slow handler: gives
            # the offline demo a real, reproducible window during which an
            # interruption can land mid-flight — a live illustration of the
            # same cancellation path the unittest suite proves with
            # threading.Event-controlled fakes (tests/test_scenarios.py).
            apply_fault_plan(
                tool_runner,
                "book_flight",
                FaultPlan.latency_then_normal(DEMO_TOOL_DELAY_SECONDS, times=2),
            )

        reasoner = AsyncLLMReasoner(
            llm=llm,
            runner=tool_runner,
            interrupt_handler=self.interrupt_handler,
            state_manager=self.state_manager,
        )
        self.orchestrator = Orchestrator(
            reasoner=reasoner,
            state_manager=self.state_manager,
            interrupt_handler=self.interrupt_handler,
            output_queue=self.output_queue,
            trace=self.trace,
            turn_timeout_seconds=turn_timeout_seconds,
        )

        self.event_processor = EventProcessor(
            input_queue=self.input_queue,
            output_queue=self.output_queue,
            interrupt_handler=self.interrupt_handler,
            state_manager=self.state_manager,
            fast_path=self.fast_path,
            turn_handler=self.orchestrator,
            manifest_handler=self.orchestrator,
            trace=self.trace,
            clock=self.clock,
        )

        self._loop_task: "asyncio.Task | None" = None

    def start(self) -> None:
        """Start the event processor loop as a background task (non-blocking)."""
        if self._loop_task is not None and not self._loop_task.done():
            logger.warning("AgentRunner already running")
            return
        self._loop_task = asyncio.create_task(self.event_processor.run())
        logger.info("AgentRunner started (session=%s)", self.session_id)

    async def submit(self, event: Dict[str, Any]) -> None:
        """Feed an event into the engine."""
        await self.input_queue.put(event)

    async def stop(self) -> None:
        """Stop the event processor loop and cancel any pending tasks."""
        self.event_processor.stop()
        if self._loop_task is not None:
            self._loop_task.cancel()
            try:
                await self._loop_task
            except asyncio.CancelledError:
                pass
        await self.interrupt_handler.cancel_all_tasks()
        logger.info("AgentRunner stopped (session=%s)", self.session_id)


async def _drain_output(runner: AgentRunner, expected: int, timeout: float = 3.0) -> list[dict]:
    """Collect up to `expected` output payloads or stop at timeout."""
    received: list[dict] = []
    deadline = time.monotonic() + timeout
    while len(received) < expected and time.monotonic() < deadline:
        remaining = max(0.0, deadline - time.monotonic())
        try:
            payload = await asyncio.wait_for(runner.output_queue.get(), timeout=remaining)
        except asyncio.TimeoutError:
            break
        print(f"  [output {len(received) + 1}] {payload}")
        received.append(payload)
    return received


CURRENT_WEATHER_MANIFEST = [
    {
        "name": "current_weather",
        "description": "Get the current weather for a city.",
        "state_changing": False,
        "parameters": [{"name": "city", "type": "string", "required": True}],
    }
]


async def _demo(use_llm: bool = False) -> None:
    # NOTE: real startup should `await setup()` with the full warm-up
    # duration; skipped here so the demo runs instantly.
    logger.info("(demo) skipping full %.0fs warm-up", DEFAULT_WARMUP_SECONDS)

    # use_llm=False (default) exercises the exact same pipeline offline
    # via DeterministicDemoLLM. Run `python3 -m agent.main --llm` for the
    # real qwen2.5:14b path (requires `ollama serve` + `ollama pull qwen2.5:14b`).
    runner = AgentRunner(use_llm=use_llm, slow_demo_tools=not use_llm)
    runner.start()

    print("\n=== 1) Dynamic tool manifest: registering an unseen tool (current_weather) ===")
    await runner.submit({"type": "tool_manifest", "manifest": CURRENT_WEATHER_MANIFEST})
    await asyncio.sleep(0.05)

    print("\n=== 2) 'Book a flight to Mumbai' (turn starts reasoning + tool call) ===")
    await runner.submit(
        {"type": "text_chunk", "text": "Book a flight to Mumbai", "end_of_turn": True}
    )
    # Give the turn just enough time to start reasoning and issue the
    # book_flight(Mumbai) tool call (which then sleeps DEMO_TOOL_DELAY_SECONDS).
    await asyncio.sleep(0.06)

    print("\n=== 3) User interrupts before the Mumbai booking finishes ===")
    await runner.submit({"type": "interruption", "timestamp": runner.clock.now()})

    print("\n=== 4) 'Actually, Delhi' (corrected turn) ===")
    await runner.submit(
        {"type": "text_chunk", "text": "Actually, Delhi", "end_of_turn": True}
    )

    print("\n=== Output actions (fillers, tool_call for Mumbai, cancellations, then the Delhi turn) ===")
    # filler+tool_call for Mumbai (2), cancel x2 for the interruption (2),
    # filler+tool_call+final for the corrected Delhi turn (3) = 7 actions.
    received = await _drain_output(runner, expected=7, timeout=3.0)

    print("\n=== 5) 'What's the weather in Delhi?' (dynamic manifest tool, never hardcoded) ===")
    await runner.submit(
        {"type": "text_chunk", "text": "What's the weather in Delhi?", "end_of_turn": True}
    )
    # filler + tool_call(current_weather) + final = 3 actions.
    received += await _drain_output(runner, expected=3, timeout=3.0)

    print("\n=== 6) 'My washer shows error E4' (lookup_manual -> create_ticket, chained) ===")
    await runner.submit(
        {
            "type": "text_chunk",
            "text": "My washer shows error E4, can you look it up and open a ticket?",
            "end_of_turn": True,
        }
    )
    # filler + tool_call(lookup_manual) + tool_call(create_ticket) + final = 4 actions.
    received += await _drain_output(runner, expected=4, timeout=3.0)

    final_actions = [a for a in received if a.get("action") == "final"]
    print("\n=== Final state snapshot ===")
    print(f"  destination = {runner.state_manager.get_slot('destination')!r}")
    print(f"  ticket_id = {runner.state_manager.get_slot('ticket_id')!r}")
    if final_actions:
        print(f"  last final response = {final_actions[-1].get('response')!r}")
        print(f"  snapshot in that response = {final_actions[-1].get('state_snapshot')}")

    await runner.stop()

    print("\n=== Trace (for replay/debugging) ===")
    for line in runner.trace.to_jsonl().splitlines():
        print(f"  {line}")


if __name__ == "__main__":
    import sys

    asyncio.run(_demo(use_llm="--llm" in sys.argv))
