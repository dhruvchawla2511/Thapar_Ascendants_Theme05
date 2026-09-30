"""
test_scenarios.py — canonical, deterministic, end-to-end scenario tests
for the full pipeline (EventProcessor + Orchestrator + AsyncLLMReasoner +
ToolRunner/ToolEngine + StateManager + InterruptHandler), all driven
through a scripted fake LLM so nothing here needs Ollama or a network.

This is the integration layer above the unit tests in test_event_loop.py,
test_orchestrator.py, test_async_reasoner.py, test_run_cancellable.py,
and test_tool_manifest.py — those prove each piece in isolation; this file
proves they behave correctly wired together, the way agent/main.py wires
them.

See demo/ollama_smoke.py for a companion smoke test against a REAL
Ollama server (qwen2.5:14b) — that one is skipped automatically when
Ollama isn't reachable, and is never part of this suite's dependency.
"""

from __future__ import annotations

import asyncio
import json
import threading
import unittest
from typing import Callable, Optional

from agent.async_reasoner import AsyncLLMReasoner
from agent.event_loop import EventProcessor
from agent.interrupt_handler import InterruptHandler
from agent.orchestrator import Orchestrator
from agent.state_manager import StateManager
from agent.tool_engine import ParamSpec, ToolSpec
from agent.tools_builtin import ToolExecutionError, ToolRunner, create_builtin_runner
from agent.trace import TraceRecorder


class NoOpFastPath:
    async def handle_text_chunk(self, text, end_of_turn, **kwargs):
        pass


class ScriptedLLM:
    """A fake LLM whose replies are produced by a script function you
    control per-test: `script(messages) -> str (a JSON action, as text)`.
    Records every call it receives for assertions on prompt/tool-result
    content. Matches the real LLMClient sync interface exactly.
    """

    def __init__(self, script: Callable[[list[dict]], str]):
        self.script = script
        self.calls: list[list[dict]] = []

    def chat(self, messages, *, json_mode=False):
        self.calls.append([dict(m) for m in messages])
        return self.script(messages)


def final_json(text: str) -> str:
    return json.dumps({"action": "final", "response": text})


def tool_json(name: str, **arguments) -> str:
    return json.dumps({"action": "tool", "tool": name, "arguments": arguments})


def last_user_text(messages: list[dict]) -> str:
    for m in reversed(messages):
        if m["role"] == "user" and not m["content"].startswith(("TOOL_RESULT", "TOOL_ERROR")):
            return m["content"]
    return ""


def had_tool_result(messages: list[dict], tool: Optional[str] = None) -> bool:
    for m in messages:
        if m["role"] == "user" and m["content"].startswith(("TOOL_RESULT", "TOOL_ERROR")):
            if tool is None or f" {tool}:" in m["content"]:
                return True
    return False


def had_tool_message(messages: list[dict], tool: str, kind: str) -> bool:
    """kind = 'TOOL_RESULT' (success) or 'TOOL_ERROR' (failure) — unlike
    had_tool_result() above, distinguishes the two so a script can choose
    to retry on an error but finalize on a genuine success.
    """
    for m in messages:
        if m["role"] == "user" and m["content"].startswith(kind) and f" {tool}:" in m["content"]:
            return True
    return False


def make_pipeline(llm, tool_runner: Optional[ToolRunner] = None, session_id="s1", clock=None):
    """Builds one full, real (non-mocked, except for the LLM) pipeline:
    StateManager + InterruptHandler + ToolRunner/ToolEngine +
    AsyncLLMReasoner + Orchestrator + EventProcessor + TraceRecorder,
    exactly as agent.main.AgentRunner wires them.
    """
    from agent.clock import REAL_CLOCK

    clock = clock or REAL_CLOCK
    state = StateManager(session_id=session_id, clock=clock)
    interrupts = InterruptHandler()
    output_q: "asyncio.Queue" = asyncio.Queue()
    trace = TraceRecorder(session_id=session_id, clock=clock)
    runner = tool_runner or create_builtin_runner(include_flight_demo_tools=True)
    reasoner = AsyncLLMReasoner(llm, runner, interrupts, state)
    orch = Orchestrator(reasoner, state, interrupts, output_q, trace=trace)
    proc = EventProcessor(
        input_queue=asyncio.Queue(),
        output_queue=output_q,
        interrupt_handler=interrupts,
        state_manager=state,
        fast_path=NoOpFastPath(),
        turn_handler=orch,
        manifest_handler=orch,
    )
    return proc, state, interrupts, output_q, trace, runner


async def drain(output_q: "asyncio.Queue", n: int, timeout=2.0) -> list[dict]:
    out = []
    for _ in range(n):
        out.append(await asyncio.wait_for(output_q.get(), timeout=timeout))
    return out


async def turn(proc: EventProcessor, text: str) -> None:
    await proc._dispatch({"type": "text_chunk", "text": text, "end_of_turn": True})


async def interrupt(proc: EventProcessor) -> None:
    await proc._dispatch({"type": "interruption", "timestamp": 0.0})


# ---------------------------------------------------------------------------
# 1. Normal text request
# ---------------------------------------------------------------------------
class TestScenario1NormalRequest(unittest.IsolatedAsyncioTestCase):
    async def test_greeting_gets_a_direct_final_response(self):
        llm = ScriptedLLM(lambda msgs: final_json("Hi! How can I help?"))
        proc, state, interrupts, output_q, trace, runner = make_pipeline(llm)

        await turn(proc, "hello")
        [action] = await drain(output_q, 1)

        self.assertEqual(action["action"], "final")
        self.assertEqual(action["response"], "Hi! How can I help?")


# ---------------------------------------------------------------------------
# 2. User interruption during reasoning (before any tool call)
# ---------------------------------------------------------------------------
class TestScenario2InterruptionDuringReasoning(unittest.IsolatedAsyncioTestCase):
    async def test_interrupting_mid_llm_call_drops_the_result(self):
        release = threading.Event()

        def slow_script(messages):
            release.wait(timeout=5)
            return final_json("too late")

        llm = ScriptedLLM(slow_script)
        proc, state, interrupts, output_q, trace, runner = make_pipeline(llm)

        await turn(proc, "tell me something")
        for _ in range(50):
            if interrupts.active_tasks:
                break
            await asyncio.sleep(0.01)
        self.assertTrue(interrupts.active_tasks)

        await interrupt(proc)
        release.set()
        await asyncio.sleep(0.05)

        # Only the cancellation action(s) from the interruption itself —
        # never the "too late" final.
        while not output_q.empty():
            action = output_q.get_nowait()
            self.assertNotEqual(action.get("action"), "final")


# ---------------------------------------------------------------------------
# 3. Slot correction during an active task (the canonical Mumbai -> Delhi
#    scenario) + covers race conditions A, B, D from the task list.
# ---------------------------------------------------------------------------
class TestScenario3SlotCorrection(unittest.IsolatedAsyncioTestCase):
    async def test_correcting_destination_mid_booking(self):
        release = threading.Event()

        def slow_book_handler(args):
            release.wait(timeout=5)
            return {"destination": args["destination"], "booking_id": f"BK-{args['destination']}"}

        runner = create_builtin_runner(include_flight_demo_tools=True)
        runner.register(runner.get_spec("book_flight"), slow_book_handler)

        def script(messages):
            text = last_user_text(messages).lower()
            destination = "Delhi" if "delhi" in text else ("Mumbai" if "mumbai" in text else None)
            if had_tool_result(messages, "book_flight"):
                return final_json(f"Booked to {destination or 'your destination'}.")
            if destination:
                return tool_json("book_flight", destination=destination)
            return final_json("ok")

        llm = ScriptedLLM(script)
        proc, state, interrupts, output_q, trace, _r = make_pipeline(llm, tool_runner=runner)

        await turn(proc, "book a flight to Mumbai")
        # Let the Mumbai turn register its tool call (which is now blocked
        # on `release`).
        for _ in range(50):
            if len(interrupts.active_tasks) >= 2:  # turn + tool call
                break
            await asyncio.sleep(0.01)
        self.assertGreaterEqual(len(interrupts.active_tasks), 2)

        await interrupt(proc)  # race B/D: cancel the in-flight Mumbai call
        release.set()  # allow the (now cancelled) Mumbai handler thread to unblock

        await turn(proc, "actually, Delhi")

        # Drain everything the pipeline produces for both turns.
        actions = []
        deadline = asyncio.get_event_loop().time() + 2.0
        while asyncio.get_event_loop().time() < deadline:
            try:
                actions.append(await asyncio.wait_for(output_q.get(), timeout=0.2))
            except asyncio.TimeoutError:
                break

        finals = [a for a in actions if a["action"] == "final"]
        self.assertEqual(len(finals), 1, f"expected exactly one final, got {actions}")
        self.assertIn("Delhi", finals[0]["response"])
        self.assertNotIn("Mumbai", finals[0]["response"])

        # Race B: the stale Mumbai result must never have touched state.
        self.assertEqual(state.get_slot("destination"), "Delhi")
        self.assertEqual(finals[0]["state_snapshot"]["slots"]["destination"], "Delhi")


# ---------------------------------------------------------------------------
# 4. State-changing tool duplicate prevention (race condition C)
# ---------------------------------------------------------------------------
class TestScenario4DuplicatePrevention(unittest.IsolatedAsyncioTestCase):
    async def test_two_concurrent_identical_book_flight_calls_second_rejected(self):
        release = threading.Event()
        call_count = {"n": 0}

        def slow_handler(args):
            call_count["n"] += 1
            release.wait(timeout=5)
            return {"destination": args["destination"]}

        runner = ToolRunner()
        runner.register(
            ToolSpec(
                name="book_flight",
                description="book",
                parameters=(ParamSpec("destination", str),),
                state_changing=True,
            ),
            slow_handler,
        )
        interrupts = InterruptHandler()

        first = asyncio.create_task(
            runner.run_cancellable("book_flight", {"destination": "Pune"}, interrupts)
        )
        await asyncio.sleep(0.02)  # let the first call register (still pending)

        second_outcome = await runner.run_cancellable(
            "book_flight", {"destination": "Pune"}, interrupts
        )
        self.assertFalse(second_outcome.ok)
        self.assertIn("already pending", second_outcome.text.lower())

        release.set()
        first_outcome = await first
        self.assertTrue(first_outcome.ok)
        # The handler itself must only have run once — the duplicate was
        # rejected before ever reaching it.
        self.assertEqual(call_count["n"], 1)


# ---------------------------------------------------------------------------
# 5. Stale tool result after cancellation cannot modify state (race B again,
#    isolated at the reasoner level for a tighter assertion window)
# ---------------------------------------------------------------------------
class TestScenario5StaleResultAfterCancellation(unittest.IsolatedAsyncioTestCase):
    async def test_late_arriving_result_never_updates_state(self):
        release = threading.Event()

        def slow_handler(args):
            release.wait(timeout=5)
            return {"destination": args["destination"]}

        runner = create_builtin_runner(include_flight_demo_tools=True)
        runner.register(runner.get_spec("book_flight"), slow_handler)

        llm = ScriptedLLM(lambda msgs: tool_json("book_flight", destination="Mumbai"))
        proc, state, interrupts, output_q, trace, _r = make_pipeline(llm, tool_runner=runner)

        await turn(proc, "book Mumbai")
        for _ in range(50):
            if interrupts.active_tasks:
                break
            await asyncio.sleep(0.01)

        await interrupt(proc)
        release.set()
        await asyncio.sleep(0.05)

        self.assertIsNone(state.get_slot("destination"))


# ---------------------------------------------------------------------------
# 6. Tool failure / retry
# ---------------------------------------------------------------------------
class TestScenario6ToolFailureRetry(unittest.IsolatedAsyncioTestCase):
    async def test_llm_retries_after_a_tool_error_and_succeeds(self):
        attempts = {"n": 0}

        def flaky_handler(args):
            attempts["n"] += 1
            if attempts["n"] == 1:
                raise ToolExecutionError("temporary booking system error")
            return {"destination": args["destination"], "booking_id": "BK-OK"}

        runner = create_builtin_runner(include_flight_demo_tools=True)
        runner.register(runner.get_spec("book_flight"), flaky_handler)

        def script(messages):
            if had_tool_message(messages, "book_flight", "TOOL_RESULT"):
                return final_json("Booked successfully.")
            return tool_json("book_flight", destination="Goa")

        llm = ScriptedLLM(script)
        proc, state, interrupts, output_q, trace, _r = make_pipeline(llm, tool_runner=runner)

        await turn(proc, "book a flight to Goa")
        actions = await drain(output_q, 3)
        action = next(a for a in actions if a["action"] == "final")

        self.assertEqual(action["action"], "final")
        self.assertEqual(action["response"], "Booked successfully.")
        self.assertEqual(attempts["n"], 2)  # failed once, retried, succeeded


# ---------------------------------------------------------------------------
# 7. Missing argument -> clarification
# ---------------------------------------------------------------------------
class TestScenario7MissingArgumentClarification(unittest.IsolatedAsyncioTestCase):
    async def test_missing_destination_asks_for_clarification_not_final(self):
        llm = ScriptedLLM(lambda msgs: tool_json("book_flight"))  # no destination
        proc, state, interrupts, output_q, trace, _r = make_pipeline(llm)

        await turn(proc, "book me a flight")
        [action] = await drain(output_q, 1)

        self.assertEqual(action["action"], "clarification_request")
        self.assertIn("destination", action["reason"])
        self.assertIsNone(state.get_slot("destination"))


# ---------------------------------------------------------------------------
# 8. Dynamic unseen tool from manifest
# ---------------------------------------------------------------------------
class TestScenario8DynamicUnseenTool(unittest.IsolatedAsyncioTestCase):
    async def test_manifest_tool_never_hardcoded_is_callable_end_to_end(self):
        def script(messages):
            if had_tool_result(messages, "current_weather"):
                return final_json("It's sunny in Delhi.")
            return tool_json("current_weather", city="Delhi")

        llm = ScriptedLLM(script)
        proc, state, interrupts, output_q, trace, runner = make_pipeline(llm)

        self.assertNotIn("current_weather", runner.tool_names())
        await proc._dispatch(
            {
                "type": "tool_manifest",
                "manifest": [
                    {
                        "name": "current_weather",
                        "description": "weather",
                        "state_changing": False,
                        "parameters": [{"name": "city", "type": "string"}],
                    }
                ],
            }
        )
        self.assertIn("current_weather", runner.tool_names())

        await turn(proc, "what's the weather in Delhi?")
        tool_call_action, final_action = await drain(output_q, 2)

        self.assertEqual(tool_call_action["tool_name"], "current_weather")
        self.assertIsNotNone(tool_call_action["call_id"])
        self.assertEqual(final_action["action"], "final")
        self.assertIn("sunny", final_action["response"])


# ---------------------------------------------------------------------------
# 9 & 10. Audio / PNG input normalization + grounding
# ---------------------------------------------------------------------------
class TestScenario9And10MultimodalGrounding(unittest.IsolatedAsyncioTestCase):
    async def test_well_formed_audio_reaches_the_reasoner_as_grounded_text(self):
        import struct

        llm = ScriptedLLM(lambda msgs: final_json("Heard you."))
        proc, state, interrupts, output_q, trace, _r = make_pipeline(llm)

        sample_rate = 16000
        data = b"\x00" * (sample_rate * 2)  # ~1s of 16-bit mono
        wav = (
            b"RIFF"
            + struct.pack("<I", 36 + len(data))
            + b"WAVE"
            + b"fmt "
            + struct.pack("<IHHIIHH", 16, 1, 1, sample_rate, sample_rate * 2, 2, 16)
            + b"data"
            + struct.pack("<I", len(data))
            + data
        )

        await proc._dispatch(
            {
                "type": "raw_input",
                "payload": {"modality": "audio_wav", "data": wav, "sample_rate": sample_rate},
            }
        )
        [action] = await drain(output_q, 1)
        self.assertEqual(action["action"], "final")
        # Prove the model actually SAW a grounded description of the audio.
        seen_text = last_user_text(llm.calls[0])
        self.assertIn("audio clip", seen_text)

    async def test_well_formed_png_reaches_the_reasoner_as_grounded_text(self):
        llm = ScriptedLLM(lambda msgs: final_json("Saw the image."))
        proc, state, interrupts, output_q, trace, _r = make_pipeline(llm)

        png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 40
        await proc._dispatch(
            {
                "type": "raw_input",
                "payload": {"modality": "image_png", "data": png, "width": 800, "height": 600},
            }
        )
        [action] = await drain(output_q, 1)
        self.assertEqual(action["action"], "final")
        seen_text = last_user_text(llm.calls[0])
        self.assertIn("800x600", seen_text)

    async def test_insufficient_audio_metadata_asks_for_clarification(self):
        llm = ScriptedLLM(lambda msgs: final_json("should not be called"))
        proc, state, interrupts, output_q, trace, _r = make_pipeline(llm)

        await proc._dispatch(
            {
                "type": "raw_input",
                # No sample_rate metadata -> grounding reports insufficient.
                "payload": {"modality": "audio_wav", "data": b"RIFF" + b"\x00" * 8 + b"WAVE" + b"\x00" * 40},
            }
        )
        [action] = await drain(output_q, 1)
        self.assertEqual(action["action"], "clarification_request")
        self.assertEqual(llm.calls, [])  # the model was never even invoked


# ---------------------------------------------------------------------------
# 11. Chained tool calls
# ---------------------------------------------------------------------------
class TestScenario11ChainedToolCalls(unittest.IsolatedAsyncioTestCase):
    async def test_search_then_book_then_final(self):
        def script(messages):
            if had_tool_result(messages, "book_flight"):
                return final_json("All set — booked to Goa.")
            if had_tool_result(messages, "search_flights"):
                return tool_json("book_flight", destination="Goa")
            return tool_json("search_flights", destination="Goa")

        llm = ScriptedLLM(script)
        proc, state, interrupts, output_q, trace, runner = make_pipeline(llm)

        await turn(proc, "find and book a flight to Goa")
        # 2 tool_call actions + 1 final
        actions = await drain(output_q, 3)

        tool_calls = [a for a in actions if "tool_name" in a]
        finals = [a for a in actions if a.get("action") == "final"]
        self.assertEqual([a["tool_name"] for a in tool_calls], ["search_flights", "book_flight"])
        self.assertEqual(len(finals), 1)
        self.assertEqual(state.get_slot("destination"), "Goa")


# ---------------------------------------------------------------------------
# 12. Final response contains correct state snapshot (intent/slots/timestamp)
# ---------------------------------------------------------------------------
class TestScenario12StateSnapshotShape(unittest.IsolatedAsyncioTestCase):
    async def test_snapshot_has_intent_slots_and_timestamp(self):
        def script(messages):
            if had_tool_result(messages, "search_flights"):
                return final_json("Here are some flights.")
            return tool_json("search_flights", destination="Pune")

        llm = ScriptedLLM(script)
        proc, state, interrupts, output_q, trace, _r = make_pipeline(llm)
        state.set_intent("booking")

        await turn(proc, "flights to Pune")
        actions = await drain(output_q, 2)
        final_action = next(a for a in actions if a["action"] == "final")
        snap = final_action["state_snapshot"]

        self.assertEqual(snap["intent"], "booking")
        self.assertEqual(snap["slots"]["destination"], "Pune")
        self.assertIn("last_updated", snap)


# ---------------------------------------------------------------------------
# Race conditions E and F (A, B, C, D are covered above)
# ---------------------------------------------------------------------------
class TestRaceConditionE_SessionIsolation(unittest.IsolatedAsyncioTestCase):
    async def test_two_independent_sessions_never_share_state(self):
        llm_a = ScriptedLLM(lambda msgs: tool_json("search_flights", destination="Mumbai"))
        llm_b = ScriptedLLM(lambda msgs: tool_json("search_flights", destination="Delhi"))
        proc_a, state_a, *_a = make_pipeline(llm_a, session_id="session-a")
        proc_b, state_b, *_b = make_pipeline(llm_b, session_id="session-b")

        await turn(proc_a, "flights to Mumbai")
        await turn(proc_b, "flights to Delhi")
        await asyncio.sleep(0.05)

        self.assertEqual(state_a.get_slot("destination"), "Mumbai")
        self.assertEqual(state_b.get_slot("destination"), "Delhi")
        self.assertNotEqual(state_a.session_id, state_b.session_id)


class TestRaceConditionF_InterruptionConcurrentWithToolResult(unittest.IsolatedAsyncioTestCase):
    async def test_interruption_racing_a_completing_tool_call(self):
        """Line up cancellation and handler completion as closely as
        possible: the handler is released at (almost) the same moment the
        interruption is dispatched. Either ordering must leave state
        unmodified — there is no safe interleaving where a superseded
        result gets applied.
        """
        release = threading.Event()

        def instant_after_release(args):
            release.wait(timeout=5)
            return {"destination": args["destination"]}

        runner = create_builtin_runner(include_flight_demo_tools=True)
        runner.register(runner.get_spec("book_flight"), instant_after_release)

        llm = ScriptedLLM(lambda msgs: tool_json("book_flight", destination="Mumbai"))
        proc, state, interrupts, output_q, trace, _r = make_pipeline(llm, tool_runner=runner)

        await turn(proc, "book Mumbai")
        for _ in range(50):
            if interrupts.active_tasks:
                break
            await asyncio.sleep(0.01)

        # Fire the release and the interruption essentially back-to-back.
        release.set()
        await interrupt(proc)
        await asyncio.sleep(0.05)

        self.assertIsNone(state.get_slot("destination"))


# ---------------------------------------------------------------------------
# Protocol Compliance: every action emitted across a full, realistic run
# must be schema-valid (spec §3.2.6 / Safety & Protocol, 10% of scoring)
# ---------------------------------------------------------------------------
class TestProtocolCompliance(unittest.IsolatedAsyncioTestCase):
    async def test_every_action_across_a_full_scenario_passes_schema_validation(self):
        from agent.action_schema import validate_action

        def script(messages):
            text = last_user_text(messages).lower()
            if had_tool_message(messages, "book_flight", "TOOL_RESULT"):
                return final_json("Booked!")
            if had_tool_result(messages, "search_flights"):
                return tool_json("book_flight", destination="Goa")
            if "search" in text or "flight" in text:
                return tool_json("search_flights", destination="Goa")
            return final_json("hi")

        llm = ScriptedLLM(script)
        proc, state, interrupts, output_q, trace, runner = make_pipeline(llm)

        await proc._dispatch(
            {
                "type": "tool_manifest",
                "manifest": [
                    {
                        "name": "current_weather",
                        "description": "weather",
                        "state_changing": False,
                        "parameters": [{"name": "city", "type": "string"}],
                    }
                ],
            }
        )
        await turn(proc, "search for a flight and book it")

        actions = []
        deadline = asyncio.get_event_loop().time() + 2.0
        while asyncio.get_event_loop().time() < deadline:
            try:
                actions.append(await asyncio.wait_for(output_q.get(), timeout=0.2))
            except asyncio.TimeoutError:
                break

        self.assertGreater(len(actions), 0)
        for action in actions:
            validate_action(action)  # raises on any violation


# ---------------------------------------------------------------------------
# Virtual Clock: a full run driven entirely by a deterministic clock
# (spec §4: "Virtual Clock Streaming Harness: Deterministic event replay")
# ---------------------------------------------------------------------------
class TestVirtualClockDrivenRun(unittest.IsolatedAsyncioTestCase):
    async def test_full_scenario_produces_deterministic_trace_timestamps(self):
        from agent.clock import VirtualClock

        clock = VirtualClock(start=1_000_000.0)

        def script(messages):
            if had_tool_result(messages, "search_flights"):
                return final_json("Found some flights.")
            return tool_json("search_flights", destination="Goa")

        llm = ScriptedLLM(script)
        proc, state, interrupts, output_q, trace, runner = make_pipeline(llm, clock=clock)

        clock.tick(0.1)
        await turn(proc, "find flights to Goa")
        await drain(output_q, 2)  # tool_call + final

        events = trace.events()
        self.assertGreater(len(events), 0)
        # Every timestamp came from the shared VirtualClock, never real
        # wall-clock time — so this run's exact timestamps are 100%
        # reproducible on a re-run with the same clock starting point.
        for e in events:
            self.assertGreaterEqual(e.timestamp, 1_000_000.1)
            self.assertLess(e.timestamp, 1_000_001.0)  # nothing advanced the clock further
        # And the final state snapshot's own timestamp agrees with the
        # clock too (StateManager shares the same clock as TraceRecorder).
        snapshot = state.get_snapshot()
        self.assertEqual(snapshot.last_updated, clock.now())

    async def test_rerunning_with_a_fresh_identically_seeded_clock_reproduces_timestamps(self):
        """The actual point of a virtual clock: replay determinism. Two
        separate runs, each with its own VirtualClock seeded identically
        and ticked identically, must produce byte-identical timestamps.
        """
        from agent.clock import VirtualClock

        def script(messages):
            if had_tool_result(messages, "search_flights"):
                return final_json("done")
            return tool_json("search_flights", destination="Goa")

        async def run_once():
            clock = VirtualClock(start=42.0)
            llm = ScriptedLLM(script)
            proc, state, interrupts, output_q, trace, runner = make_pipeline(
                llm, clock=clock, session_id="replay"
            )
            clock.tick(1.0)
            await turn(proc, "find flights to Goa")
            await drain(output_q, 2)
            return [e.timestamp for e in trace.events()]

        timestamps_a = await run_once()
        timestamps_b = await run_once()
        self.assertEqual(timestamps_a, timestamps_b)
        self.assertGreater(len(timestamps_a), 0)


if __name__ == "__main__":
    unittest.main()
