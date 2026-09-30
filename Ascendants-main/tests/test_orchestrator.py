import asyncio
import unittest

from agent.async_reasoner import AsyncReasonResult
from agent.interrupt_handler import InterruptHandler
from agent.orchestrator import Orchestrator
from agent.state_manager import StateManager
from agent.trace import TraceRecorder


class FakeAsyncReasoner:
    """Stands in for AsyncLLMReasoner so these tests never touch a real
    Ollama server or a real ToolRunner. `respond` is a native coroutine,
    exactly like the real one, and honors is_current the same way.
    """

    def __init__(self, result: AsyncReasonResult, block_event=None, runner=None):
        self._result = result
        self._block_event = block_event
        self.runner = runner  # only needed for register_manifest() tests
        self.calls = []

    async def respond(self, messages, is_current=None, on_tool_call=None):
        self.calls.append((messages, on_tool_call))
        if self._block_event is not None:
            await asyncio.get_event_loop().run_in_executor(None, self._block_event.wait, 5)
            if is_current is not None and not is_current():
                return AsyncReasonResult(cancelled=True)
        return self._result


def make_orchestrator(result: AsyncReasonResult, block_event=None, trace=None):
    state = StateManager()
    interrupts = InterruptHandler()
    output_q: "asyncio.Queue" = asyncio.Queue()
    reasoner = FakeAsyncReasoner(result, block_event)
    orch = Orchestrator(reasoner, state, interrupts, output_q, trace=trace)
    return orch, state, interrupts, output_q, reasoner


class TestOrchestratorHappyPath(unittest.IsolatedAsyncioTestCase):
    async def test_final_action_carries_state_snapshot(self):
        state = StateManager()
        state.update_slot("destination", "Goa")
        interrupts = InterruptHandler()
        output_q: "asyncio.Queue" = asyncio.Queue()
        reasoner = FakeAsyncReasoner(AsyncReasonResult(text="Found 3 flights to Goa."))
        orch = Orchestrator(reasoner, state, interrupts, output_q)

        call_id = await orch.start_turn("find flights to Goa")
        action = await asyncio.wait_for(output_q.get(), timeout=2)

        self.assertEqual(action["action"], "final")
        self.assertEqual(action["call_id"], call_id)
        self.assertEqual(action["response"], "Found 3 flights to Goa.")
        self.assertEqual(action["state_snapshot"]["slots"]["destination"], "Goa")

    async def test_clarification_result_emits_clarification_request_not_final(self):
        orch, _state, _interrupts, output_q, _r = make_orchestrator(
            AsyncReasonResult(clarification="Which destination did you mean?")
        )
        await orch.start_turn("book me a flight")
        action = await asyncio.wait_for(output_q.get(), timeout=2)
        self.assertEqual(action["action"], "clarification_request")
        self.assertEqual(action["reason"], "Which destination did you mean?")

    async def test_turn_is_registered_with_interrupt_handler(self):
        import threading

        block = threading.Event()
        orch, _state, interrupts, _out, _r = make_orchestrator(
            AsyncReasonResult(text="done"), block_event=block
        )
        call_id = await orch.start_turn("hello")
        await asyncio.sleep(0.02)
        self.assertIn(call_id, interrupts.active_tasks)
        block.set()


class TestOrchestratorCancellation(unittest.IsolatedAsyncioTestCase):
    async def test_interruption_prevents_stale_final_from_ever_being_emitted(self):
        """The core Interruption Recovery guarantee: once a turn is
        cancelled, its result must never reach the output queue, even if
        the underlying reasoning had already produced an answer.
        """
        import threading

        block = threading.Event()
        orch, _state, interrupts, output_q, _r = make_orchestrator(
            AsyncReasonResult(text="stale answer"), block_event=block
        )

        call_id = await orch.start_turn("book a flight to Delhi")
        await asyncio.sleep(0.02)
        self.assertIn(call_id, interrupts.active_tasks)

        await interrupts.cancel_all_tasks()
        block.set()

        await asyncio.sleep(0.05)
        self.assertTrue(output_q.empty(), "a superseded turn must not emit output")

    async def test_second_turn_after_interruption_still_completes_normally(self):
        state = StateManager()
        interrupts = InterruptHandler()
        output_q: "asyncio.Queue" = asyncio.Queue()
        reasoner = FakeAsyncReasoner(AsyncReasonResult(text="Delhi it is."))
        orch = Orchestrator(reasoner, state, interrupts, output_q)

        await orch.start_turn("actually, Delhi")
        action = await asyncio.wait_for(output_q.get(), timeout=2)
        self.assertEqual(action["response"], "Delhi it is.")


class TestOrchestratorTrace(unittest.IsolatedAsyncioTestCase):
    async def test_final_turn_is_recorded_in_trace(self):
        trace = TraceRecorder(session_id="s1")
        orch, _state, _interrupts, output_q, _r = make_orchestrator(
            AsyncReasonResult(text="hi"), trace=trace
        )
        await orch.start_turn("hello")
        await asyncio.wait_for(output_q.get(), timeout=2)
        kinds = [(e.kind, e.action) for e in trace.events()]
        self.assertIn(("input", None), kinds)
        self.assertIn(("action", "final"), kinds)

    async def test_cancelled_turn_is_recorded_as_a_decision(self):
        import threading

        trace = TraceRecorder(session_id="s1")
        block = threading.Event()
        orch, _state, interrupts, _out, _r = make_orchestrator(
            AsyncReasonResult(text="stale"), block_event=block, trace=trace
        )
        await orch.start_turn("hi")
        await asyncio.sleep(0.02)
        await interrupts.cancel_all_tasks()
        block.set()
        await asyncio.sleep(0.05)
        statuses = [e.status for e in trace.events() if e.kind == "decision"]
        self.assertIn("cancelled", statuses)


class TestOrchestratorManifest(unittest.IsolatedAsyncioTestCase):
    async def test_register_manifest_delegates_to_reasoners_runner(self):
        from agent.tools_builtin import ToolRunner

        runner = ToolRunner()
        state = StateManager()
        interrupts = InterruptHandler()
        output_q: "asyncio.Queue" = asyncio.Queue()
        reasoner = FakeAsyncReasoner(AsyncReasonResult(text="ok"), runner=runner)
        orch = Orchestrator(reasoner, state, interrupts, output_q)

        manifest = [
            {
                "name": "search_flights",
                "description": "search",
                "state_changing": False,
                "parameters": [{"name": "destination", "type": "string"}],
            }
        ]
        result = orch.register_manifest(manifest)
        self.assertEqual(result.errors, [])
        self.assertIn("search_flights", runner.tool_names())


class TestOrchestratorTimeout(unittest.IsolatedAsyncioTestCase):
    async def test_turn_exceeding_timeout_emits_a_final_with_timeout_error(self):
        """Spec §6: '120s wall-clock cap per scenario.' A hung reasoning
        call (bad Ollama connection, runaway tool loop) must not just
        block the turn forever — it must be cut off and reported.
        """

        class NeverFinishesReasoner:
            runner = None

            async def respond(self, messages, is_current=None, on_tool_call=None):
                await asyncio.sleep(3600)

        state = StateManager()
        state.update_slot("destination", "Goa")
        interrupts = InterruptHandler()
        output_q: "asyncio.Queue" = asyncio.Queue()
        trace = TraceRecorder(session_id="s1")
        orch = Orchestrator(
            NeverFinishesReasoner(),
            state,
            interrupts,
            output_q,
            trace=trace,
            turn_timeout_seconds=0.05,
        )

        call_id = await orch.start_turn("do something that hangs")
        action = await asyncio.wait_for(output_q.get(), timeout=2)

        self.assertEqual(action["action"], "final")
        self.assertEqual(action["call_id"], call_id)
        self.assertEqual(action["error"], "timeout")
        # A timeout must not lose track of session state either.
        self.assertEqual(action["state_snapshot"]["slots"]["destination"], "Goa")

    async def test_timed_out_turn_is_unregistered_not_leaked(self):
        class NeverFinishesReasoner:
            async def respond(self, messages, is_current=None, on_tool_call=None):
                await asyncio.sleep(3600)

        state = StateManager()
        interrupts = InterruptHandler()
        output_q: "asyncio.Queue" = asyncio.Queue()
        orch = Orchestrator(
            NeverFinishesReasoner(), state, interrupts, output_q, turn_timeout_seconds=0.05
        )

        call_id = await orch.start_turn("hang")
        await asyncio.wait_for(output_q.get(), timeout=2)

        self.assertNotIn(call_id, interrupts.active_tasks)

    async def test_timeout_is_recorded_in_trace(self):
        class NeverFinishesReasoner:
            async def respond(self, messages, is_current=None, on_tool_call=None):
                await asyncio.sleep(3600)

        trace = TraceRecorder(session_id="s1")
        state = StateManager()
        interrupts = InterruptHandler()
        output_q: "asyncio.Queue" = asyncio.Queue()
        orch = Orchestrator(
            NeverFinishesReasoner(),
            state,
            interrupts,
            output_q,
            trace=trace,
            turn_timeout_seconds=0.05,
        )

        await orch.start_turn("hang")
        await asyncio.wait_for(output_q.get(), timeout=2)

        statuses = [e.status for e in trace.events() if e.kind == "decision"]
        self.assertIn("timeout", statuses)


if __name__ == "__main__":
    unittest.main()
