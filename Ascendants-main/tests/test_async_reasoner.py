import asyncio
import json
import unittest

from agent.async_reasoner import AsyncLLMReasoner
from agent.interrupt_handler import InterruptHandler
from agent.state_manager import StateManager
from agent.tool_engine import ParamSpec, ToolSpec
from agent.tools_builtin import ToolRunner, create_builtin_runner


def final(text):
    return json.dumps({"action": "final", "response": text})


def tool(name, **arguments):
    return json.dumps({"action": "tool", "tool": name, "arguments": arguments})


class FakeLLM:
    """Returns scripted replies in order and records every call. Sync,
    same as the real OllamaClient's interface — AsyncLLMReasoner runs it
    via asyncio.to_thread, exactly like production.
    """

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def chat(self, messages, *, json_mode=False):
        self.calls.append([dict(m) for m in messages])
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def make_flight_runner() -> ToolRunner:
    runner = ToolRunner()
    runner.register(
        ToolSpec(
            name="search_flights",
            description="search",
            parameters=(ParamSpec("destination", str),),
            state_changing=False,
            slot_updates=lambda args, result: {"destination": args["destination"]},
        ),
        lambda args: {"destination": args["destination"], "flights": ["A1"]},
    )
    return runner


USER = lambda t: [{"role": "user", "content": t}]  # noqa: E731


class TestAsyncReasonerHappyPath(unittest.IsolatedAsyncioTestCase):
    async def test_final_response_without_tools(self):
        llm = FakeLLM([final("Hello there.")])
        reasoner = AsyncLLMReasoner(llm, create_builtin_runner(), InterruptHandler(), StateManager())
        result = await reasoner.respond(USER("hi"))
        self.assertEqual(result.text, "Hello there.")
        self.assertFalse(result.cancelled)

    async def test_tool_call_then_final(self):
        llm = FakeLLM([tool("calculator", expression="2+2"), final("2+2 = 4.")])
        reasoner = AsyncLLMReasoner(llm, create_builtin_runner(), InterruptHandler(), StateManager())
        result = await reasoner.respond(USER("what is 2+2?"))
        self.assertEqual(result.text, "2+2 = 4.")
        self.assertEqual(len(result.tool_trace), 1)
        self.assertTrue(result.tool_trace[0]["ok"])
        self.assertIsNotNone(result.tool_trace[0]["call_id"])

    async def test_on_tool_call_hook_fires_with_call_id(self):
        llm = FakeLLM([tool("calculator", expression="1+1"), final("2.")])
        reasoner = AsyncLLMReasoner(llm, create_builtin_runner(), InterruptHandler(), StateManager())
        seen = []

        async def on_tool_call(call):
            seen.append(call.call_id)

        result = await reasoner.respond(USER("1+1?"), on_tool_call=on_tool_call)
        self.assertEqual(seen, [result.tool_trace[0]["call_id"]])

    async def test_successful_tool_applies_slot_updates_to_state(self):
        llm = FakeLLM([tool("search_flights", destination="Goa"), final("Found flights to Goa.")])
        state = StateManager()
        reasoner = AsyncLLMReasoner(llm, make_flight_runner(), InterruptHandler(), state)
        await reasoner.respond(USER("flights to Goa"))
        self.assertEqual(state.get_slot("destination"), "Goa")


class TestAsyncReasonerClarification(unittest.IsolatedAsyncioTestCase):
    async def test_missing_required_argument_returns_clarification_not_final(self):
        llm = FakeLLM([tool("search_flights", **{})])  # missing 'destination'
        state = StateManager()
        reasoner = AsyncLLMReasoner(llm, make_flight_runner(), InterruptHandler(), state)
        result = await reasoner.respond(USER("book me a flight"))
        self.assertIsNotNone(result.clarification)
        self.assertEqual(result.text, "")
        # No slot should have been set from a call that never happened.
        self.assertIsNone(state.get_slot("destination"))


class TestAsyncReasonerCancellation(unittest.IsolatedAsyncioTestCase):
    async def test_cancelled_before_first_llm_call_returns_immediately(self):
        llm = FakeLLM([final("should never be reached")])
        reasoner = AsyncLLMReasoner(llm, create_builtin_runner(), InterruptHandler(), StateManager())
        result = await reasoner.respond(USER("hi"), is_current=lambda: False)
        self.assertTrue(result.cancelled)
        self.assertEqual(llm.calls, [])

    async def test_tool_call_cancellation_marks_result_cancelled_not_completed(self):
        import threading

        release = threading.Event()

        def slow(args):
            release.wait(timeout=5)
            return {"destination": args["destination"]}

        runner = ToolRunner()
        runner.register(
            ToolSpec(
                name="search_flights",
                description="x",
                parameters=(ParamSpec("destination", str),),
                state_changing=False,
                slot_updates=lambda args, result: {"destination": args["destination"]},
            ),
            slow,
        )
        llm = FakeLLM([tool("search_flights", destination="Mumbai")])
        state = StateManager()
        interrupts = InterruptHandler()
        reasoner = AsyncLLMReasoner(llm, runner, interrupts, state)

        task = asyncio.create_task(reasoner.respond(USER("flights to Mumbai")))
        for _ in range(50):
            if interrupts.active_tasks:
                break
            await asyncio.sleep(0.01)

        await interrupts.cancel_all_tasks()
        release.set()

        with self.assertRaises(asyncio.CancelledError):
            await task

        # The critical guarantee: a cancelled tool call must never reach
        # the state-application step, regardless of what its handler
        # eventually returned.
        self.assertIsNone(state.get_slot("destination"))


if __name__ == "__main__":
    unittest.main()
