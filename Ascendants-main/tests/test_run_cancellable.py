import asyncio
import unittest

from agent.interrupt_handler import InterruptHandler
from agent.tool_engine import CallStatus, ParamSpec, ToolSpec
from agent.tools_builtin import ToolExecutionError, ToolRunner


def make_runner() -> ToolRunner:
    runner = ToolRunner()
    runner.register(
        ToolSpec(
            name="search_flights",
            description="search",
            parameters=(ParamSpec("destination", str),),
            state_changing=False,
        ),
        lambda args: {"destination": args["destination"], "flights": ["A1"]},
    )
    runner.register(
        ToolSpec(
            name="book_flight",
            description="book",
            parameters=(ParamSpec("destination", str),),
            state_changing=True,
        ),
        lambda args: {"destination": args["destination"], "booking_id": "BK-1"},
    )
    return runner


class TestRunCancellableHappyPath(unittest.IsolatedAsyncioTestCase):
    async def test_successful_call_completes_and_returns_result(self):
        runner = make_runner()
        interrupts = InterruptHandler()
        outcome = await runner.run_cancellable("search_flights", {"destination": "Goa"}, interrupts)
        self.assertTrue(outcome.ok)
        self.assertIsNotNone(outcome.call_id)
        self.assertEqual(runner.engine.get_call(outcome.call_id).status, CallStatus.COMPLETED)

    async def test_on_call_created_fires_before_handler_completes(self):
        runner = make_runner()
        interrupts = InterruptHandler()
        seen = []

        async def on_created(call):
            seen.append(call.call_id)

        outcome = await runner.run_cancellable(
            "search_flights", {"destination": "Goa"}, interrupts, on_call_created=on_created
        )
        self.assertEqual(seen, [outcome.call_id])

    async def test_missing_required_argument_sets_needs_clarification(self):
        runner = make_runner()
        interrupts = InterruptHandler()
        outcome = await runner.run_cancellable("search_flights", {}, interrupts)
        self.assertFalse(outcome.ok)
        self.assertTrue(outcome.needs_clarification)
        self.assertIsNone(outcome.call_id)  # never even got a call_id — rejected pre-creation

    async def test_unknown_tool_is_not_a_clarification(self):
        runner = make_runner()
        interrupts = InterruptHandler()
        outcome = await runner.run_cancellable("teleport", {}, interrupts)
        self.assertFalse(outcome.ok)
        self.assertFalse(outcome.needs_clarification)

    async def test_duplicate_state_changing_call_is_rejected(self):
        runner = make_runner()
        interrupts = InterruptHandler()
        # First call still "pending" in a separate engine call, simulated by
        # directly creating one via engine (bypassing handler execution).
        runner.engine.create_call("book_flight", {"destination": "Pune"})
        outcome = await runner.run_cancellable("book_flight", {"destination": "Pune"}, interrupts)
        self.assertFalse(outcome.ok)
        self.assertFalse(outcome.needs_clarification)

    async def test_tool_execution_error_reported_not_raised(self):
        runner = ToolRunner()

        def boom(args):
            raise ToolExecutionError("bad input")

        runner.register(
            ToolSpec(name="boom", description="x", parameters=(), state_changing=False), boom
        )
        interrupts = InterruptHandler()
        outcome = await runner.run_cancellable("boom", {}, interrupts)
        self.assertFalse(outcome.ok)
        self.assertEqual(outcome.text, "bad input")


class TestRunCancellableCancellation(unittest.IsolatedAsyncioTestCase):
    async def test_cancelling_the_call_id_marks_it_cancelled_and_raises(self):
        import threading

        runner = ToolRunner()
        release = threading.Event()

        def slow_handler(args):
            release.wait(timeout=5)
            return "done"

        runner.register(
            ToolSpec(name="slow", description="x", parameters=(), state_changing=False),
            slow_handler,
        )
        interrupts = InterruptHandler()

        run_task = asyncio.create_task(runner.run_cancellable("slow", {}, interrupts))
        # Let run_cancellable create the call and register its task.
        for _ in range(50):
            if interrupts.active_tasks:
                break
            await asyncio.sleep(0.01)
        self.assertEqual(len(interrupts.active_tasks), 1)
        call_id = next(iter(interrupts.active_tasks))

        cancelled_ids = await interrupts.cancel_all_tasks()
        self.assertEqual(cancelled_ids, [call_id])
        release.set()

        with self.assertRaises(asyncio.CancelledError):
            await run_task

        self.assertEqual(runner.engine.get_call(call_id).status, CallStatus.CANCELLED)

    async def test_stale_result_cannot_resurrect_a_cancelled_call(self):
        """Even though complete_call() is technically reachable after
        cancellation (a background thread finishing late), ToolEngine must
        keep the call CANCELLED rather than letting the late result
        override that status — this is the structural stale-result
        protection the spec asks for.
        """
        runner = make_runner()
        interrupts = InterruptHandler()
        call = runner.engine.create_call("search_flights", {"destination": "Goa"})
        runner.engine.cancel_call(call.call_id)
        runner.engine.complete_call(call.call_id, {"late": "result"})
        self.assertEqual(runner.engine.get_call(call.call_id).status, CallStatus.CANCELLED)


if __name__ == "__main__":
    unittest.main()
