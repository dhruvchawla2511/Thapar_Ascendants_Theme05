import asyncio
import unittest

from agent.interrupt_handler import InterruptHandler


async def _never_ending():
    try:
        await asyncio.sleep(3600)
    except asyncio.CancelledError:
        raise


async def _raises_after_cancel():
    try:
        await asyncio.sleep(3600)
    except asyncio.CancelledError:
        raise RuntimeError("cleanup failed")


class TestInterruptHandler(unittest.IsolatedAsyncioTestCase):
    async def test_register_and_unregister(self):
        h = InterruptHandler()
        task = asyncio.create_task(_never_ending())
        h.register_task("call_1", task)
        self.assertIn("call_1", h.active_tasks)
        h.unregister_task("call_1")
        self.assertNotIn("call_1", h.active_tasks)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task

    def test_register_rejects_empty_call_id(self):
        h = InterruptHandler()
        task = None
        with self.assertRaises(ValueError):
            h.register_task("", task)

    async def test_cancel_all_tasks_cancels_and_clears(self):
        h = InterruptHandler()
        t1 = asyncio.create_task(_never_ending())
        t2 = asyncio.create_task(_never_ending())
        h.register_task("call_1", t1)
        h.register_task("call_2", t2)

        cancelled_ids = await h.cancel_all_tasks()

        self.assertCountEqual(cancelled_ids, ["call_1", "call_2"])
        self.assertTrue(t1.cancelled())
        self.assertTrue(t2.cancelled())
        self.assertEqual(h.active_tasks, {})

    async def test_cancel_all_tasks_survives_a_task_that_misbehaves(self):
        """A task that swallows CancelledError and raises something else
        during cleanup must not stop the other tasks from being cancelled.
        """
        h = InterruptHandler()
        bad = asyncio.create_task(_raises_after_cancel())
        good = asyncio.create_task(_never_ending())
        h.register_task("bad", bad)
        h.register_task("good", good)

        cancelled_ids = await h.cancel_all_tasks()

        self.assertCountEqual(cancelled_ids, ["bad", "good"])
        self.assertTrue(good.cancelled())
        self.assertEqual(h.active_tasks, {})

    async def test_cancel_all_tasks_on_empty_registry(self):
        h = InterruptHandler()
        self.assertEqual(await h.cancel_all_tasks(), [])

    async def test_already_done_task_is_not_double_cancelled(self):
        h = InterruptHandler()

        async def finishes_immediately():
            return "done"

        t = asyncio.create_task(finishes_immediately())
        await asyncio.sleep(0)  # let it finish
        h.register_task("call_1", t)
        cancelled_ids = await h.cancel_all_tasks()
        self.assertEqual(cancelled_ids, ["call_1"])
        self.assertEqual(t.result(), "done")

    def test_format_cancellation_payload(self):
        h = InterruptHandler()
        payload = h.format_cancellation_payload("call_9")
        self.assertEqual(payload, {"action": "cancel", "call_id": "call_9"})

    async def test_survives_a_task_unregistering_itself_mid_cancellation(self):
        """Regression test: cancel_all_tasks() used to iterate a *live*
        dict view (`self.active_tasks.items()`), and a cancelled task is
        allowed to call unregister_task() on its OWN call_id while
        handling its own CancelledError (this is exactly what
        ToolRunner.run_cancellable does). Since `await task` below yields
        control, that self-unregister can happen mid-iteration and used to
        raise "dictionary changed size during iteration", aborting
        cancellation of every task queued after it. Must not raise, and
        every task must still actually get cancelled.
        """
        h = InterruptHandler()

        async def self_unregistering(call_id: str):
            try:
                await asyncio.sleep(3600)
            except asyncio.CancelledError:
                h.unregister_task(call_id)  # mutates active_tasks mid-cancel_all_tasks()
                raise

        t1 = asyncio.create_task(self_unregistering("call_1"))
        t2 = asyncio.create_task(self_unregistering("call_2"))
        t3 = asyncio.create_task(self_unregistering("call_3"))
        h.register_task("call_1", t1)
        h.register_task("call_2", t2)
        h.register_task("call_3", t3)

        cancelled_ids = await h.cancel_all_tasks()  # must not raise RuntimeError

        self.assertCountEqual(cancelled_ids, ["call_1", "call_2", "call_3"])
        self.assertTrue(t1.cancelled() and t2.cancelled() and t3.cancelled())
        self.assertEqual(h.active_tasks, {})


if __name__ == "__main__":
    unittest.main()
