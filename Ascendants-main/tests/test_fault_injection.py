import unittest

from agent.fault_injection import FaultPlan, FaultStep, apply_fault_plan, wrap_with_faults
from agent.tools_builtin import ToolExecutionError, ToolRunner, register_flight_demo_tools


class FakeSleep:
    """Records requested durations instead of actually blocking — makes
    latency-injection tests instant and fully deterministic.
    """

    def __init__(self):
        self.calls: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)


class TestFaultStep(unittest.TestCase):
    def test_rejects_negative_latency(self):
        with self.assertRaises(ValueError):
            FaultStep(latency_seconds=-1.0)

    def test_defaults_are_a_no_op(self):
        step = FaultStep()
        self.assertEqual(step.latency_seconds, 0.0)
        self.assertIsNone(step.exception)


class TestFaultPlan(unittest.TestCase):
    def test_step_for_call_is_one_indexed(self):
        plan = FaultPlan(steps=[FaultStep(latency_seconds=1.0), FaultStep(latency_seconds=2.0)])
        self.assertEqual(plan.step_for_call(1).latency_seconds, 1.0)
        self.assertEqual(plan.step_for_call(2).latency_seconds, 2.0)

    def test_calls_past_the_plan_length_get_a_no_op_step(self):
        plan = FaultPlan(steps=[FaultStep(latency_seconds=1.0)])
        self.assertEqual(plan.step_for_call(2), FaultStep())
        self.assertEqual(plan.step_for_call(100), FaultStep())

    def test_rejects_call_number_below_one(self):
        plan = FaultPlan()
        with self.assertRaises(ValueError):
            plan.step_for_call(0)

    def test_fail_then_succeed_constructor(self):
        exc = ToolExecutionError("boom")
        plan = FaultPlan.fail_then_succeed(exc, times=2)
        self.assertIs(plan.step_for_call(1).exception, exc)
        self.assertIs(plan.step_for_call(2).exception, exc)
        self.assertIsNone(plan.step_for_call(3).exception)

    def test_latency_then_normal_constructor(self):
        plan = FaultPlan.latency_then_normal(0.5, times=1)
        self.assertEqual(plan.step_for_call(1).latency_seconds, 0.5)
        self.assertEqual(plan.step_for_call(2).latency_seconds, 0.0)


class TestWrapWithFaults(unittest.TestCase):
    def test_no_op_plan_behaves_identically_to_unwrapped_handler(self):
        handler = lambda args: args["x"] * 2
        wrapped = wrap_with_faults(handler, FaultPlan())
        self.assertEqual(wrapped({"x": 21}), 42)

    def test_injects_latency_via_the_injected_sleep_fn_deterministically(self):
        sleep = FakeSleep()
        handler = lambda args: "ok"
        plan = FaultPlan.latency_then_normal(0.3, times=2)
        wrapped = wrap_with_faults(handler, plan, sleep_fn=sleep)

        wrapped({})
        wrapped({})
        wrapped({})  # past the plan — no more injected latency

        self.assertEqual(sleep.calls, [0.3, 0.3])

    def test_raises_the_configured_exception_instead_of_calling_the_real_handler(self):
        calls = []
        handler = lambda args: calls.append(args) or "real result"
        plan = FaultPlan.fail_then_succeed(ToolExecutionError("simulated outage"), times=1)
        wrapped = wrap_with_faults(handler, plan, sleep_fn=FakeSleep())

        with self.assertRaises(ToolExecutionError):
            wrapped({"x": 1})
        self.assertEqual(calls, [])  # the real handler must never have run

        result = wrapped({"x": 1})
        self.assertEqual(result, "real result")
        self.assertEqual(calls, [{"x": 1}])

    def test_latency_and_exception_can_be_combined_on_one_step(self):
        sleep = FakeSleep()
        plan = FaultPlan(steps=[FaultStep(latency_seconds=1.0, exception=ToolExecutionError("x"))])
        wrapped = wrap_with_faults(lambda args: "unreached", plan, sleep_fn=sleep)
        with self.assertRaises(ToolExecutionError):
            wrapped({})
        self.assertEqual(sleep.calls, [1.0])  # latency happens BEFORE the raise

    def test_two_wraps_of_the_same_handler_have_independent_call_counters(self):
        handler = lambda args: "ok"
        plan = FaultPlan.fail_then_succeed(ToolExecutionError("x"), times=1)
        wrapped_a = wrap_with_faults(handler, plan, sleep_fn=FakeSleep())
        wrapped_b = wrap_with_faults(handler, plan, sleep_fn=FakeSleep())

        with self.assertRaises(ToolExecutionError):
            wrapped_a({})
        # wrapped_b's OWN first call still fails too — it doesn't inherit
        # wrapped_a's call count.
        with self.assertRaises(ToolExecutionError):
            wrapped_b({})


class TestApplyFaultPlan(unittest.TestCase):
    def make_runner(self) -> ToolRunner:
        runner = ToolRunner()
        register_flight_demo_tools(runner)
        return runner

    def test_wraps_the_currently_registered_handler_in_place(self):
        runner = self.make_runner()
        plan = FaultPlan.fail_then_succeed(ToolExecutionError("system down"), times=1)
        apply_fault_plan(runner, "book_flight", plan, sleep_fn=FakeSleep())

        first = runner.run("book_flight", {"destination": "Goa"})
        self.assertFalse(first.ok)
        self.assertEqual(first.text, "system down")

        second = runner.run("book_flight", {"destination": "Goa"})
        self.assertTrue(second.ok)

    def test_schema_and_state_changing_flag_are_preserved(self):
        runner = self.make_runner()
        apply_fault_plan(runner, "book_flight", FaultPlan(), sleep_fn=FakeSleep())
        spec = runner.get_spec("book_flight")
        self.assertTrue(spec.state_changing)
        self.assertEqual([p.name for p in spec.parameters], ["destination", "flight_number"])

    def test_slot_updates_still_apply_after_wrapping(self):
        runner = self.make_runner()
        apply_fault_plan(runner, "book_flight", FaultPlan(), sleep_fn=FakeSleep())
        outcome = runner.run("book_flight", {"destination": "Goa"})
        self.assertTrue(outcome.ok)
        call = runner.engine.get_call(outcome.call_id)
        spec = runner.get_spec("book_flight")
        slots = spec.slot_updates(call.args, call.result)
        self.assertEqual(slots["destination"], "Goa")

    def test_unknown_tool_name_raises_key_error(self):
        runner = self.make_runner()
        with self.assertRaises(KeyError):
            apply_fault_plan(runner, "does_not_exist", FaultPlan())

    def test_duplicate_protection_still_applies_after_wrapping(self):
        """Wrapping with a fault plan must not bypass the engine's own
        fingerprint duplicate-protection for state-changing tools — the
        wrapper only affects the HANDLER, not the schema/create_call path.
        """
        from agent.tool_engine import DuplicateStateChangingCallError

        runner = self.make_runner()
        apply_fault_plan(
            runner, "book_flight", FaultPlan.latency_then_normal(0.01, times=5), sleep_fn=FakeSleep()
        )
        runner.engine.create_call("book_flight", {"destination": "Pune"})
        with self.assertRaises(DuplicateStateChangingCallError):
            runner.engine.create_call("book_flight", {"destination": "Pune"})


if __name__ == "__main__":
    unittest.main()
