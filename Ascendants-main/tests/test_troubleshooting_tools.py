import unittest

from agent.tool_engine import CallStatus, DuplicateStateChangingCallError
from agent.tools_builtin import ToolRunner, create_builtin_runner, register_troubleshooting_demo_tools


def make_runner() -> ToolRunner:
    runner = ToolRunner()
    register_troubleshooting_demo_tools(runner)
    return runner


class TestCreateTicketTool(unittest.TestCase):
    def test_registered_as_state_changing(self):
        runner = make_runner()
        spec = runner.get_spec("create_ticket")
        self.assertIsNotNone(spec)
        self.assertTrue(spec.state_changing)

    def test_creates_a_ticket_with_deterministic_sequential_ids(self):
        runner = make_runner()
        first = runner.run("create_ticket", {"subject": "washer broken"})
        second = runner.run("create_ticket", {"subject": "router down"})
        self.assertTrue(first.ok and second.ok)
        self.assertIn("TCK-00001", first.text)
        self.assertIn("TCK-00002", second.text)

    def test_two_separate_runners_each_start_ticket_numbering_at_one(self):
        """Regression guard: ticket numbering must be session-scoped (per
        ToolRunner), never a shared module-level global — otherwise two
        sessions running in the same process would get non-reproducible,
        order-dependent ticket IDs.
        """
        runner_a = make_runner()
        runner_b = make_runner()
        runner_a.run("create_ticket", {"subject": "x"})
        runner_a.run("create_ticket", {"subject": "y"})
        first_in_b = runner_b.run("create_ticket", {"subject": "z"})
        self.assertIn("TCK-00001", first_in_b.text)

    def test_priority_defaults_to_normal(self):
        runner = make_runner()
        outcome = runner.run("create_ticket", {"subject": "x"})
        self.assertIn("'priority': 'normal'", outcome.text)

    def test_priority_can_be_overridden(self):
        runner = make_runner()
        outcome = runner.run("create_ticket", {"subject": "x", "priority": "high"})
        self.assertIn("'priority': 'high'", outcome.text)

    def test_missing_subject_needs_clarification(self):
        runner = make_runner()
        outcome = runner.run("create_ticket", {})
        self.assertFalse(outcome.ok)
        self.assertTrue(outcome.needs_clarification)

    def test_duplicate_pending_ticket_call_is_rejected(self):
        runner = make_runner()
        runner.engine.create_call("create_ticket", {"subject": "washer broken"})
        with self.assertRaises(DuplicateStateChangingCallError):
            runner.engine.create_call("create_ticket", {"subject": "washer broken"})

    def test_slot_updates_capture_subject_and_ticket_id(self):
        runner = make_runner()
        call = runner.engine.create_call("create_ticket", {"subject": "washer broken"})
        result = {"subject": "washer broken", "ticket_id": "TCK-00001", "status": "open"}
        runner.engine.complete_call(call.call_id, result)
        completed = runner.engine.get_call(call.call_id)
        spec = runner.get_spec("create_ticket")
        slots = spec.slot_updates(completed.args, completed.result)
        self.assertEqual(slots, {"ticket_subject": "washer broken", "ticket_id": "TCK-00001"})


class TestLookupManualTool(unittest.TestCase):
    def test_registered_as_read_only(self):
        runner = make_runner()
        spec = runner.get_spec("lookup_manual")
        self.assertFalse(spec.state_changing)

    def test_known_device_and_query_returns_a_specific_snippet(self):
        runner = make_runner()
        outcome = runner.run("lookup_manual", {"device": "washer", "query": "E4"})
        self.assertTrue(outcome.ok)
        self.assertIn("drainage", outcome.text)

    def test_unknown_device_returns_a_helpful_fallback_not_a_crash(self):
        runner = make_runner()
        outcome = runner.run("lookup_manual", {"device": "toaster", "query": "smoke"})
        self.assertTrue(outcome.ok)
        self.assertIn("No manual section found", outcome.text)

    def test_query_is_optional(self):
        runner = make_runner()
        outcome = runner.run("lookup_manual", {"device": "washer"})
        self.assertTrue(outcome.ok)

    def test_accepts_a_frame_description_and_passes_it_through(self):
        """This is the actual 'frame-grounded' part: the LLM can copy the
        exact string grounding.py produces for a well-formed image event
        (e.g. '[image frame: 640x480px]') straight into this argument.
        """
        runner = make_runner()
        outcome = runner.run(
            "lookup_manual",
            {
                "device": "router",
                "query": "blinking_red",
                "frame_description": "[image frame: 640x480px]",
            },
        )
        self.assertTrue(outcome.ok)
        self.assertIn("[image frame: 640x480px]", outcome.text)

    def test_read_only_calls_are_never_deduplicated(self):
        runner = make_runner()
        runner.engine.create_call("lookup_manual", {"device": "washer", "query": "E4"})
        # Must NOT raise — read-only calls are repeatable.
        runner.engine.create_call("lookup_manual", {"device": "washer", "query": "E4"})

    def test_missing_device_needs_clarification(self):
        runner = make_runner()
        outcome = runner.run("lookup_manual", {})
        self.assertFalse(outcome.ok)
        self.assertTrue(outcome.needs_clarification)


class TestCreateBuiltinRunnerFlag(unittest.TestCase):
    def test_default_runner_does_not_include_troubleshooting_tools(self):
        runner = create_builtin_runner()
        self.assertNotIn("create_ticket", runner.tool_names())
        self.assertNotIn("lookup_manual", runner.tool_names())

    def test_flag_adds_both_tools(self):
        runner = create_builtin_runner(include_troubleshooting_demo_tools=True)
        self.assertIn("create_ticket", runner.tool_names())
        self.assertIn("lookup_manual", runner.tool_names())

    def test_flag_is_independent_of_flight_demo_flag(self):
        runner = create_builtin_runner(
            include_flight_demo_tools=True, include_troubleshooting_demo_tools=True
        )
        self.assertEqual(
            set(runner.tool_names()),
            {
                "calculator",
                "current_time",
                "search_flights",
                "book_flight",
                "create_ticket",
                "lookup_manual",
            },
        )


if __name__ == "__main__":
    unittest.main()
