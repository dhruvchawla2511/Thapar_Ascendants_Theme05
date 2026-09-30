import unittest

from agent.tool_engine import (
    CallStatus,
    DuplicateStateChangingCallError,
    InvalidArgumentsError,
    ParamSpec,
    ToolEngine,
    ToolSpec,
    UnknownToolError,
)


def make_engine() -> ToolEngine:
    engine = ToolEngine()
    engine.register_tool(
        ToolSpec(
            name="search_flights",
            description="Search for flights to a destination.",
            parameters=(ParamSpec("destination", str),),
            state_changing=False,
        )
    )
    engine.register_tool(
        ToolSpec(
            name="book_flight",
            description="Book a flight (mutates real-world state).",
            parameters=(
                ParamSpec("destination", str),
                ParamSpec("date", str, required=False),
            ),
            state_changing=True,
        )
    )
    return engine


class TestToolEngine(unittest.TestCase):
    def test_registers_a_tool_and_reports_read_only_vs_state_changing(self):
        engine = make_engine()
        self.assertFalse(engine.is_state_changing("search_flights"))
        self.assertTrue(engine.is_state_changing("book_flight"))

    def test_unknown_tool_raises(self):
        engine = make_engine()
        with self.assertRaises(UnknownToolError):
            engine.is_state_changing("teleport")

    def test_creates_a_call_with_unique_call_id(self):
        engine = make_engine()
        call1 = engine.create_call("search_flights", {"destination": "Mumbai"})
        call2 = engine.create_call("search_flights", {"destination": "Delhi"})
        self.assertNotEqual(call1.call_id, call2.call_id)
        self.assertEqual(call1.status, CallStatus.PENDING)
        self.assertEqual(call1.to_dict()["action"], "tool_call")
        self.assertEqual(call1.to_dict()["call_id"], call1.call_id)

    def test_validates_arguments_missing_required(self):
        engine = make_engine()
        with self.assertRaises(InvalidArgumentsError):
            engine.create_call("search_flights", {})

    def test_validates_arguments_wrong_type(self):
        engine = make_engine()
        with self.assertRaises(InvalidArgumentsError):
            engine.create_call("search_flights", {"destination": 123})

    def test_validates_arguments_unknown_arg(self):
        engine = make_engine()
        with self.assertRaises(InvalidArgumentsError):
            engine.create_call("search_flights", {"destination": "Mumbai", "extra": True})

    def test_prevents_duplicate_pending_state_changing_call(self):
        engine = make_engine()
        engine.create_call("book_flight", {"destination": "Mumbai"})
        with self.assertRaises(DuplicateStateChangingCallError):
            engine.create_call("book_flight", {"destination": "Mumbai"})

    def test_duplicate_state_changing_call_allowed_after_first_completes(self):
        engine = make_engine()
        call = engine.create_call("book_flight", {"destination": "Mumbai"})
        engine.complete_call(call.call_id, result={"confirmation": "XYZ"})
        call2 = engine.create_call("book_flight", {"destination": "Mumbai"})
        self.assertNotEqual(call2.call_id, call.call_id)

    def test_read_only_calls_are_never_deduplicated(self):
        engine = make_engine()
        engine.create_call("search_flights", {"destination": "Mumbai"})
        call2 = engine.create_call("search_flights", {"destination": "Mumbai"})
        self.assertEqual(call2.status, CallStatus.PENDING)

    def test_cancelling_a_call_then_completing_it_keeps_it_cancelled(self):
        engine = make_engine()
        call = engine.create_call("search_flights", {"destination": "Delhi"})
        engine.cancel_call(call.call_id)
        self.assertEqual(engine.get_call(call.call_id).status, CallStatus.CANCELLED)

        completed = engine.complete_call(call.call_id, result={"flights": ["DL123"]})
        self.assertEqual(completed.status, CallStatus.CANCELLED)
        self.assertEqual(completed.result, {"flights": ["DL123"]})

    def test_cancelling_a_state_changing_call_frees_it_for_retry(self):
        engine = make_engine()
        call = engine.create_call("book_flight", {"destination": "Mumbai"})
        engine.cancel_call(call.call_id)
        call2 = engine.create_call("book_flight", {"destination": "Mumbai"})
        self.assertNotEqual(call2.call_id, call.call_id)

    def test_mark_stale_on_completed_call(self):
        engine = make_engine()
        call = engine.create_call("search_flights", {"destination": "Delhi"})
        engine.complete_call(call.call_id, result={"flights": ["DL1"]})
        engine.mark_stale(call.call_id)
        self.assertEqual(engine.get_call(call.call_id).status, CallStatus.STALE)

    def test_pending_calls_lists_only_pending(self):
        engine = make_engine()
        call1 = engine.create_call("search_flights", {"destination": "Delhi"})
        call2 = engine.create_call("search_flights", {"destination": "Mumbai"})
        engine.complete_call(call1.call_id, result={})
        pending = engine.pending_calls()
        self.assertEqual([c.call_id for c in pending], [call2.call_id])


if __name__ == "__main__":
    unittest.main()
