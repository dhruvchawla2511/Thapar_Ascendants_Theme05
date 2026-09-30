import unittest

from agent.trace import TraceRecorder


class TestTraceRecorder(unittest.TestCase):
    def test_records_input_and_action_events(self):
        t = TraceRecorder(session_id="s1")
        t.input("text_chunk", text="hi")
        t.action_emitted("final", call_id="turn_1", response="ok")
        events = t.events()
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0].kind, "input")
        self.assertEqual(events[0].event_type, "text_chunk")
        self.assertEqual(events[1].kind, "action")
        self.assertEqual(events[1].action, "final")
        self.assertEqual(events[1].call_id, "turn_1")

    def test_every_event_carries_session_id_and_timestamp(self):
        t = TraceRecorder(session_id="s42")
        e = t.decision("superseded", call_id="call_1", status="cancelled")
        self.assertEqual(e.session_id, "s42")
        self.assertIsInstance(e.timestamp, float)
        self.assertGreater(e.timestamp, 0)

    def test_for_call_filters_by_call_id(self):
        t = TraceRecorder()
        t.action_emitted("tool_call", call_id="call_1")
        t.action_emitted("tool_call", call_id="call_2")
        t.decision("stale", call_id="call_1", status="cancelled")
        entries = t.for_call("call_1")
        self.assertEqual(len(entries), 2)
        self.assertTrue(all(e.call_id == "call_1" for e in entries))

    def test_to_jsonl_round_trips_through_json(self):
        import json

        t = TraceRecorder(session_id="s1")
        t.input("interruption", timestamp=1.0)
        t.decision("cancelled by user", call_id="call_9", status="cancelled")
        lines = t.to_jsonl().splitlines()
        self.assertEqual(len(lines), 2)
        parsed = json.loads(lines[1])
        self.assertEqual(parsed["call_id"], "call_9")
        self.assertEqual(parsed["reason"], "cancelled by user")

    def test_clear_empties_the_log(self):
        t = TraceRecorder()
        t.input("x")
        t.clear()
        self.assertEqual(t.events(), [])

    def test_two_recorders_are_fully_independent(self):
        """No global/shared state — matches the StateManager no-cross-
        session-cache rule this repo already follows.
        """
        a = TraceRecorder(session_id="a")
        b = TraceRecorder(session_id="b")
        a.input("text_chunk")
        self.assertEqual(len(a.events()), 1)
        self.assertEqual(len(b.events()), 0)

    def test_virtual_clock_drives_trace_timestamps_deterministically(self):
        """Spec §4: 'Virtual Clock Streaming Harness: Deterministic event
        replay... complete event/action trace logging.' A trace recorded
        against an injected VirtualClock is fully reproducible run to run.
        """
        from agent.clock import VirtualClock

        clock = VirtualClock(start=1000.0)
        t = TraceRecorder(session_id="s1", clock=clock)
        e1 = t.input("text_chunk")
        clock.tick(0.5)
        e2 = t.action_emitted("filler", call_id="filler_1")
        clock.tick(2.0)
        e3 = t.decision("cancelled by user", call_id="turn_1", status="cancelled")

        self.assertEqual(e1.timestamp, 1000.0)
        self.assertEqual(e2.timestamp, 1000.5)
        self.assertEqual(e3.timestamp, 1002.5)

    def test_two_recorders_sharing_one_virtual_clock_agree_on_now(self):
        from agent.clock import VirtualClock

        clock = VirtualClock()
        a = TraceRecorder(session_id="a", clock=clock)
        b = TraceRecorder(session_id="b", clock=clock)
        clock.tick(42.0)
        ea = a.input("x")
        eb = b.input("y")
        self.assertEqual(ea.timestamp, eb.timestamp)


if __name__ == "__main__":
    unittest.main()
