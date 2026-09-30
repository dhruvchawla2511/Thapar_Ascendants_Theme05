import unittest

from agent.state_manager import StateManager


class TestStateManager(unittest.TestCase):
    def test_update_and_get_slot(self):
        sm = StateManager(session_id="s1")
        sm.update_slot("destination", "Mumbai")
        self.assertEqual(sm.get_slot("destination"), "Mumbai")
        self.assertIsNone(sm.get_slot("missing"))
        self.assertEqual(sm.get_slot("missing", "default"), "default")

    def test_update_slot_rejects_empty_name(self):
        sm = StateManager()
        with self.assertRaises(ValueError):
            sm.update_slot("", "x")

    def test_update_slots_bulk(self):
        sm = StateManager()
        sm.update_slots({"origin": "Delhi", "destination": "Mumbai"})
        self.assertEqual(sm.get_slot("origin"), "Delhi")
        self.assertEqual(sm.get_slot("destination"), "Mumbai")

    def test_localized_correction_preserves_other_slots(self):
        """The spec requires 'localized slot corrections' — overwriting one
        slot must never disturb the others already collected this session.
        """
        sm = StateManager()
        sm.update_slots({"origin": "Delhi", "destination": "Mumbai", "date": "2026-10-01"})
        sm.update_slot("destination", "Pune")  # a correction, e.g. barge-in
        self.assertEqual(sm.get_slot("origin"), "Delhi")
        self.assertEqual(sm.get_slot("date"), "2026-10-01")
        self.assertEqual(sm.get_slot("destination"), "Pune")

    def test_set_and_get_intent(self):
        sm = StateManager()
        self.assertIsNone(sm.get_intent())
        sm.set_intent("book_flight")
        self.assertEqual(sm.get_intent(), "book_flight")

    def test_remove_slot(self):
        sm = StateManager()
        sm.update_slot("x", 1)
        sm.remove_slot("x")
        self.assertIsNone(sm.get_slot("x"))
        sm.remove_slot("does_not_exist")  # must not raise

    def test_snapshot_is_a_defensive_copy(self):
        sm = StateManager()
        sm.update_slot("a", 1)
        snap = sm.get_snapshot()
        sm.update_slot("a", 2)
        sm.update_slot("b", 3)
        self.assertEqual(snap.slots, {"a": 1})  # snapshot frozen at capture time

    def test_snapshot_model_dump_shape(self):
        sm = StateManager()
        sm.set_intent("search_flights")
        sm.update_slot("destination", "Goa")
        dumped = sm.get_snapshot().model_dump()
        self.assertEqual(dumped["intent"], "search_flights")
        self.assertEqual(dumped["slots"], {"destination": "Goa"})
        self.assertIn("last_updated", dumped)

    def test_reset_clears_everything(self):
        sm = StateManager()
        sm.update_slots({"a": 1, "b": 2})
        sm.set_intent("x")
        sm.reset()
        self.assertIsNone(sm.get_intent())
        self.assertEqual(sm.get_snapshot().slots, {})

    def test_virtual_clock_drives_last_updated_deterministically(self):
        """Spec §4: 'Virtual Clock Streaming Harness'. Injecting a
        VirtualClock means last_updated is fully reproducible run to run,
        instead of depending on real wall-clock jitter.
        """
        from agent.clock import VirtualClock

        clock = VirtualClock(start=100.0)
        sm = StateManager(clock=clock)
        self.assertEqual(sm.get_snapshot().last_updated, 100.0)

        clock.tick(5.0)
        sm.update_slot("destination", "Goa")
        self.assertEqual(sm.get_snapshot().last_updated, 105.0)

        clock.tick(2.5)
        sm.set_intent("booking")
        self.assertEqual(sm.get_snapshot().last_updated, 107.5)

    def test_two_state_managers_sharing_one_virtual_clock_agree_on_now(self):
        from agent.clock import VirtualClock

        clock = VirtualClock()
        a = StateManager(session_id="a", clock=clock)
        b = StateManager(session_id="b", clock=clock)
        clock.tick(10.0)
        a.update_slot("x", 1)
        b.update_slot("y", 2)
        self.assertEqual(a.get_snapshot().last_updated, b.get_snapshot().last_updated)


if __name__ == "__main__":
    unittest.main()
