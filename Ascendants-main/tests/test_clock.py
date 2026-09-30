import time
import unittest

from agent.clock import REAL_CLOCK, RealClock, VirtualClock


class TestRealClock(unittest.TestCase):
    def test_now_is_close_to_wall_clock_time(self):
        clock = RealClock()
        before = time.time()
        got = clock.now()
        after = time.time()
        self.assertTrue(before <= got <= after)

    def test_real_clock_singleton_behaves_the_same_as_a_fresh_instance(self):
        before = time.time()
        got = REAL_CLOCK.now()
        after = time.time()
        self.assertTrue(before <= got <= after)


class TestVirtualClock(unittest.TestCase):
    def test_starts_at_zero_by_default(self):
        clock = VirtualClock()
        self.assertEqual(clock.now(), 0.0)

    def test_starts_at_given_value(self):
        clock = VirtualClock(start=100.0)
        self.assertEqual(clock.now(), 100.0)

    def test_rejects_negative_start(self):
        with self.assertRaises(ValueError):
            VirtualClock(start=-1.0)

    def test_never_advances_on_its_own(self):
        clock = VirtualClock()
        self.assertEqual(clock.now(), 0.0)
        self.assertEqual(clock.now(), 0.0)  # calling now() repeatedly doesn't move it

    def test_tick_advances_by_the_given_amount(self):
        clock = VirtualClock()
        self.assertEqual(clock.tick(5.0), 5.0)
        self.assertEqual(clock.now(), 5.0)
        clock.tick(2.5)
        self.assertEqual(clock.now(), 7.5)

    def test_tick_default_is_one_second(self):
        clock = VirtualClock()
        clock.tick()
        self.assertEqual(clock.now(), 1.0)

    def test_tick_rejects_negative_seconds(self):
        clock = VirtualClock()
        with self.assertRaises(ValueError):
            clock.tick(-1.0)

    def test_set_jumps_to_an_absolute_time(self):
        clock = VirtualClock(start=10.0)
        self.assertEqual(clock.set(50.0), 50.0)
        self.assertEqual(clock.now(), 50.0)

    def test_set_rejects_moving_backwards(self):
        clock = VirtualClock(start=10.0)
        with self.assertRaises(ValueError):
            clock.set(5.0)

    def test_two_independent_virtual_clocks_do_not_share_state(self):
        a = VirtualClock()
        b = VirtualClock()
        a.tick(100.0)
        self.assertEqual(a.now(), 100.0)
        self.assertEqual(b.now(), 0.0)


if __name__ == "__main__":
    unittest.main()
