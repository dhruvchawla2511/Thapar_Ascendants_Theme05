import unittest

from agent.fast_path import FastPath, is_safe_fast_path_message


class TestFastPath(unittest.TestCase):
    def test_produces_an_acknowledgement(self):
        fp = FastPath()
        msg = fp.acknowledge("I want to book a flight to Mumbai.")
        self.assertIsInstance(msg, str)
        self.assertGreater(len(msg), 0)

    def test_does_not_falsely_claim_completion(self):
        fp = FastPath()
        msg = fp.acknowledge("I want to book a flight to Mumbai.")
        self.assertTrue(is_safe_fast_path_message(msg))
        self.assertNotIn("booked", msg.lower())
        self.assertNotIn("confirmed", msg.lower())

    def test_interruption_acknowledgement_is_also_safe(self):
        fp = FastPath()
        fp.acknowledge("Book me a flight to Delhi.")
        msg = fp.acknowledge_interruption("Actually Mumbai.")
        self.assertTrue(is_safe_fast_path_message(msg))

    def test_is_safe_fast_path_message_catches_bad_messages(self):
        self.assertFalse(is_safe_fast_path_message("Your flight is booked!"))
        self.assertFalse(is_safe_fast_path_message("Order confirmed."))
        self.assertTrue(is_safe_fast_path_message("Got it, checking that now."))

    def test_does_not_always_repeat_identical_filler(self):
        fp = FastPath()
        seen = {fp.acknowledge("book a flight") for _ in range(20)}
        self.assertGreater(len(seen), 1)


if __name__ == "__main__":
    unittest.main()
