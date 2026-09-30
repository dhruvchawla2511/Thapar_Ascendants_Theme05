import unittest
from datetime import datetime, timezone

from agent.tool_engine import CallStatus, ToolEngine
from agent.tools_builtin import (
    ToolExecutionError,
    create_builtin_runner,
    safe_calculate,
)


class TestCalculator(unittest.TestCase):
    def test_basic(self):
        self.assertEqual(safe_calculate("25 * 37"), "925")
        self.assertEqual(safe_calculate("2 + 3 * 4"), "14")
        self.assertEqual(safe_calculate("(2 + 3) * 4"), "20")
        self.assertEqual(safe_calculate("-5 + 2"), "-3")
        self.assertEqual(safe_calculate("7 // 2"), "3")
        self.assertEqual(safe_calculate("7 % 4"), "3")
        self.assertEqual(safe_calculate("2 ** 10"), "1024")

    def test_floats_and_friendly_symbols(self):
        self.assertEqual(safe_calculate("10 / 4"), "2.5")
        self.assertEqual(safe_calculate("10 / 2"), "5")
        self.assertEqual(safe_calculate("0.1 + 0.2"), "0.3")
        self.assertEqual(safe_calculate("25 × 37"), "925")
        self.assertEqual(safe_calculate("2 ^ 3"), "8")
        self.assertEqual(safe_calculate("25 * 37 ="), "925")

    def test_errors(self):
        bad = [
            "1 / 0", "", "   ", "abc", "__import__('os').system('ls')",
            "open('x')", "(1).real", "[1,2]", "'a' * 3", "1j", "True + 1",
            "2 ** 100000", "9 ** 9 ** 9", "(-8) ** 0.5", "1 +", "x = 5",
            "1" * 201, "0 ** -1",
        ]
        for expr in bad:
            with self.subTest(expr=expr):
                with self.assertRaises(ToolExecutionError):
                    safe_calculate(expr)

    def test_deep_nesting_does_not_crash(self):
        try:
            safe_calculate("(" * 500 + "1" + ")" * 500)
        except ToolExecutionError:
            pass


class TestCurrentTime(unittest.TestCase):
    def setUp(self):
        fixed = datetime(2026, 9, 22, 10, 30, 0, tzinfo=timezone.utc)
        self.runner = create_builtin_runner(
            now_fn=lambda tz: fixed.astimezone(tz) if tz else fixed
        )

    def test_default_timezone(self):
        out = self.runner.run("current_time", {})
        self.assertTrue(out.ok)
        self.assertIn("2026", out.text)
        self.assertIn("September", out.text)

    def test_named_timezone(self):
        out = self.runner.run("current_time", {"timezone": "Asia/Kolkata"})
        self.assertTrue(out.ok)
        self.assertIn("16:00:00", out.text)  # 10:30 UTC = 16:00 IST

    def test_invalid_timezone(self):
        out = self.runner.run("current_time", {"timezone": "Mars/Base"})
        self.assertFalse(out.ok)
        self.assertIn("Unknown timezone", out.text)

    def test_null_optional_argument_is_accepted(self):
        self.assertTrue(self.runner.run("current_time", {"timezone": None}).ok)


class TestToolRunnerValidation(unittest.TestCase):
    def setUp(self):
        self.engine = ToolEngine()
        self.runner = create_builtin_runner(self.engine)

    def test_success_is_tracked_in_engine(self):
        out = self.runner.run("calculator", {"expression": "25 * 37"})
        self.assertTrue(out.ok)
        self.assertEqual(out.text, "925")
        call = self.engine.get_call(out.call_id)
        self.assertEqual(call.status, CallStatus.COMPLETED)
        self.assertEqual(call.result, "925")

    def test_missing_argument_rejected_before_handler(self):
        out = self.runner.run("calculator", {})
        self.assertFalse(out.ok)
        self.assertIn("Missing required argument", out.text)
        self.assertIsNone(out.call_id)

    def test_wrong_type_rejected(self):
        out = self.runner.run("calculator", {"expression": ["1+1"]})
        self.assertFalse(out.ok)

    def test_number_is_coerced_to_string(self):
        out = self.runner.run("calculator", {"expression": 5})
        self.assertTrue(out.ok)
        self.assertEqual(out.text, "5")

    def test_extra_argument_rejected(self):
        out = self.runner.run("calculator", {"expression": "1+1", "shell": "ls"})
        self.assertFalse(out.ok)
        self.assertIn("Unknown arguments", out.text)

    def test_unknown_tool_rejected(self):
        out = self.runner.run("run_shell", {"cmd": "ls"})
        self.assertFalse(out.ok)
        self.assertIn("Unknown tool", out.text)

    def test_non_dict_arguments_rejected(self):
        self.assertFalse(self.runner.run("calculator", "25*37").ok)

    def test_tool_error_is_reported_not_raised(self):
        out = self.runner.run("calculator", {"expression": "1/0"})
        self.assertFalse(out.ok)
        self.assertIn("Division by zero", out.text)

    def test_no_dangerous_tools_registered(self):
        self.assertEqual(sorted(self.runner.tool_names()), ["calculator", "current_time"])

    def test_describe_lists_tools(self):
        text = self.runner.describe()
        self.assertIn("calculator(expression: str)", text)
        self.assertIn("current_time(timezone: str (optional))", text)


if __name__ == "__main__":
    unittest.main()
