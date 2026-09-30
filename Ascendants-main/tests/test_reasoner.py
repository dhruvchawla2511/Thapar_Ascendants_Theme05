import json
import threading
import unittest

from agent.llm_client import LLMTimeoutError
from agent.reasoner import ConversationHistory, LLMReasoner, parse_action
from agent.tool_engine import CallStatus
from agent.tools_builtin import create_builtin_runner


def final(text):
    return json.dumps({"action": "final", "response": text})


def tool(name, **arguments):
    return json.dumps({"action": "tool", "tool": name, "arguments": arguments})


class FakeLLM:
    """Returns scripted replies in order and records every call."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def chat(self, messages, *, json_mode=False):
        self.calls.append([dict(m) for m in messages])
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def make(replies):
    llm = FakeLLM(replies)
    runner = create_builtin_runner()
    return llm, runner, LLMReasoner(llm, runner)


USER = lambda t: [{"role": "user", "content": t}]  # noqa: E731


class TestParseAction(unittest.TestCase):
    def test_variants(self):
        self.assertEqual(parse_action(final("hi")).response, "hi")
        self.assertEqual(parse_action("```json\n" + final("hi") + "\n```").response, "hi")
        self.assertEqual(parse_action("Sure! " + final("hi") + " done").response, "hi")
        a = parse_action(tool("calculator", expression="1+1"))
        self.assertEqual((a.kind, a.tool, a.arguments), ("tool", "calculator", {"expression": "1+1"}))
        self.assertEqual(parse_action('{"response": "hey"}').response, "hey")

    def test_invalid(self):
        for raw in ["", "hello there", "[1,2]", '{"action":"tool"}', '{"action":"final"}']:
            self.assertIsNone(parse_action(raw), raw)


class TestReasoner(unittest.TestCase):
    def test_plain_answer(self):
        llm, _, r = make([final("Hello! How can I help?")])
        res = r.respond(USER("hello"))
        self.assertEqual(res.text, "Hello! How can I help?")
        self.assertEqual(res.tool_trace, [])
        self.assertFalse(res.cancelled)
        self.assertEqual(llm.calls[0][0]["role"], "system")
        self.assertIn("calculator", llm.calls[0][0]["content"])

    def test_calculator_flow(self):
        llm, runner, r = make([tool("calculator", expression="25 * 37"), final("It is 925.")])
        res = r.respond(USER("what is 25 * 37?"))
        self.assertEqual(res.text, "It is 925.")
        self.assertEqual(len(res.tool_trace), 1)
        self.assertEqual(res.tool_trace[0]["result"], "925")
        # The second LLM call must have received the real tool result.
        self.assertIn("TOOL_RESULT calculator: 925", llm.calls[1][-1]["content"])
        call = runner.engine.get_call("call_1")
        self.assertEqual(call.status, CallStatus.COMPLETED)

    def test_current_time_flow(self):
        llm, _, r = make([tool("current_time"), final("It is now.")])
        res = r.respond(USER("what time is it?"))
        self.assertEqual(res.tool_trace[0]["tool"], "current_time")
        self.assertTrue(res.tool_trace[0]["ok"])

    def test_invalid_json_then_retry_succeeds(self):
        llm, _, r = make(["oops not json", final("ok")])
        self.assertEqual(r.respond(USER("hi")).text, "ok")
        self.assertIn("not a valid JSON", llm.calls[1][-1]["content"])

    def test_invalid_json_twice_falls_back_to_raw_text(self):
        _, _, r = make(["plain words", "still plain"])
        self.assertEqual(r.respond(USER("hi")).text, "still plain")

    def test_unknown_tool_error_is_fed_back(self):
        llm, _, r = make([tool("run_shell", cmd="ls"), final("I can't do that.")])
        res = r.respond(USER("delete everything"))
        self.assertEqual(res.text, "I can't do that.")
        self.assertFalse(res.tool_trace[0]["ok"])
        self.assertIn("TOOL_ERROR", llm.calls[1][-1]["content"])

    def test_invalid_arguments_rejected_by_tool_engine(self):
        _, _, r = make([tool("calculator", wrong="1+1"), final("sorry")])
        res = r.respond(USER("math"))
        self.assertFalse(res.tool_trace[0]["ok"])

    def test_max_tool_iterations(self):
        _, _, r = make([tool("current_time")] * 10)
        res = r.respond(USER("loop"))
        self.assertEqual(len(res.tool_trace), 4)
        self.assertIsNotNone(res.error)

    def test_llm_error_is_handled(self):
        _, _, r = make([LLMTimeoutError("slow")])
        res = r.respond(USER("hi"))
        self.assertIn("slow", res.text)
        self.assertEqual(res.error, "slow")


class TestConversation(unittest.TestCase):
    def test_second_turn_sees_first_turn(self):
        llm, runner, r = make([final("Nice to meet you, Dhruv."), final("Your name is Dhruv.")])
        hist = ConversationHistory()
        hist.add("user", "My name is Dhruv.")
        a1 = r.respond(hist.snapshot())
        hist.add("assistant", a1.text)
        hist.add("user", "What is my name?")
        a2 = r.respond(hist.snapshot())
        sent = [m["content"] for m in llm.calls[1]]
        self.assertIn("My name is Dhruv.", sent)
        self.assertIn("Nice to meet you, Dhruv.", sent)
        self.assertEqual(a2.text, "Your name is Dhruv.")

    def test_history_truncates(self):
        h = ConversationHistory(max_messages=4)
        for i in range(10):
            h.add("user", str(i))
        snap = h.snapshot()
        self.assertEqual([m["content"] for m in snap], ["6", "7", "8", "9"])
        snap[0]["content"] = "mutated"  # snapshot is a copy
        self.assertEqual(h.snapshot()[0]["content"], "6")


class BlockingLLM:
    """Blocks inside chat() until released, like a slow model."""

    def __init__(self, reply):
        self.reply = reply
        self.started = threading.Event()
        self.release = threading.Event()

    def chat(self, messages, *, json_mode=False):
        self.started.set()
        self.release.wait(timeout=5)
        return self.reply


class TestStaleResults(unittest.TestCase):
    def test_late_result_of_obsolete_task_is_cancelled(self):
        llm = BlockingLLM(final("Flights to Delhi"))
        reasoner = LLMReasoner(llm, create_builtin_runner())
        current = {"task": "A"}
        out = {}

        def run_task_a():
            out["A"] = reasoner.respond(USER("delhi"), is_current=lambda: current["task"] == "A")

        t = threading.Thread(target=run_task_a)
        t.start()
        self.assertTrue(llm.started.wait(timeout=2))
        current["task"] = "B"  # user interrupts: task B is now current
        llm.release.set()      # A's slow answer arrives late
        t.join(timeout=5)
        self.assertTrue(out["A"].cancelled)
        self.assertEqual(out["A"].text, "")

    def test_task_becoming_stale_before_tool_run_skips_the_tool(self):
        llm = FakeLLM([tool("calculator", expression="1+1")])
        runner = create_builtin_runner()
        reasoner = LLMReasoner(llm, runner)
        checks = iter([True, True, False])  # ok, ok (after LLM), stale before tool
        res = reasoner.respond(USER("x"), is_current=lambda: next(checks))
        self.assertTrue(res.cancelled)
        self.assertEqual(res.tool_trace, [])
        self.assertEqual(runner.engine.pending_calls(), [])

    def test_tool_finished_but_task_became_stale_marks_call_stale(self):
        llm = FakeLLM([tool("calculator", expression="1+1")])
        runner = create_builtin_runner()
        reasoner = LLMReasoner(llm, runner)
        checks = iter([True, True, True, False])  # stale only after tool ran
        res = reasoner.respond(USER("x"), is_current=lambda: next(checks))
        self.assertTrue(res.cancelled)
        self.assertEqual(runner.engine.get_call("call_1").status, CallStatus.STALE)

    def test_current_task_is_not_cancelled(self):
        _, _, r = make([final("fine")])
        self.assertFalse(r.respond(USER("x"), is_current=lambda: True).cancelled)


if __name__ == "__main__":
    unittest.main()
