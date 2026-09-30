"""LLM reasoning component for Ascendants.

The model must answer with ONE JSON object per step:
    {"action": "final", "response": "<text for the user>"}
    {"action": "tool", "tool": "<name>", "arguments": {...}}

The reasoner loops: ask the LLM -> if it wants a tool, run it through the
ToolRunner (which goes through ToolEngine validation) -> feed the TOOL_RESULT
back -> repeat until a final answer.

Staleness: `is_current` is a callable (backed by InterruptHandler in the real
agent). It is checked before every LLM call, before every tool run, and before
returning. If the task became obsolete, the result is marked `cancelled` and
the caller must discard it.
"""

from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass, field
from typing import Any, Callable

from agent.llm_client import LLMClient, LLMError
from agent.tools_builtin import ToolRunner

SYSTEM_PROMPT_TEMPLATE = """You are Ascendants, a helpful local AI assistant.
You must reply with EXACTLY ONE JSON object and nothing else.

To answer the user, reply with:
{"action": "final", "response": "<your answer to the user>"}

To use a tool, reply with:
{"action": "tool", "tool": "<tool name>", "arguments": {<arguments>}}

Available tools:
<<TOOLS>>

Rules:
- ALWAYS use the calculator tool for any arithmetic. Never calculate in your head.
- ALWAYS use the current_time tool for questions about the current time or date. Never guess.
- If a tool such as search_flights or book_flight is available and the user asks for it, use the tool — never invent flight numbers, prices, or booking confirmations yourself.
- If the user corrects a detail you already used (e.g. a different destination), use the corrected value in your next tool call rather than the old one.
- After a TOOL_RESULT message arrives, use it to write the final answer. Never invent tool results.
- For greetings, general questions, and anything the user told you earlier in this conversation, answer directly with a final action.
- Keep answers short and friendly.

Examples:
User: hello
{"action": "final", "response": "Hello! How can I help?"}

User: what is 25 * 37?
{"action": "tool", "tool": "calculator", "arguments": {"expression": "25 * 37"}}
TOOL_RESULT calculator: 925
{"action": "final", "response": "25 * 37 = 925."}

User: what time is it?
{"action": "tool", "tool": "current_time", "arguments": {}}
TOOL_RESULT current_time: Monday, 01 January 2024, 10:30:00 IST (UTC+0530)
{"action": "final", "response": "It is 10:30 AM on Monday, 1 January 2024."}
"""

RETRY_NUDGE = (
    "Your last reply was not a valid JSON action. Reply with exactly one JSON "
    'object: {"action": "final", "response": "..."} or '
    '{"action": "tool", "tool": "...", "arguments": {...}}.'
)


@dataclass
class ReasonResult:
    text: str = ""
    tool_trace: list[dict] = field(default_factory=list)
    error: str | None = None
    cancelled: bool = False


@dataclass
class _Action:
    kind: str  # "final" | "tool"
    response: str = ""
    tool: str = ""
    arguments: dict = field(default_factory=dict)


class ConversationHistory:
    """Thread-safe, size-limited list of {"role", "content"} messages."""

    def __init__(self, max_messages: int = 20) -> None:
        self.max_messages = max_messages
        self._messages: list[dict] = []
        self._lock = threading.Lock()

    def add(self, role: str, content: str) -> None:
        with self._lock:
            self._messages.append({"role": role, "content": content})
            if len(self._messages) > self.max_messages:
                self._messages = self._messages[-self.max_messages :]

    def snapshot(self) -> list[dict]:
        with self._lock:
            return [dict(m) for m in self._messages]

    def clear(self) -> None:
        with self._lock:
            self._messages.clear()


_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def parse_action(raw: str) -> _Action | None:
    """Turn the model's raw text into an _Action, or None if it is not valid."""
    text = _FENCE_RE.sub("", (raw or "").strip()).strip()
    data: Any = None
    try:
        data = json.loads(text)
    except ValueError:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            try:
                data = json.loads(text[start : end + 1])
            except ValueError:
                data = None
    if not isinstance(data, dict):
        return None

    action = str(data.get("action", "")).lower()
    if action == "tool" or (not action and "tool" in data):
        name = data.get("tool") or data.get("name")
        args = data.get("arguments", data.get("args", data.get("parameters", {})))
        if isinstance(name, str) and name:
            return _Action("tool", tool=name, arguments=args if isinstance(args, dict) else {})
        return None
    response = data.get("response", data.get("text", data.get("answer")))
    if action == "final" or (not action and response is not None):
        if isinstance(response, str):
            return _Action("final", response=response.strip())
    return None


class LLMReasoner:
    def __init__(
        self,
        llm: LLMClient,
        runner: ToolRunner,
        max_tool_iterations: int = 4,
    ) -> None:
        self.llm = llm
        self.runner = runner
        self.max_tool_iterations = max_tool_iterations

    def system_prompt(self) -> str:
        return SYSTEM_PROMPT_TEMPLATE.replace("<<TOOLS>>", self.runner.describe())

    def respond(
        self,
        messages: list[dict],
        is_current: Callable[[], bool] | None = None,
    ) -> ReasonResult:
        """`messages` = conversation so far, ending with the current user message."""
        current = is_current or (lambda: True)
        convo: list[dict] = [{"role": "system", "content": self.system_prompt()}, *messages]
        trace: list[dict] = []
        tools_used = 0
        retried = False

        while True:
            if not current():
                return ReasonResult(tool_trace=trace, cancelled=True)
            try:
                raw = self.llm.chat(convo, json_mode=True)
            except LLMError as exc:
                return ReasonResult(
                    text=f"Sorry, I couldn't get an answer from the language model: {exc}",
                    tool_trace=trace,
                    error=str(exc),
                )
            if not current():
                return ReasonResult(tool_trace=trace, cancelled=True)

            action = parse_action(raw)
            if action is None:
                if retried:
                    # Give up on JSON; show the model's plain text instead.
                    return ReasonResult(text=(raw or "").strip() or "Sorry, I have no answer.", tool_trace=trace)
                retried = True
                convo += [
                    {"role": "assistant", "content": raw},
                    {"role": "user", "content": RETRY_NUDGE},
                ]
                continue

            if action.kind == "final":
                return ReasonResult(text=action.response, tool_trace=trace)

            # ---- tool request ----
            if tools_used >= self.max_tool_iterations:
                return ReasonResult(
                    text="Sorry, I couldn't finish that within the allowed number of tool steps.",
                    tool_trace=trace,
                    error="max tool iterations reached",
                )
            if not current():
                return ReasonResult(tool_trace=trace, cancelled=True)

            outcome = self.runner.run(action.tool, action.arguments)
            tools_used += 1
            trace.append(
                {
                    "tool": action.tool,
                    "arguments": action.arguments,
                    "ok": outcome.ok,
                    "result": outcome.text,
                }
            )
            if not current():
                self.runner.mark_stale(outcome.call_id)
                return ReasonResult(tool_trace=trace, cancelled=True)

            label = "TOOL_RESULT" if outcome.ok else "TOOL_ERROR"
            convo += [
                {"role": "assistant", "content": raw},
                {
                    "role": "user",
                    "content": f"{label} {action.tool}: {outcome.text}\n"
                    "Now reply with a final JSON action using this result"
                    + ("." if outcome.ok else ", or fix the arguments and call the tool again."),
                },
            ]
