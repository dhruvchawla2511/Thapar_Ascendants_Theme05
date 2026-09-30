"""
fault_injection.py — reusable, deterministic latency/fault injection for
tool handlers.

Spec §4 describes the Mock Environment as: "Deterministic latency and
fault injection for flight search, booking, ticket creation, and frame-
grounded manual lookups." Before this module existed, the only place
this codebase demonstrated ANY latency injection was one hand-rolled
`time.sleep()` call inside a single demo-only book_flight handler
(`_slow_book_flight_handler` in agent/main.py) — not reusable, not
applicable to any other tool, and not testable without a real wall-clock
wait.

wrap_with_faults() wraps ANY existing tool handler with a FaultPlan: an
explicit, ordered schedule of what should happen on the 1st call, 2nd
call, and so on — a fixed latency to (fake-)sleep for, and/or an
exception to raise instead of running the real handler. "Deterministic"
here is literal: nothing here is randomized. The same plan produces the
exact same sequence of behavior every single run, which is what makes it
usable both in a live demo (real time.sleep) and in a test (an injected,
non-blocking fake sleep_fn that just records what would have happened).

apply_fault_plan() is the convenience most callers actually want: wrap
whatever handler is CURRENTLY registered for a tool name on a live
ToolRunner, in place, without needing to hold onto the original handler
reference yourself.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

SleepFn = Callable[[float], None]


@dataclass(frozen=True)
class FaultStep:
    """What should happen on one specific call to a wrapped handler.

    latency_seconds: how long to (fake-)sleep before running the real
      handler (or before raising, if `exception` is also set). 0 means no
      injected delay.
    exception: if set, raised instead of ever calling the real handler.
      Stored as a pre-built exception instance so a FaultPlan can name the
      exact error type/message a scenario expects to see (e.g. a
      ToolExecutionError with a specific, assertable message).
    """

    latency_seconds: float = 0.0
    exception: Optional[BaseException] = None

    def __post_init__(self) -> None:
        if self.latency_seconds < 0:
            raise ValueError("latency_seconds must be >= 0")


@dataclass
class FaultPlan:
    """An ordered, 1-indexed schedule of FaultSteps: `steps[0]` applies to
    the 1st call the wrapped handler receives, `steps[1]` to the 2nd, and
    so on. Once the plan is exhausted, every subsequent call proceeds
    normally (latency_seconds=0, no exception) — a plan only needs to
    describe the calls it wants to affect, not every call a scenario will
    ever make.
    """

    steps: list[FaultStep] = field(default_factory=list)

    def step_for_call(self, call_number: int) -> FaultStep:
        """call_number is 1-indexed (the handler's 1st invocation ever)."""
        if call_number < 1:
            raise ValueError("call_number must be >= 1")
        if call_number <= len(self.steps):
            return self.steps[call_number - 1]
        return FaultStep()

    @classmethod
    def fail_then_succeed(cls, exception: BaseException, times: int = 1) -> "FaultPlan":
        """Convenience constructor for the single most common pattern in
        the spec's own scenario list ("retries"): fail the first `times`
        call(s) with `exception`, succeed on every call after that.
        """
        return cls(steps=[FaultStep(exception=exception) for _ in range(times)])

    @classmethod
    def latency_then_normal(cls, latency_seconds: float, times: int = 1) -> "FaultPlan":
        """Convenience constructor: inject `latency_seconds` of delay on
        the first `times` call(s), then proceed with no injected delay.
        """
        return cls(steps=[FaultStep(latency_seconds=latency_seconds) for _ in range(times)])


def wrap_with_faults(
    handler: Callable[[dict], Any],
    plan: FaultPlan,
    sleep_fn: SleepFn = time.sleep,
) -> Callable[[dict], Any]:
    """Return a NEW handler wrapping `handler` with `plan`'s deterministic
    latency/fault schedule.

    Each call to wrap_with_faults() gets its own independent, private call
    counter — wrapping the same underlying handler twice (e.g. for two
    different tool names that happen to share an implementation) produces
    two completely independent schedules.

    `sleep_fn` defaults to real `time.sleep` (correct for production/demo
    use, where the handler already runs off the event loop thread via
    `asyncio.to_thread`); tests can inject a fake that just records the
    requested duration instead of actually blocking.
    """
    call_count = {"n": 0}

    def wrapped(args: dict) -> Any:
        call_count["n"] += 1
        step = plan.step_for_call(call_count["n"])
        if step.latency_seconds > 0:
            sleep_fn(step.latency_seconds)
        if step.exception is not None:
            raise step.exception
        return handler(args)

    return wrapped


def apply_fault_plan(
    runner: Any,  # agent.tools_builtin.ToolRunner (duck-typed to avoid a cycle)
    tool_name: str,
    plan: FaultPlan,
    sleep_fn: SleepFn = time.sleep,
) -> None:
    """Wrap the handler CURRENTLY registered for `tool_name` on `runner`
    with `plan`, replacing it in place. The tool's schema (ToolSpec —
    parameters, state_changing, slot_updates) is untouched; only the
    handler changes. Raises KeyError if `tool_name` isn't registered.
    """
    spec = runner.get_spec(tool_name)
    handler = runner.get_handler(tool_name)
    if spec is None or handler is None:
        raise KeyError(f"No such tool registered: {tool_name!r}")
    runner.register(spec, wrap_with_faults(handler, plan, sleep_fn=sleep_fn))
