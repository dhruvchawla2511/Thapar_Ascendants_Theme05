"""Safe built-in tools + a ToolRunner that executes them THROUGH ToolEngine.

ToolEngine only validates and tracks calls; it never runs anything. ToolRunner
adds the missing piece: a name -> handler registry. Every execution goes:

    create_call (schema validation, duplicate protection)
        -> handler runs
        -> complete_call (status tracking, cancelled calls stay cancelled)

Tools provided: calculator (AST-based, no eval) and current_time.
There is deliberately NO shell, code-execution or filesystem tool here.
"""

from __future__ import annotations

import ast
import asyncio
import itertools
import math
import operator
from dataclasses import dataclass
from datetime import datetime, tzinfo
from typing import Any, Awaitable, Callable, Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from agent.tool_engine import (
    CallStatus,
    InvalidArgumentsError,
    ParamSpec,
    ToolCall,
    ToolEngine,
    ToolEngineError,
    ToolSpec,
)

Handler = Callable[[dict], Any]
OnCallCreated = Callable[[ToolCall], Awaitable[None]]


class ToolExecutionError(Exception):
    """A tool ran but could not produce a result (bad expression, bad timezone...)."""


@dataclass
class ToolOutcome:
    tool_name: str
    ok: bool
    text: str
    call_id: str | None = None
    # True specifically when the call was rejected because required
    # arguments were missing/invalid — as opposed to any other failure
    # (unknown tool, tool bug, duplicate). This is what lets the caller
    # distinguish "ask the user for the missing bit" from "just tell the
    # LLM to retry" (spec §"Clarification handling": missing required tool
    # argument -> clarification, not a silent LLM retry loop).
    needs_clarification: bool = False


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------


class ToolRunner:
    def __init__(self, engine: ToolEngine | None = None) -> None:
        self.engine = engine or ToolEngine()
        self._handlers: dict[str, Handler] = {}
        self._specs: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec, handler: Handler) -> None:
        self.engine.register_tool(spec)
        self._specs[spec.name] = spec
        self._handlers[spec.name] = handler

    def get_spec(self, tool_name: str) -> Optional[ToolSpec]:
        return self._specs.get(tool_name)

    def get_handler(self, tool_name: str) -> Optional[Handler]:
        """Public accessor for the handler currently registered under
        `tool_name` — lets a caller (e.g. fault_injection.apply_fault_plan)
        wrap or replace it without reaching into private state.
        """
        return self._handlers.get(tool_name)

    def tool_names(self) -> list[str]:
        return list(self._specs)

    def describe(self) -> str:
        """Human/LLM-readable tool list used in the system prompt."""
        lines = []
        for spec in self._specs.values():
            params = ", ".join(
                f"{p.name}: {p.type.__name__}" + ("" if p.required else " (optional)")
                for p in spec.parameters
            )
            lines.append(f"- {spec.name}({params}): {spec.description}")
        return "\n".join(lines)

    def run(self, tool_name: str, args: Any) -> ToolOutcome:
        """Validate via ToolEngine, execute, record. Never raises."""
        if tool_name not in self._handlers:
            return ToolOutcome(
                tool_name,
                False,
                f"Unknown tool '{tool_name}'. Available tools: {', '.join(self._specs)}.",
            )
        if not isinstance(args, dict):
            return ToolOutcome(tool_name, False, "Tool arguments must be a JSON object.")

        args = self._clean_args(self._specs[tool_name], args)
        try:
            call = self.engine.create_call(tool_name, args)  # validation happens here
        except InvalidArgumentsError as exc:
            return ToolOutcome(tool_name, False, str(exc), needs_clarification=True)
        except ToolEngineError as exc:
            return ToolOutcome(tool_name, False, str(exc))

        try:
            result = self._handlers[tool_name](dict(call.args))
        except ToolExecutionError as exc:
            self.engine.complete_call(call.call_id, {"error": str(exc)})
            return ToolOutcome(tool_name, False, str(exc), call.call_id)
        except Exception as exc:  # noqa: BLE001 - a tool bug must not crash the agent
            self.engine.complete_call(call.call_id, {"error": repr(exc)})
            return ToolOutcome(
                tool_name, False, f"Tool '{tool_name}' failed unexpectedly: {exc}", call.call_id
            )

        self.engine.complete_call(call.call_id, result)
        return ToolOutcome(tool_name, True, str(result), call.call_id)

    async def run_cancellable(
        self,
        tool_name: str,
        args: Any,
        interrupt_handler: Any,
        on_call_created: OnCallCreated | None = None,
    ) -> ToolOutcome:
        """Async counterpart to `run()` that makes ONE tool call a real,
        separately cancellable unit of work instead of an opaque blocking
        step inside a bigger synchronous call.

        - The call gets its schema-validated, fingerprint-guarded call_id
          exactly like `run()` (same ToolEngine, same duplicate-protection
          for state-changing tools) — nothing about validation or dedup
          changes.
        - The handler itself executes in a background asyncio.Task
          registered with `interrupt_handler` under `call.call_id` (its OWN
          id, distinct from whatever "turn" id the caller might also be
          registered under), so InterruptHandler can see and cancel this
          specific tool call.
        - `on_call_created`, if given, is awaited with the ToolCall right
          after it's created (before the handler runs) — this is the hook
          that lets a `tool_call` action be emitted immediately, satisfying
          "non-blocking tool calls (with explicit call_id)" as an output.
        - If the task is cancelled, the call is marked CANCELLED in the
          engine and the cancellation propagates to the caller. Because
          `ToolEngine.complete_call` refuses to resurrect an already-
          CANCELLED call, a stale result that finishes computing after
          cancellation (impossible to truly kill a raw thread) can never
          silently become a real result — it just complete_call()s onto a
          call that stays CANCELLED forever.

        Known limitation: because the caller `await`s this coroutine
        directly, cancelling JUST this call's call_id while leaving the
        caller's own task alive is not supported today — see
        agent/async_reasoner.py's module docstring.
        """
        if tool_name not in self._handlers:
            return ToolOutcome(
                tool_name,
                False,
                f"Unknown tool '{tool_name}'. Available tools: {', '.join(self._specs)}.",
            )
        if not isinstance(args, dict):
            return ToolOutcome(tool_name, False, "Tool arguments must be a JSON object.")

        args = self._clean_args(self._specs[tool_name], args)
        try:
            call = self.engine.create_call(tool_name, args)
        except InvalidArgumentsError as exc:
            return ToolOutcome(tool_name, False, str(exc), needs_clarification=True)
        except ToolEngineError as exc:
            return ToolOutcome(tool_name, False, str(exc))

        if on_call_created is not None:
            await on_call_created(call)

        handler = self._handlers[tool_name]
        task: "asyncio.Task[Any]" = asyncio.create_task(asyncio.to_thread(handler, dict(call.args)))
        interrupt_handler.register_task(call.call_id, task)
        try:
            result = await task
        except asyncio.CancelledError:
            # Unregister BEFORE re-raising: cancel_all_tasks() clears its
            # whole active_tasks dict regardless, so this only matters for
            # a targeted/timeout-driven cancellation that doesn't go
            # through cancel_all_tasks() — without this, this call_id
            # would stay in active_tasks forever, an invisible leak.
            interrupt_handler.unregister_task(call.call_id)
            self.engine.cancel_call(call.call_id)
            raise
        except ToolExecutionError as exc:
            interrupt_handler.unregister_task(call.call_id)
            self.engine.complete_call(call.call_id, {"error": str(exc)})
            return ToolOutcome(tool_name, False, str(exc), call.call_id)
        except Exception as exc:  # noqa: BLE001 - a tool bug must not crash the agent
            interrupt_handler.unregister_task(call.call_id)
            self.engine.complete_call(call.call_id, {"error": repr(exc)})
            return ToolOutcome(
                tool_name, False, f"Tool '{tool_name}' failed unexpectedly: {exc}", call.call_id
            )

        # Was this call's own entry removed from active_tasks (e.g. by a
        # targeted cancellation) even though the underlying handler thread
        # still ran to completion? If so, treat it as cancelled — stale-
        # result protection applies even without the CancelledError path.
        still_tracked = call.call_id in getattr(interrupt_handler, "active_tasks", {})
        interrupt_handler.unregister_task(call.call_id)
        if not still_tracked:
            self.engine.cancel_call(call.call_id)
            return ToolOutcome(tool_name, False, "Cancelled.", call.call_id)

        self.engine.complete_call(call.call_id, result)
        return ToolOutcome(tool_name, True, str(result), call.call_id)

    def mark_stale(self, call_id: str | None) -> None:
        if call_id:
            try:
                self.engine.mark_stale(call_id)
            except ToolEngineError:
                pass

    @staticmethod
    def _clean_args(spec: ToolSpec, args: dict) -> dict:
        """Forgive common small-model slips BEFORE validation:
        drop null values, and turn numbers into strings for str parameters."""
        cleaned = {k: v for k, v in args.items() if v is not None}
        for param in spec.parameters:
            value = cleaned.get(param.name)
            if (
                param.type is str
                and isinstance(value, (int, float))
                and not isinstance(value, bool)
            ):
                cleaned[param.name] = str(value)
        return cleaned


# --------------------------------------------------------------------------
# Calculator (AST based; never eval/exec)
# --------------------------------------------------------------------------

MAX_EXPRESSION_LENGTH = 200
MAX_EXPONENT = 1000
MAX_RESULT_BITS = 20000


def _safe_pow(a: Any, b: Any) -> Any:
    if abs(b) > MAX_EXPONENT:
        raise ToolExecutionError("Exponent too large.")
    if isinstance(a, int) and isinstance(b, int) and b > 0 and abs(a) > 1:
        if b * a.bit_length() > MAX_RESULT_BITS:
            raise ToolExecutionError("Result too large.")
    try:
        result = a**b
    except ZeroDivisionError as exc:
        raise ToolExecutionError("Division by zero.") from exc
    except OverflowError as exc:
        raise ToolExecutionError("Result too large.") from exc
    if isinstance(result, complex):
        raise ToolExecutionError("Result is not a real number.")
    return result


_BIN_OPS: dict[type, Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: _safe_pow,
}
_UNARY_OPS: dict[type, Callable[[Any], Any]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _eval_node(node: ast.AST) -> Any:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ToolExecutionError("Only plain numbers are allowed.")
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        return _BIN_OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_node(node.operand))
    raise ToolExecutionError(
        "Unsupported expression. Use only numbers, + - * / // % ** and parentheses."
    )


def _format_number(value: Any) -> str:
    if isinstance(value, int):
        try:
            return str(value)
        except ValueError as exc:  # Python's int->str digit limit
            raise ToolExecutionError("Result too large.") from exc
    if math.isnan(value) or math.isinf(value):
        raise ToolExecutionError("Result is not a finite number.")
    return format(value, ".10g")


def safe_calculate(expression: str) -> str:
    """Evaluate a plain arithmetic expression. Raises ToolExecutionError on any problem."""
    expr = (expression or "").strip().rstrip("=?").strip()
    if not expr:
        raise ToolExecutionError("Empty expression.")
    if len(expr) > MAX_EXPRESSION_LENGTH:
        raise ToolExecutionError(f"Expression too long (max {MAX_EXPRESSION_LENGTH} characters).")
    expr = expr.replace("×", "*").replace("÷", "/").replace("^", "**")
    try:
        tree = ast.parse(expr, mode="eval")
    except (SyntaxError, ValueError, RecursionError, MemoryError) as exc:
        raise ToolExecutionError("Could not parse that expression.") from exc
    try:
        return _format_number(_eval_node(tree))
    except ZeroDivisionError as exc:
        raise ToolExecutionError("Division by zero.") from exc
    except OverflowError as exc:
        raise ToolExecutionError("Result too large.") from exc
    except RecursionError as exc:
        raise ToolExecutionError("Expression too deeply nested.") from exc


def _calculator_handler(args: dict) -> str:
    return safe_calculate(args["expression"])


# --------------------------------------------------------------------------
# Current time
# --------------------------------------------------------------------------

NowFn = Callable[[tzinfo | None], datetime]


def _default_now(tz: tzinfo | None) -> datetime:
    return datetime.now(tz)


def make_current_time_handler(now_fn: NowFn | None = None) -> Handler:
    now_fn = now_fn or _default_now

    def handler(args: dict) -> str:
        tz_name = args.get("timezone")
        if tz_name:
            try:
                tz = ZoneInfo(tz_name)
            except (ZoneInfoNotFoundError, ValueError, KeyError, OSError) as exc:
                raise ToolExecutionError(
                    f"Unknown timezone '{tz_name}'. Use an IANA name like 'Asia/Kolkata'."
                ) from exc
            now = now_fn(tz)
        else:
            now = now_fn(None).astimezone()
        return now.strftime("%A, %d %B %Y, %H:%M:%S %Z (UTC%z)")

    return handler


# --------------------------------------------------------------------------
# Demo domain tools (search_flights / book_flight) — used by the canonical
# "book a flight, then correct the destination mid-turn" scenario. Fully
# mock/deterministic (no network) but wired through the exact same
# ToolEngine schema validation + fingerprint duplicate-protection as any
# other tool, and demonstrate `slot_updates` (search is read-only and does
# NOT touch session state; book_flight is state-changing and DOES).
# --------------------------------------------------------------------------


def _search_flights_handler(args: dict) -> dict:
    destination = args["destination"]
    return {
        "destination": destination,
        "flights": [
            {"flight_number": "AS101", "price_inr": 4200},
            {"flight_number": "AS202", "price_inr": 5100},
        ],
    }


def _book_flight_handler(args: dict) -> dict:
    destination = args["destination"]
    flight_number = args.get("flight_number", "AS101")
    return {
        "destination": destination,
        "flight_number": flight_number,
        "booking_id": f"BK-{destination[:3].upper()}-{flight_number}",
        "status": "confirmed",
    }


def _search_flights_slots(args: dict, result: Any) -> dict:
    return {"destination": args.get("destination")}


def _book_flight_slots(args: dict, result: Any) -> dict:
    slots = {"destination": args.get("destination")}
    if isinstance(result, dict) and result.get("booking_id"):
        slots["booking_id"] = result["booking_id"]
    return slots


def register_flight_demo_tools(runner: "ToolRunner") -> None:
    """Adds search_flights (read-only) + book_flight (state-changing) to an
    existing ToolRunner. Kept separate from create_builtin_runner() so the
    original two general-purpose tools stay the default, minimal surface;
    callers that want the flight-booking demo scenario opt in explicitly.
    """
    runner.register(
        ToolSpec(
            name="search_flights",
            description="Search available flights to a destination city.",
            parameters=(ParamSpec("destination", str),),
            state_changing=False,
            slot_updates=_search_flights_slots,
        ),
        _search_flights_handler,
    )
    runner.register(
        ToolSpec(
            name="book_flight",
            description="Book a flight to a destination. Requires a destination; flight_number is optional.",
            parameters=(
                ParamSpec("destination", str),
                ParamSpec("flight_number", str, required=False),
            ),
            state_changing=True,
            slot_updates=_book_flight_slots,
        ),
        _book_flight_handler,
    )


# --------------------------------------------------------------------------
# Troubleshooting demo tools (create_ticket / lookup_manual) — the other
# two named categories from spec §4's Mock Environment description
# ("...ticket creation, and frame-grounded manual lookups"), and the
# concrete tools behind §2's "Customer Support Bots" (create_ticket, a
# state-changing action needing the same duplicate-protection as booking)
# and "Field & Consumer Troubleshooting: Grounding device queries in
# camera frames and manuals" (lookup_manual, read-only, and able to accept
# a `frame_description` argument — the exact string grounding.py produces
# for a well-formed image event, e.g. "[image frame: 640x480px]" — so an
# LLM that saw one in the conversation can pass it straight through).
# --------------------------------------------------------------------------

_MANUAL_SNIPPETS: dict[str, str] = {
    "washer:e4": (
        "Error E4 indicates a water-drainage blockage. Power off, clean "
        "the drain filter (bottom-front access panel), and check the "
        "drain hose for kinks before restarting."
    ),
    "washer:e1": (
        "Error E1 indicates the door/lid is not detected as closed. "
        "Check the door latch and sensor for obstruction."
    ),
    "router:blinking_red": (
        "A steadily blinking red light indicates no WAN connection. "
        "Check the modem uplink cable and power-cycle both devices in "
        "modem-then-router order, waiting 30s between each."
    ),
}


def make_create_ticket_handler(id_source: Optional[Callable[[], int]] = None) -> Handler:
    """Factory for the create_ticket handler, matching
    make_current_time_handler()'s existing dependency-injection pattern.

    `id_source` defaults to a FRESH itertools.count(1) private to this one
    handler instance — never a module-level/global counter — so two
    different sessions (two different ToolRunners, each getting their own
    call to register_troubleshooting_demo_tools()) each start ticket
    numbering at 1 independently, matching StateManager/TraceRecorder's
    own no-cross-session-state rule, and keeping ticket IDs deterministic
    and reproducible within one session's run.
    """
    counter = id_source or itertools.count(1).__next__

    def handler(args: dict) -> dict:
        subject = args["subject"]
        priority = args.get("priority", "normal")
        return {
            "subject": subject,
            "priority": priority,
            "ticket_id": f"TCK-{counter():05d}",
            "status": "open",
        }

    return handler


def _create_ticket_slots(args: dict, result: Any) -> dict:
    slots = {"ticket_subject": args.get("subject")}
    if isinstance(result, dict) and result.get("ticket_id"):
        slots["ticket_id"] = result["ticket_id"]
    return slots


def _lookup_manual_handler(args: dict) -> dict:
    device = args["device"]
    query = args.get("query", "")
    frame_description = args.get("frame_description")
    key = f"{device.lower()}:{query.lower()}" if query else device.lower()
    snippet = _MANUAL_SNIPPETS.get(
        key,
        f"No manual section found for device={device!r} query={query!r}. "
        "Ask the user for the model number or a clearer photo of the "
        "display/error code.",
    )
    return {
        "device": device,
        "query": query,
        "frame_description": frame_description,
        "manual_snippet": snippet,
    }


def register_troubleshooting_demo_tools(runner: "ToolRunner") -> None:
    """Adds create_ticket (state-changing) + lookup_manual (read-only) to
    an existing ToolRunner. Kept separate from create_builtin_runner() for
    the same reason as register_flight_demo_tools(): opt-in, not a change
    to the minimal default tool surface.
    """
    runner.register(
        ToolSpec(
            name="create_ticket",
            description="Create a support ticket. Requires a subject; priority is optional (low/normal/high).",
            parameters=(
                ParamSpec("subject", str),
                ParamSpec("priority", str, required=False),
            ),
            state_changing=True,
            slot_updates=_create_ticket_slots,
        ),
        make_create_ticket_handler(),
    )
    runner.register(
        ToolSpec(
            name="lookup_manual",
            description=(
                "Look up troubleshooting steps for a device, optionally "
                "grounded in a description of a photographed error screen "
                "or indicator (frame_description)."
            ),
            parameters=(
                ParamSpec("device", str),
                ParamSpec("query", str, required=False),
                ParamSpec("frame_description", str, required=False),
            ),
            state_changing=False,
        ),
        _lookup_manual_handler,
    )


# --------------------------------------------------------------------------
# Registration
# --------------------------------------------------------------------------


def create_builtin_runner(
    engine: ToolEngine | None = None,
    now_fn: NowFn | None = None,
    include_flight_demo_tools: bool = False,
    include_troubleshooting_demo_tools: bool = False,
) -> ToolRunner:
    runner = ToolRunner(engine)
    runner.register(
        ToolSpec(
            name="calculator",
            description=(
                "Evaluate an arithmetic expression using + - * / // % ** and "
                "parentheses. Example expression: '25 * 37'."
            ),
            parameters=(ParamSpec("expression", str),),
            state_changing=False,
        ),
        _calculator_handler,
    )
    runner.register(
        ToolSpec(
            name="current_time",
            description=(
                "Get the current date and time. Optional timezone as an IANA "
                "name such as 'Asia/Kolkata'; defaults to the local timezone."
            ),
            parameters=(ParamSpec("timezone", str, required=False),),
            state_changing=False,
        ),
        make_current_time_handler(now_fn),
    )
    if include_flight_demo_tools:
        register_flight_demo_tools(runner)
    if include_troubleshooting_demo_tools:
        register_troubleshooting_demo_tools(runner)
    return runner
