# AGENTS.md — Ascendants (Samsung PRISM Hackathon, Theme 05: Interruptible Real-Time Agents)

Read this file first before touching anything in this repo.

## Team split

- **Shivansh — Core Controller** (`agent/event_loop.py`, `agent/state_manager.py`,
  `agent/interrupt_handler.py`, `agent/main.py`): receives input events, keeps
  session state, handles interruptions, cancels obsolete tasks, wires everything
  together. Not built yet as of this commit.
- **Dhruv — Capabilities** (`agent/fast_path.py`, `agent/tool_engine.py`,
  `agent/perception.py`): fast acknowledgements, dynamic tool execution,
  multimodal input normalization. These are consumed by Shivansh's controller
  but do not depend on it.

## Directory layout

```
ascendants/
  AGENTS.md
  README.md
  requirements.txt
  agent/
    __init__.py
    fast_path.py        # Dhruv
    tool_engine.py       # Dhruv
    perception.py        # Dhruv
    event_loop.py         # Shivansh (not yet created)
    state_manager.py      # Shivansh (not yet created)
    interrupt_handler.py  # Shivansh (not yet created)
    main.py                # Shivansh (not yet created)
  tests/
    test_fast_path.py
    test_tool_engine.py
    test_perception.py
  demo/
    demo.py
```

## Design contract between Dhruv's modules and the future controller

These are the shapes the controller is expected to call. Keeping these stable
means Shivansh's code and Dhruv's code can be merged without rewrites.

- `fast_path.py` exposes `FastPath.acknowledge(user_text: str) -> str`. It is a
  pure function of the input (plus minimal internal state to avoid repeating
  the same filler twice in a row). It never claims task completion.
- `tool_engine.py` exposes a `ToolEngine` class with:
  - `register_tool(spec: ToolSpec)`
  - `create_call(tool_name: str, args: dict) -> ToolCall` (assigns a unique
    `call_id`, validates args against the tool's schema)
  - `is_state_changing(tool_name: str) -> bool`
  - `complete_call(call_id, result)` / `cancel_call(call_id)`
  - Guards against a second state-changing call with the same `(tool_name, args)`
    fingerprint being created while an earlier one is still pending.
- `perception.py` exposes `normalize(input_event) -> PerceptionResult`, which
  accepts text, WAV audio bytes/metadata, or PNG/image frame bytes/metadata,
  and returns a common structure (`modality`, `payload`, `metadata`,
  `timestamp`) that the rest of the agent can consume regardless of input type.
  Malformed input raises a typed `PerceptionError` instead of crashing.

## Status as of this commit

Only `README.md` existed before this session. This session adds the full
skeleton above plus a working first version of Dhruv's three modules, tests,
and a small demo script showing an interruption (Delhi -> Mumbai) being
represented correctly (old call obsoleted, new call created, stale result
ignored). Shivansh's controller files do not exist yet — nothing here should
block him from adding them.

## Status update — gap review against Theme_5_Guide.pdf (PS)

By the time this update was written, both halves existed (controller +
reasoning), but they were disconnected, and the controller side had zero
test coverage. Fixed in this pass:

1. **Coordination Layer was missing entirely** (spec §1: "Coordination
   Layer: handles non-blocking execution, call cancellation, state snapshot
   updates, and idempotency"). `agent/main.py`'s demo drove only fast path +
   interrupt handling with canned events; `demo/ollama_smoke.py` drove only
   the LLM reasoner, synchronously, with no cancellation at all. Added
   `agent/orchestrator.py`: turns a completed turn into a cancellable
   `asyncio.Task` registered with `InterruptHandler`, runs the (synchronous)
   `LLMReasoner` via `asyncio.to_thread` so it never blocks the event loop,
   and structurally drops any result superseded by a later interruption
   instead of emitting-then-tagging-stale. Wired into `AgentRunner` in
   `main.py` (`use_llm=True` by default; qwen2.5:14b via Ollama).
2. **`state_manager.reset()` on every interruption wiped ALL session
   slots** (spec §3.2.3: "Session Slot Tracking... apply localized slot
   corrections"; §2 use case: "Dynamically adjusting parameters mid-booking
   without double-booking"). A correction like "actually, Mumbai" would have
   silently forgotten `origin`/`travel_date` too. `event_loop.py`'s
   `_handle_interruption` no longer resets state — it only cancels in-flight
   tasks, clears the pending turn buffer, and timestamps the interruption.
3. **`perception.py` was never called by the controller.** It was fully
   unit-tested in isolation but nothing in `event_loop.py` invoked
   `normalize()`. Added a `raw_input` event type + `_handle_raw_input` that
   normalizes and, on `PerceptionError`, emits a `clarification_request`
   action (an output type the spec requires — §3.1) instead of crashing or
   silently dropping malformed audio/image input.
4. **No output ever carried a State Snapshot.** The spec requires "final
   responses carrying structured State Snapshots (intent and slot values)"
   — `orchestrator.py` now attaches `state_manager.get_snapshot().model_dump()`
   to every `final` action.
5. **Zero tests for the controller.** Added `tests/test_state_manager.py`,
   `tests/test_interrupt_handler.py`, `tests/test_event_loop.py`,
   `tests/test_orchestrator.py` (107 tests total now, up from 75). Writing
   the orchestrator test caught a real bug during this pass: `_run_turn` was
   unregistering a task from `InterruptHandler` *before* checking whether it
   was still current, which made the "still current" check always false and
   silently dropped every legitimate result. Fixed and regression-tested.
6. **Hard dependency on pydantic could crash the whole controller on
   import** in a sandbox without network/pydantic installed (this happened
   while testing this change). `state_manager.py` now falls back to a plain
   frozen dataclass with the same `.model_dump()` surface if the import
   fails.

Not fixed / still open, flagged for whoever picks this up next:

- `tools_builtin.ToolRunner.run()` is synchronous and blocks — tool
  execution inside `LLMReasoner.respond()` runs on the same worker thread as
  the model call, so a slow tool delays the whole turn rather than running
  as an independently-cancellable non-blocking call with its own `call_id`
  surfaced to the controller mid-turn (right now only the *turn* is a
  cancellable unit, not each tool call within it). Getting per-tool-call
  cancellation would need `ToolRunner` to grow an async variant that emits
  a `tool_call` action immediately and resolves later via the existing
  `tool_result` event path in `event_loop.py` (`pending_tool_calls`) — the
  machinery for that already exists, it's just unused by the reasoner.
- No scenario-tool-manifest parsing (spec §3.1 lists "scenario tool
  manifests" as an input) — tools are still hardcoded in
  `tools_builtin.create_builtin_runner()`; nothing dynamically registers
  tools from a manifest at session start.
- No frame/audio grounding into the LLM prompt itself — `perception.py`
  validates and normalizes multimodal input, but `reasoner.py`'s system
  prompt only ever sees text; a WAV/PNG payload updates a state slot
  (`last_audio_wav` / `last_image_png` metadata) rather than being
  described to the model.
- The nested `ascendants/` subfolder is a stale, byte-identical snapshot of
  the pre-merge skeleton (verified via `diff`). Left in place per the "don't
  delete existing files" rule below — flagging it here so a human can
  decide whether to remove it.

## Status update 2 — submission-ready pass (this session)

Every "not fixed / still open" item from the previous update except one
is now implemented and tested. In order:

1. **Dynamic scenario tool manifests** (`agent/tool_manifest.py`). A
   manifest is a list of dicts (`name`, `description`, `parameters`,
   `state_changing`); `register_manifest_tools()` parses it into
   `ToolSpec`s and registers them on a live `ToolRunner` — a tool the
   agent has never seen before becomes callable on the very next turn.
   Any entry with no supplied handler gets a deterministic mock handler
   (`make_mock_handler`) so "unseen tool" scenarios are safely testable
   without a real backend. State-changing manifest tools get the exact
   same `ToolEngine` fingerprint duplicate-protection as the two
   built-ins — nothing about validation/dedup is manifest-specific.
   Wired into the controller via a new `tool_manifest` event type in
   `event_loop.py` and `Orchestrator.register_manifest()`.

2. **Per-tool-call cancellation**
   (`ToolRunner.run_cancellable` in `tools_builtin.py`, used by the new
   `agent/async_reasoner.py`). Previously only the whole reasoning turn
   was cancellable — a tool call was an opaque blocking step inside it.
   Now every tool call gets its own `call_id`, its own `asyncio.Task`
   registered with `InterruptHandler`, and its own lifecycle status in
   `ToolEngine`. `tool_call` and `cancel` actions are emitted with the
   correct `call_id` live, not bundled into the final response. Stale-
   result protection is structural: `ToolEngine.complete_call()` refuses
   to resurrect an already-`CANCELLED` call, and `AsyncLLMReasoner`
   re-checks currency immediately before applying any `slot_updates` to
   `StateManager`.
   **Known, disclosed limitation**: because a tool call is `await`ed
   synchronously within its turn's coroutine, cancelling ONE tool call's
   `call_id` while its parent turn keeps reasoning isn't supported today —
   cancelling a tool call cancels its containing turn too. Full
   independence would need tool calls to run as fire-and-forget work the
   reasoner polls rather than awaits — flagged as real future work, not
   silently left broken.

3. **Session slot tracking**, verified end-to-end (not just at the
   `StateManager` unit level): `tests/test_scenarios.py`'s
   `TestScenario3SlotCorrection` runs the full "book Mumbai → interrupt →
   actually Delhi" flow through the real pipeline and asserts
   `state.get_slot("destination") == "Delhi"` with no Mumbai artifact
   anywhere in the final response or snapshot.

4. **Multimodal grounding actually wired in.** The previous update built
   `perception.py`'s normalization but never connected it to anything
   past validation. This session added `agent/grounding.py` (adapters
   that describe — never fabricate — audio/image metadata as bounded
   text, with a documented seam for a real ASR/vision model later) AND
   fixed `event_loop._handle_raw_input` to actually call it: a
   well-formed WAV/PNG event now becomes grounded text that flows into
   the same turn-handling path as typed text; an event that passes
   `perception.normalize()` but lacks the metadata grounding needs
   (no sample rate, no width/height) gets a `clarification_request`
   instead of a guess.

5. **Clarification handling as a first-class output.** Was previously
   only wired for malformed perception input. Now also fires when
   `AsyncLLMReasoner`'s tool call is rejected specifically for a missing
   required argument (`ToolOutcome.needs_clarification`, distinguished
   from other rejection reasons like "duplicate pending call" which
   should NOT prompt the user).

6. **Structured trace logging** (`agent/trace.py`). Session-scoped,
   append-only, JSONL-exportable log of every input/action/decision with
   timestamp/call_id/status/reason — wired through `Orchestrator`
   (optional `trace=` parameter). Used in the offline demo's final
   section to show a full replay of a scenario.

7. **Fast/slow path architecture, verified under load**: the LLM's own
   HTTP call runs via `asyncio.to_thread` inside `AsyncLLMReasoner`, so
   the event loop can keep processing an `interruption` event while a
   slow qwen2.5:14b call is in flight —
   `TestScenario2InterruptionDuringReasoning` proves this by blocking the
   scripted LLM on a `threading.Event` and confirming the interruption
   still lands and the eventual (late) LLM result is dropped.

8. **12 canonical scenarios + race conditions A–F**
   (`tests/test_scenarios.py`, 15 tests total, all deterministic via
   `threading.Event`-controlled fake handlers — no real sleeps used for
   correctness, only for demo narration in `agent/main.py`). Covers
   normal requests, interruption timing, slot correction, duplicate
   prevention, stale-result rejection, tool failure/retry, missing-
   argument clarification, dynamic manifest tools, both multimodal
   grounding paths, chained tool calls, snapshot shape, and — for the
   race conditions specifically — cross-session isolation (E) and
   interruption racing a completing tool call (F).

9. **`agent/main.py` rewritten** so the offline demo (`python3 -m
   agent.main`, no `--llm`) exercises the REAL orchestrator/controller/
   tool-engine code end-to-end via `DeterministicDemoLLM` — a scripted
   stand-in for ONLY the model, not a parallel mock pipeline. Ran and
   inspected this personally this session: manifest registration →
   Mumbai booking starts → interrupted mid-flight (both the turn's and
   the tool call's `call_id` get cancelled) → "Actually, Delhi" → final
   response and state snapshot correctly show Delhi only → a dynamically
   registered `current_weather` tool gets called successfully. `--llm`
   swaps in a real `OllamaClient` (still defaulting to `qwen2.5:14b`)
   with nothing else changing.

10. **`ascendants/` duplicate resolved safely without deleting anything**:
    added `ascendants/ARCHIVED_README.md` explaining it's an inert,
    never-imported, never-discovered-by-`unittest discover` pre-merge
    snapshot, so nobody mistakes it for the active codebase again.

Test count: 176 (up from 107 last update). Every test in the suite is
offline/deterministic; `demo/ollama_smoke.py` remains the one piece that
talks to a real server, and it self-checks reachability and exits cleanly
(non-zero, clear message) if Ollama isn't running rather than hanging or
crashing — confirmed by running it in this sandbox (no Ollama available
here), where it printed "Ollama is not reachable..." and exited 1 as
designed.

**Genuinely still open** (disclosed, not silently skipped): true
independent-of-parent-turn cancellation for a single tool call (item 2
above); the LLM only ever sees a text *description* of audio/image
content, never real transcription/vision (no such model is available in
this environment — the adapter seam in `grounding.py` is where a real one
would plug in).

## Status update 3 — post-submission gap review against Theme_5_Guide.pdf

Re-reading the PDF against the shipped zip surfaced four more gaps. This
update fixes the first two (the other two — action-schema validation and
a virtual/injectable clock — are tracked as still open, see below).

1. **Fast-path acknowledgements never reached the output queue or trace.**
   `_FastPathAdapter.handle_text_chunk` only called `logger.info()` —
   spec §3.1 lists "spoken fillers" as a first-class output type, and
   Response Latency (15% of scoring) is "scored strictly on trace logs"
   per §5. A filler that only exists as a log line is, to a trace-based
   evaluator, indistinguishable from no filler at all. Fixed:
   `FastPathProtocol.handle_text_chunk` now returns the filler text (or
   `None` to stay silent), and `EventProcessor._handle_text_chunk` emits
   it as a real `{"action": "filler", "call_id": ..., "text": ...}`
   action via `emit_action()`. `emit_action()` also now records every
   action it emits to an optional `TraceRecorder` automatically (passed
   into `EventProcessor` the same optional way as `turn_handler`), so
   cancellations and clarification_requests it emits directly are traced
   too, not just Orchestrator-originated actions.

2. **No per-turn wall-clock timeout.** Spec §6: "120s wall-clock cap per
   scenario." Nothing bounded how long a reasoning turn (LLM call plus
   however many chained tool calls) could run — a hung Ollama connection
   would have blocked a scenario forever from our side. Fixed:
   `Orchestrator` takes a `turn_timeout_seconds` parameter (default
   `110.0`, a 10s margin under the spec's 120s cap) and wraps
   `reasoner.respond()` in `asyncio.wait_for()`. On timeout: the turn is
   unregistered (not leaked), the decision is traced with
   `status="timeout"`, and a `final` action is still emitted — with
   `error: "timeout"` and the CURRENT state snapshot (not stale/empty) —
   so the user gets a concrete answer instead of silence.

   **Building this surfaced a real, separate concurrency bug**, caught by
   the test suite immediately: fixing a genuine leak in
   `ToolRunner.run_cancellable` (it wasn't unregistering its own call_id
   from `InterruptHandler` on cancellation — harmless before because
   `cancel_all_tasks()` always cleared everything anyway, but would leak
   forever under a *targeted*, non-full cancellation like this new
   timeout path) exposed that `InterruptHandler.cancel_all_tasks()` was
   iterating a *live* `dict.items()` view while one of the tasks it was
   awaiting could — and, once the leak fix was in, did — mutate that same
   dict mid-iteration (a cancelled `run_cancellable` unregistering its own
   call_id from inside its own `except CancelledError` handler). This
   raised `RuntimeError: dictionary changed size during iteration` and
   would have silently aborted cancellation of every task queued after
   the mutating one. Fixed by snapshotting to a plain list before any
   `await` in `cancel_all_tasks()`. Three tests in the existing suite
   (`test_async_reasoner`, `test_run_cancellable`, `test_scenarios`)
   failed immediately after the leak fix and before the snapshot fix —
   confirming this was a real, reachable bug, not a hypothetical one — and
   a dedicated regression test
   (`test_survives_a_task_unregistering_itself_mid_cancellation`) was
   added to `tests/test_interrupt_handler.py`.

Test count: 184 (up from 176). Demo re-verified end-to-end after both
changes (`python3 -m agent.main`) — fillers now appear as real actions in
the printed output and the trace log.

**Still open** (not yet done — next planned steps): (3) formal
action-schema validation ("Protocol Compliance" — every emitted action
dict should be checked against one canonical schema per action type, and
tested for `json.dumps()`-safety) and (4) an injectable/virtual clock (so
`StateSnapshot.last_updated`/trace timestamps and our own tests can run
against a controllable clock instead of real `time.time()`/sleeps, matching
the spec's "Virtual Clock Streaming Harness" description of the hidden
evaluation environment).

## Status update 4 — items 3 & 4 (Protocol Compliance, Virtual Clock)

1. **Formal action-schema validation** (`agent/action_schema.py`,
   spec §3.2.6: "Protocol Compliance: Emit well-formed JSON payloads with
   valid snapshots and identifiers"). Every action type (`filler`,
   `tool_call`, `cancel`, `clarification_request`, `final`) now has a
   canonical required-fields schema, checked via `validate_action()`:
   non-empty string `call_id`s where required, a `final` action's
   `state_snapshot` having `intent`/`slots`/`last_updated` in the right
   shapes, and — taken literally — that the WHOLE action dict actually
   serializes to JSON and round-trips through `json.loads(json.dumps(...))`
   unchanged (catches things like `NaN`/raw bytes that `json.dumps` alone
   wouldn't reject). Wired into `EventProcessor.emit_action()` (raises
   loudly — a violation here is always our own bug, caught by `run()`'s
   per-event try/except so one bad action never crashes the loop) and
   into a new `Orchestrator._emit()` helper used by every one of
   Orchestrator's action-emission points (tool_call, cancel-via-timeout,
   clarification_request, final) — there `_emit()` logs and DROPS a
   malformed action rather than raising, since Orchestrator's own task
   isn't wrapped in the same per-event try/except EventProcessor.run()
   has. Every real call site in the codebase already conformed — the
   26 new schema tests plus a new integration test
   (`TestProtocolCompliance` in `test_scenarios.py`, validating every
   action from a real multi-turn run) all pass without needing to touch
   any actual action-construction code.

2. **Injectable clock** (`agent/clock.py`, spec §4: "Virtual Clock
   Streaming Harness: Deterministic event replay..."). `Clock` is a
   two-method protocol (`now()`); `RealClock` (the default everywhere)
   is behaviorally identical to calling `time.time()` directly — pure
   refactor, zero behavior change for real usage. `VirtualClock` starts
   at a fixed instant and only advances via explicit `.tick()`/`.set()`
   calls, so every component sharing ONE instance agrees on "now" with
   zero wall-clock jitter. Threaded through every place that used to call
   `time.time()` directly: `StateManager` (drives `last_updated`),
   `TraceRecorder` (drives every entry's `timestamp`), `perception.
   normalize()` (as the FALLBACK only — an event's own explicit
   `"timestamp"` always still wins, matching how a real replay harness
   would supply its own), and `EventProcessor`/`AgentRunner` (accepts a
   `clock=` param and passes the SAME instance to `StateManager`,
   `TraceRecorder`, and `EventProcessor` so a whole session shares one
   clock). Proven end-to-end in `tests/test_scenarios.py`
   (`TestVirtualClockDrivenRun`): a full multi-component run driven by one
   `VirtualClock` produces exactly reproducible trace timestamps, and two
   separate runs seeded identically produce byte-identical timestamp
   lists — the actual point of a virtual clock (replay determinism).

Test count: 231 (up from 184). Demo re-verified end-to-end again
(`python3 -m agent.main`) — output and trace unchanged in shape, still
uses `RealClock` by default (no behavior change for real usage).

## Status update 5 — items 5 & 6 (mock-environment tools, fault injection)

5. **The two missing Mock Environment tool categories** (spec §4: "flight
   search, booking, ticket creation, and frame-grounded manual lookups" —
   only the first two existed). Added in `agent/tools_builtin.py`, opt-in
   via `create_builtin_runner(include_troubleshooting_demo_tools=True)`
   (default False, so the minimal default tool surface and every existing
   test assumption about it are untouched):
   - `create_ticket` — state-changing, so it gets the same fingerprint
     duplicate-protection as `book_flight`; `slot_updates` records
     `ticket_subject` and `ticket_id`. Ticket IDs come from a
     `make_create_ticket_handler()` factory holding a PRIVATE
     `itertools.count(1)` per handler instance — deliberately NOT a
     module-level counter. My first draft used a module-level global and I
     replaced it before shipping: a shared global would have made ticket
     IDs order-dependent across sessions/tests in the same process (breaking
     both the spec's "no cross-session state" rule and the "deterministic"
     requirement), the same class of problem `StateManager`/`TraceRecorder`
     are already built to avoid. Regression test:
     `test_two_separate_runners_each_start_ticket_numbering_at_one`.
   - `lookup_manual` — read-only (repeatable, never deduplicated); takes
     an optional `frame_description` argument, which is exactly the string
     `grounding.py` produces for a well-formed image event
     (`"[image frame: 640x480px]"`), so an LLM that saw one in the
     conversation can pass it straight through — that is what makes the
     lookup genuinely "frame-grounded". Unknown device/query returns a
     helpful fallback asking for a model number or clearer photo, never a
     crash or a fabricated manual section.

6. **Reusable deterministic fault injection** (`agent/fault_injection.py`,
   spec §4: "Deterministic latency and fault injection..."). Previously
   the only latency injection in the repo was one hand-rolled
   `time.sleep()` inside a demo-only `book_flight` handler in `main.py` —
   not reusable, not applicable to any other tool, not testable without a
   real wall-clock wait. Now: `FaultStep` (latency and/or an exception for
   one call), `FaultPlan` (an explicit, ordered, 1-indexed schedule of
   steps; calls past the end of the plan run normally; convenience
   constructors `fail_then_succeed` and `latency_then_normal`),
   `wrap_with_faults(handler, plan, sleep_fn=time.sleep)` (each wrap gets
   its own private call counter), and `apply_fault_plan(runner, tool_name,
   plan)` (wraps whatever handler is currently registered, in place, via
   the new public `ToolRunner.get_handler()` accessor — no reaching into
   private state — leaving the tool's schema, `state_changing` flag,
   `slot_updates`, and duplicate-protection untouched; there's a test for
   each of those). `sleep_fn` is injectable, mirroring the `Clock`
   pattern from item 4, so tests verify "latency of 0.3s was requested"
   instantly and deterministically without ever actually sleeping.
   `agent/main.py` now uses this in place of the old one-off slow
   handler, which was deleted.

**Demo** (`python3 -m agent.main`) now has a step 6 — "My washer shows
error E4" — that chains `lookup_manual` → `create_ticket` → final, so a
single run exercises all four mock-environment categories plus chained
tool calls, dynamic manifest tools, interruption/cancellation, and
fault-injected latency. Verified the action counts are exact (7+3+4=14,
nothing truncated or left over in the queue).

Test count: 266 (up from 231): 18 in `test_troubleshooting_tools.py`, 17 in
`test_fault_injection.py`.

**Still open**: (7) speculative execution ahead of `end_of_turn`, and (8)
mid-utterance self-repair within a single still-arriving turn.

## Rules for future Claude sessions working on this repo

- Do not delete or blindly overwrite existing files.
- Do not build Shivansh's controller files unless he asks — only stub/document
  the interface they need to satisfy.
- Run `git status` and `git diff` before and after changes, and report them.
- Do not push to a remote unless explicitly asked.
