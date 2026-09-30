# Ascendants — Samsung PRISM Theme 05: Interruptible Real-Time Agents

A voice-native assistant architecture that can be interrupted, corrected,
and re-planned mid-turn without losing session state or ever acting on a
stale result. Built around a real, local LLM — **qwen2.5:14b via
Ollama** — but every controller behavior (interruption, cancellation,
state tracking, tool duplicate-protection) is proven with fast,
deterministic, offline tests that don't need Ollama at all.

## The idea, in plain terms

Say "book a flight to Mumbai" to a normal chatbot, and while it's still
booking, say "actually, Delhi" — most bots will either ignore you, get
confused, or (worse) book both. This agent:

1. starts booking Mumbai,
2. hears the correction,
3. **cancels** the Mumbai booking (and the "old me" that was reasoning
   about it) — not just ignores it, actually stops it and marks it
   cancelled,
4. starts a fresh turn for Delhi,
5. and if the old Mumbai booking somehow finishes anyway in the
   background (a thread can't truly be killed), its result is checked
   against its own cancelled status and **thrown away** — it can never
   overwrite `destination` back to Mumbai.

Everything else you already told the agent (dates, passenger count,
whatever) survives the correction untouched. That's "Session Slot
Tracking" + "Interruption Recovery" — the two heaviest-weighted scoring
categories in the spec (75% combined) — and it's the thing this whole
codebase is built around.

## Architecture

```text
input event (text / audio / image / interruption / tool_manifest)
        │
        ▼
┌───────────────────┐   instant ack, no LLM call        FAST PATH
│  EventProcessor    │──────────────────────────────►  (fast_path.py)
│  (event_loop.py)   │
│  + StateManager    │   perception.normalize()          PERCEPTION
│  + InterruptHandler│──────────────────────────────►  (perception.py)
└─────────┬──────────┘   grounding.ground()              GROUNDING
          │ end_of_turn=True, well-formed              (grounding.py)
          ▼
┌────────────────────┐  background asyncio.Task,       COORDINATION
│    Orchestrator     │  registered with                    LAYER
│  (orchestrator.py)  │  InterruptHandler                (new)
└─────────┬───────────┘
          ▼
┌────────────────────┐  LLM call via asyncio.to_thread   SLOW PATH
│  AsyncLLMReasoner    │  (never blocks the event loop)
│ (async_reasoner.py) │  each tool call its OWN
└─────────┬────────────┘  cancellable call_id
          ▼
┌────────────────────┐  schema validation, fingerprint   TOOLS
│  ToolRunner/Engine   │  duplicate-protection,
│ (tools_builtin.py,   │  dynamic manifest registration
│  tool_engine.py,     │  (tool_manifest.py)
│  tool_manifest.py)   │
└─────────┬────────────┘
          ▼
   output actions: tool_call / cancel / clarification_request / final
   (final carries a StateSnapshot: intent + slots + timestamp)
```

- **Fast path** (`fast_path.py`): instant, non-LLM acknowledgement —
  runs the moment a chunk arrives, never waits on the model.
- **Slow path** (`async_reasoner.py` + `tools_builtin.py`): the LLM call
  and every tool call run as real, independently cancellable
  `asyncio.Task`s. The LLM's own HTTP call goes through
  `asyncio.to_thread`, so a slow qwen2.5:14b response never blocks the
  event loop from handling the NEXT interruption while it waits.
- **Coordination layer** (`orchestrator.py`): the piece that didn't exist
  before this project — it's what turns a finished user turn into a
  cancellable background task, emits `tool_call`/`clarification_request`/
  `final` actions live as they happen (not just at the very end), and
  guarantees a cancelled turn's result is dropped structurally rather
  than emitted-and-hopefully-ignored.
- **Session state** (`state_manager.py`): one `StateManager` per session,
  never a shared/global cache. An interruption never wipes it — it only
  cancels in-flight work; slots only change via explicit, localized
  updates (a correction overwrites `destination`, nothing else).
- **Dynamic tools** (`tool_manifest.py`): tools the agent has never seen
  before can be registered at runtime from a manifest — see the "unseen
  tool" scenario in the demo below.
- **Multimodal grounding** (`grounding.py`): a real WAV/PNG event is
  described (duration/sample-rate, width/height) and that description
  becomes part of the turn text sent to the LLM; if the metadata needed
  to describe it is missing, the agent asks for clarification instead of
  guessing.
- **Structured trace** (`trace.py`): every input/action/decision is
  logged with a timestamp, call_id, and reason — enough to replay a
  scenario after the fact. `EventProcessor.emit_action()` and
  `Orchestrator` both feed it automatically.
- **Fillers are real, traced actions** — not just log lines. A fast-path
  acknowledgement is emitted as `{"action": "filler", "call_id": ...,
  "text": ...}` and recorded to the trace the moment it's produced.
- **Turns have a wall-clock ceiling.** `Orchestrator(turn_timeout_seconds=...)`
  (default 110s, under the spec's 120s-per-scenario cap) bounds a
  reasoning turn so a hung model/tool call can't block a scenario forever;
  a timeout still yields a `final` action with the current state snapshot
  rather than silence.
- **Every emitted action is schema-validated** (`action_schema.py`) before
  it reaches the output queue — required fields, non-empty identifiers,
  and (for `final`) a well-formed state snapshot, plus a literal
  JSON-round-trip check. A malformed action is a bug caught at the source,
  not a silent Safety & Protocol deduction.
- **Time itself is injectable** (`clock.py`). Every timestamp in the
  codebase (`StateSnapshot.last_updated`, every trace entry, perception's
  fallback event timestamp) goes through a `Clock` — `RealClock` (the
  default, identical to `time.time()`) in production, or a `VirtualClock`
  shared across a session for fully deterministic, replay-reproducible
  timestamps (matching the spec's "Virtual Clock Streaming Harness").
- **All four Mock Environment tool categories exist** (`tools_builtin.py`,
  spec §4: "flight search, booking, ticket creation, and frame-grounded
  manual lookups"): `search_flights`/`book_flight`, `create_ticket`
  (state-changing, session-scoped deterministic sequential ticket IDs —
  never a shared global counter), and `lookup_manual` (read-only, accepts
  an optional `frame_description` argument — the exact string
  `grounding.py` produces for a well-formed image event — so a manual
  lookup can genuinely be "frame-grounded").
- **Deterministic, reusable fault injection** (`fault_injection.py`): wrap
  any tool handler with a `FaultPlan` — an explicit, ordered, 1-indexed
  schedule of injected latency and/or exceptions per call number, never
  randomized. `apply_fault_plan(runner, tool_name, plan)` wraps whatever
  handler is currently registered, in place, without disturbing its
  schema/duplicate-protection. The offline demo uses this (instead of the
  old one-off hand-rolled slow handler) to create a reproducible window
  during which the Mumbai booking can be interrupted mid-flight.

## Structure

```text
Ascendants-main/
├── agent/
│   ├── main.py            # wires everything into AgentRunner
│   ├── event_loop.py       # async controller: text/raw_input/interruption/manifest events
│   ├── state_manager.py    # session-scoped intent + slots (StateSnapshot)
│   ├── interrupt_handler.py# task cancellation registry
│   ├── orchestrator.py     # Coordination Layer: Controller <-> Reasoning
│   ├── async_reasoner.py   # native-async LLM loop, per-tool-call cancellation
│   ├── reasoner.py         # sync LLM loop (used by demo/ollama_smoke.py)
│   ├── llm_client.py       # Ollama HTTP client (default model: qwen2.5:14b)
│   ├── tools_builtin.py    # calculator/current_time/search_flights/book_flight + ToolRunner
│   ├── tool_engine.py      # schema validation + fingerprint duplicate-protection
│   ├── tool_manifest.py    # dynamic/unseen-tool registration from a manifest
│   ├── fault_injection.py  # reusable deterministic latency/fault injection wrapper
│   ├── action_schema.py    # Protocol Compliance: canonical schema per action type
│   ├── clock.py            # injectable time source (RealClock / VirtualClock)
│   ├── fast_path.py        # instant acknowledgements
│   ├── perception.py       # text/audio/image input validation+normalization
│   ├── grounding.py        # turns normalized input into LLM-usable text or "ask"
│   └── trace.py            # structured event/action trace log
├── tests/                  # 176 tests, all offline/deterministic (no Ollama needed)
│   └── test_scenarios.py   # the 12 canonical scenarios + race conditions A-F
├── demo/
│   ├── demo.py             # offline: fast path + tool engine only
│   └── ollama_smoke.py     # separate, REAL-Ollama smoke test (skips itself if unreachable)
└── ascendants/              # ARCHIVED pre-merge snapshot — see its own README, unused, safe to ignore
```

## Run

```bash
python3 -m pip install -r requirements.txt   # optional; see "Dependencies" below

python3 -m unittest discover -s tests -v     # 176 tests, no Ollama/network needed

python3 -m agent.main                        # full pipeline, offline, deterministic
                                              # (Mumbai -> interrupted -> Delhi -> weather via
                                              #  a dynamically-registered manifest tool)

python3 -m agent.main --llm                  # same pipeline against a REAL qwen2.5:14b:
                                              #   ollama serve
                                              #   ollama pull qwen2.5:14b
                                              #   python3 -m agent.main --llm

python3 demo/ollama_smoke.py                 # separate, minimal, REAL-Ollama-only smoke test
python3 demo/demo.py                         # original fast-path/tool-engine-only smoke demo
```

`python3 -m agent.main` (no `--llm`) runs the exact same controller,
orchestrator, and tool code that production uses — only the LLM itself is
a small scripted stand-in (`DeterministicDemoLLM` in `agent/main.py`), so
the demo is fully reproducible and needs no network. `--llm` swaps that
one piece for a real `OllamaClient`; nothing else changes.

## What "interruptible" actually means here

1. **A turn is a cancellable task.** `Orchestrator.start_turn()` wraps the
   reasoning call in a background `asyncio.Task` registered with
   `InterruptHandler`. The next `interruption` event cancels it —
   promptly, via `task.cancel()`, not by waiting for it to notice.
2. **A tool call is its OWN cancellable task**, with its own `call_id`,
   distinct from the turn's. `tool_call`/`cancel` actions are emitted with
   the correct id the instant each happens — not bundled into the final
   response.
3. **Stale results can't win.** `ToolEngine.complete_call()` refuses to
   resurrect an already-`CANCELLED` call — even if the underlying handler
   thread (which genuinely cannot be killed, only abandoned) finishes
   computing afterward. `AsyncLLMReasoner` also re-checks "is this turn
   still current?" immediately before applying any slot updates to
   `StateManager`, closing the exact race where a cancelled call's result
   could sneak into session state between "handler finished" and
   "caller noticed the cancellation."
4. **State-changing duplicate protection** is unconditional: `ToolEngine`
   fingerprints `(tool_name, args)` for state-changing tools and rejects a
   second identical call while the first is still pending — whether that
   tool came from the two built-ins, or from a manifest registered five
   seconds ago.
5. **Session isolation.** `StateManager`/`TraceRecorder` are per-session
   objects created fresh by `AgentRunner.__init__` — never a
   module-level global — so two sessions can never leak into each other
   (see `TestRaceConditionE_SessionIsolation` in
   `tests/test_scenarios.py`).

## Canonical scenarios (tests/test_scenarios.py)

All 12 scenarios from the spec + all 6 listed race conditions, run
end-to-end through the real `EventProcessor` → `Orchestrator` →
`AsyncLLMReasoner` → `ToolRunner` pipeline with a scripted (not mocked at
the controller level) LLM:

1. Normal text request
2. Interruption during reasoning (before any tool call)
3. Slot correction during an active task (Mumbai → Delhi)
4. State-changing tool duplicate prevention
5. Stale tool result after cancellation cannot modify state
6. Tool failure → retry → success
7. Missing required argument → clarification (not a guess)
8. Dynamic, never-hardcoded tool from a manifest, called end-to-end
9. Audio input: well-formed → grounded into the turn; insufficient → clarification
10. PNG/frame input: same as above
11. Chained tool calls (search → book → final)
12. Final response's state snapshot has the right intent/slots/timestamp

Plus race conditions A–F from the spec (interrupt-during-reasoning,
interrupt-during-tool-call, concurrent duplicate calls, cancel-then-retry,
cross-session isolation, and interruption racing a completing tool call).

## Dependencies

Only third-party dependency: Pydantic 2, used for `StateSnapshot`. If it
isn't installed (e.g. an offline eval sandbox), `state_manager.py` falls
back to a plain frozen `dataclass` with the same `.model_dump()` surface
— nothing else in the codebase needs to know or care which one is active.
Everything else is Python 3.10–3.12 standard library.

```bash
python3 -m pip install -r requirements.txt
```

## Known limitations

Documented honestly (not glossed over) in `AGENTS.md`'s status log —
short version: cancelling ONE tool call's `call_id` while its parent
reasoning turn keeps running isn't supported (cancelling a tool call
today cancels its containing turn too, since the call is `await`ed
synchronously within the turn's coroutine); the LLM's system prompt sees
a text *description* of audio/image content, not the raw
audio/pixels themselves (there's no ASR/vision model in this
environment — the grounding adapters are honest, deterministic mocks with
a documented seam for plugging a real one in later).

Also not yet implemented (identified against the spec, not yet built):
speculative execution ahead of `end_of_turn`; and mid-utterance
self-repair within a single still-arriving turn (today a correction is
only handled as a separate turn after an explicit `interruption` event,
not as a self-correction inside one utterance).
