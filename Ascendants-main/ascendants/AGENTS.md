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

## Rules for future Claude sessions working on this repo

- Do not delete or blindly overwrite existing files.
- Do not build Shivansh's controller files unless he asks — only stub/document
  the interface they need to satisfy.
- Run `git status` and `git diff` before and after changes, and report them.
- Do not push to a remote unless explicitly asked.
