# Ascendants — Interruptible Real-Time Agents

### Samsung PRISM Generative AI Hackathon — 3rd Edition 2026–27
**Theme 05: Interruptible Real-Time Agents**

**Team:** Ascendants  
**College:** Thapar Institute of Engineering & Technology

Ascendants is an interruptible real-time agent architecture designed to remain responsive while reasoning and executing tools asynchronously. When a user changes their intent, the system cancels obsolete work, updates only the affected session state, and prevents stale results or duplicate state-changing actions from taking effect.

---

## 1. Problem

Conventional assistants generally follow a sequential listen → think → respond flow. This becomes unreliable when a user interrupts or changes their mind while a task is already running.

For example:

> “Book a flight to Mumbai.”  
> *While the booking is running:* “Actually, Delhi.”

A robust agent should not continue acting on the obsolete Mumbai request. It should:

1. acknowledge the user quickly,
2. cancel the obsolete work,
3. update the affected session slot,
4. start the new task,
5. ignore any stale result from the cancelled task, and
6. return a final response based on the current state.

Ascendants is built around this interruption-and-recovery problem.

---

## 2. Solution

Ascendants uses a **Fast Path + Slow Path + Coordination Layer** architecture.

### Fast Path
Provides immediate, non-blocking acknowledgement without waiting for an LLM response.

### Slow Path
Runs reasoning and tool execution asynchronously so long-running operations do not block the event loop.

### Coordination Layer
Manages turns, cancellation, tool calls, state updates and final actions.

### Session State
Maintains intent and slots for the current session. An interruption changes only the relevant information instead of clearing the entire session.

### Tool Protection
State-changing tool calls use duplicate protection and cancellation-aware completion.

### Structured Trace
Inputs, actions and decisions are recorded with timestamps, call IDs and reasons for reproducibility and debugging.

---

## 3. Architecture

```text
Input Events
(text / audio / image / interruption / tool manifest)
                         │
                         ▼
              ┌─────────────────────┐
              │    EventProcessor   │
              │     event_loop.py   │
              └──────────┬──────────┘
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
     Fast Path       Perception      Grounding
   fast_path.py    perception.py   grounding.py
          │              │              │
          └──────────────┼──────────────┘
                         ▼
              ┌─────────────────────┐
              │    Orchestrator     │
              │   Coordination      │
              │      Layer          │
              └──────────┬──────────┘
                         ▼
              ┌─────────────────────┐
              │  AsyncLLMReasoner   │
              │  async_reasoner.py  │
              └──────────┬──────────┘
                         ▼
              ┌─────────────────────┐
              │    Tool Engine      │
              │ tool_engine.py      │
              │ tools_builtin.py    │
              │ tool_manifest.py    │
              └──────────┬──────────┘
                         ▼
             Tool / Cancel / Clarify / Final
                         │
                         ▼
                Session State + Trace
```

---

## 4. Key Capabilities

### Interrupt Recovery
A running turn is represented as a cancellable asynchronous task. An interruption can cancel the obsolete turn and start a new one.

### Stale Result Protection
A cancelled tool call cannot later resurrect itself and overwrite the current session state if its underlying work finishes after cancellation.

### Session Slot Tracking
Session state is maintained per agent session. Local corrections update the relevant slot while preserving other information.

### State-Changing Duplicate Protection
State-changing calls are fingerprinted using the tool name and arguments so identical pending actions are not executed twice.

### Dynamic Tool Registration
Tools can be registered at runtime from a tool manifest, allowing the agent to handle previously unseen tools.

### Multimodal Event Grounding
Text, audio and image events are normalized and converted into grounded context. When required information is unavailable, the system can request clarification rather than inventing an answer.

### Structured Actions
Actions are validated against the project's action schema before being emitted.

### Deterministic Fault Injection
The project includes reusable fault injection for controlled latency and exception scenarios, making interruption and recovery behavior reproducible.

### Virtual Clock Support
The time source can be replaced with a virtual clock for deterministic, replay-friendly timestamps during testing.

---

## 5. Demo Scenario

The primary interruption scenario demonstrates:

```text
User: "Book a flight to Mumbai"
                │
                ▼
        Fast acknowledgement
                │
                ▼
        Mumbai task starts
                │
                ▼
User: "Actually, Delhi"
                │
                ▼
        Mumbai task cancelled
                │
                ▼
        Destination → Delhi
                │
                ▼
        Delhi task starts
                │
                ▼
 Late Mumbai result arrives
                │
                ▼
        STALE RESULT REJECTED
                │
                ▼
        Delhi result used
```

The repository also contains scenarios for chained tool calls, retries, clarification, dynamic tools, audio/image grounding, duplicate protection and race conditions.

---

## 6. Extension Use Case

Ascendants includes troubleshooting-oriented tools in addition to the core interruption flow.

The repository contains support for:

- ticket creation,
- manual lookup,
- frame-grounded manual lookup,
- deterministic troubleshooting behavior,
- state-changing and read-only tool classification.

This provides a path from the core real-time agent architecture to a practical troubleshooting assistant where a user can change or refine their request while an operation is already in progress.

---

## 7. Technology Stack

| Technology | Purpose |
|---|---|
| Python 3.10–3.12 | Core implementation |
| `asyncio` | Asynchronous event processing and cancellation |
| Ollama | Local LLM integration |
| Qwen 2.5 14B | Configured local LLM |
| Pydantic 2 | State/schema validation |
| Python `unittest` | Deterministic automated testing |

The project can run its controller and test scenarios without requiring Ollama.

---

## 8. Repository Structure

```text
Ascendants-main/
├── agent/
│   ├── main.py
│   ├── event_loop.py
│   ├── orchestrator.py
│   ├── interrupt_handler.py
│   ├── async_reasoner.py
│   ├── reasoner.py
│   ├── llm_client.py
│   ├── state_manager.py
│   ├── action_schema.py
│   ├── tools_builtin.py
│   ├── tool_engine.py
│   ├── tool_manifest.py
│   ├── fault_injection.py
│   ├── clock.py
│   ├── fast_path.py
│   ├── perception.py
│   ├── grounding.py
│   └── trace.py
│
├── tests/
├── demo/
├── requirements.txt
├── README.md
├── AGENTS.md
└── .gitignore
```

An older pre-merge implementation is retained under `ascendants/` for reference and is not the active implementation.

---

## 9. Installation

### Requirements

- Python 3.10–3.12
- Optional: Ollama for the real local LLM path
- Qwen 2.5 14B for the configured Ollama path

Create a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

---

## 10. Run the Project

### Deterministic offline run

The default run uses a deterministic demo LLM and does not require network access or Ollama:

```bash
python3 -m agent.main
```

This exercises the controller, orchestration, interruption handling and tool pipeline while keeping the LLM behavior reproducible.

### Real Ollama path

Install and start Ollama, then make sure the configured model is available:

```bash
ollama serve
ollama pull qwen2.5:14b
```

Run:

```bash
python3 -m agent.main --llm
```

### Ollama smoke test

```bash
python3 demo/ollama_smoke.py
```

### Original fast-path/tool-engine demo

```bash
python3 demo/demo.py
```

---

## 11. Run Tests

The project contains a deterministic offline test suite.

Run:

```bash
python3 -m unittest discover -s tests -v
```

The current project test suite runs **266 tests** and is designed to run without Ollama.

The test suite covers interruption handling, tool cancellation, state management, duplicate protection, race conditions, dynamic tools, perception/grounding, tracing, fault injection and scenario-level behavior.

---

## 12. Reproducibility

For deterministic controller validation:

```bash
python3 -m unittest discover -s tests -v
```

The project also includes:

- a virtual clock,
- deterministic demo LLM behavior,
- deterministic fault injection,
- structured traces,
- reproducible scenario tests.

These components make interruption and race-condition behavior easier to reproduce and inspect.

---

## 13. Demo Video

**Demo video:** `ADD_YOUTUBE_OR_GOOGLE_DRIVE_LINK_HERE`

The demo should show:

1. the normal task starting,
2. the fast acknowledgement,
3. an interruption/correction,
4. cancellation of obsolete work,
5. updated session state,
6. the new task starting,
7. a stale result being rejected,
8. the extension/troubleshooting use case.

---

## 14. Presentation

**Presentation:** `Thapar_Ascendants_Theme05_Submission.pptx`

The presentation covers the Theme 05 problem, solution architecture, demo flow, technology stack, impact/differentiation, results, limitations and next steps.

---

## 15. AI Disclosure

The completed AI Usage Disclosure is included with the submission.

The team used **Claude, Gemini, OpenCode, ChatGPT and Antigravity** as development assistants across ideation, architecture exploration, coding assistance, debugging, testing, documentation and submission preparation.

AI-generated suggestions were reviewed, adapted and integrated by the team.

---

## 16. Limitations

The current implementation has explicit limitations:

- The repository does not contain a complete end-to-end speech I/O stack.
- Audio and image inputs are normalized/grounded into descriptions rather than being processed directly by a speech-recognition or vision model in this environment.
- Cancelling an individual tool call while allowing its parent reasoning turn to continue is not currently supported; the containing turn is cancelled.
- Speculative execution before `end_of_turn` is not implemented.
- Mid-utterance self-repair within one still-arriving utterance is not implemented; correction is handled through an explicit interruption event.

These limitations are documented so the implemented scope is clear.

---

## 17. Future Work

The architecture provides clear extension points for:

- full real-time speech input/output,
- direct audio and vision model integration,
- speculative execution,
- mid-utterance self-correction,
- further benchmark integration,
- expanded real-world troubleshooting workflows.

---

## 18. Submission Files

The final submission package contains:

- Source code
- `requirements.txt`
- Presentation
- AI Disclosure
- README
- Demo video link

---

## 19. Team

**Team:** Ascendants  
**College:** Thapar Institute of Engineering & Technology  
**Theme:** 05 — Interruptible Real-Time Agents

---

## 20. License

Add the project's applicable license here if required by the team or hackathon submission rules.
