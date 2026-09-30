"""Quick real-model check: Ollama + reasoner + calculator/current_time tools.

    python demo/ollama_smoke.py                      # interactive chat
    python demo/ollama_smoke.py "what is 25 * 37?"   # one question, then exit

Shows every tool call so you can see the model really used the tool.
This does NOT use the event loop / interrupt handler yet; it only proves that
the LLM + tools + conversation memory part works.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.llm_client import OllamaClient
from agent.reasoner import ConversationHistory, LLMReasoner
from agent.tools_builtin import create_builtin_runner


def ask(reasoner: LLMReasoner, history: ConversationHistory, text: str) -> None:
    history.add("user", text)
    result = reasoner.respond(history.snapshot())
    for step in result.tool_trace:
        status = "ok" if step["ok"] else "ERROR"
        print(f"  [tool:{status}] {step['tool']} {step['arguments']} -> {step['result']}")
    print(f"Agent: {result.text}")
    if not result.error:
        history.add("assistant", result.text)


def main() -> int:
    client = OllamaClient.from_env()
    print(f"Ollama at {client.host}, model {client.model}")
    if not client.is_available():
        print("Ollama is not reachable. Start it in another terminal: ollama serve")
        return 1
    if not client.model_installed():
        print(f"Model not installed. Run: ollama pull {client.model}")
        return 1

    reasoner = LLMReasoner(client, create_builtin_runner())
    history = ConversationHistory()

    if len(sys.argv) > 1:
        ask(reasoner, history, " ".join(sys.argv[1:]))
        return 0

    print("Type 'exit' to quit.")
    while True:
        try:
            text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if text.lower() in {"exit", "quit"}:
            break
        if text:
            ask(reasoner, history, text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
