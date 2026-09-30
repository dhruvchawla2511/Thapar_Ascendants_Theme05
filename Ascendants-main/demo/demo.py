"""
demo.py — a tiny, real, runnable demonstration of the core behavior:

    OLD TASK -> USER INTERRUPTS -> CANCEL OLD -> UPDATE STATE ->
    NEW TASK -> IGNORE STALE RESULT -> FINAL RESPONSE

This does NOT use Shivansh's controller (it doesn't exist yet). It wires
together only Dhruv's three modules to prove they behave correctly
together. Run with:

    python demo/demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make `agent` importable when running this file directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.fast_path import FastPath
from agent.perception import normalize
from agent.tool_engine import ParamSpec, ToolEngine, ToolSpec


def banner(text: str) -> None:
    print(f"\n=== {text} ===")


def main() -> None:
    fast_path = FastPath()
    engine = ToolEngine()
    engine.register_tool(
        ToolSpec(
            name="search_flights",
            description="Search for flights to a destination.",
            parameters=(ParamSpec("destination", str),),
            state_changing=False,
        )
    )

    # ---- Step 1: user's first message ----------------------------------
    banner("USER")
    user_text_1 = "Book me a flight to Delhi."
    print(f"USER: {user_text_1}")

    perceived_1 = normalize({"modality": "text", "text": user_text_1})
    print(f"[perception] normalized -> modality={perceived_1.modality.value!r} "
          f"payload={perceived_1.payload!r}")

    ack_1 = fast_path.acknowledge(perceived_1.payload)
    banner("FAST PATH")
    print(f"FAST PATH: {ack_1}")

    banner("TOOL ENGINE — starting Delhi search")
    delhi_call = engine.create_call("search_flights", {"destination": "Delhi"})
    print(f"Created call: {delhi_call.to_dict()}")

    # ---- Step 2: the user interrupts before the Delhi search finishes --
    banner("USER INTERRUPTS")
    user_text_2 = "Actually Mumbai."
    print(f"USER: {user_text_2}")

    perceived_2 = normalize({"modality": "text", "text": user_text_2})

    ack_2 = fast_path.acknowledge_interruption(perceived_2.payload)
    banner("FAST PATH (interruption)")
    print(f"FAST PATH: {ack_2}")

    # This is exactly the coordination-layer behavior we need to prove:
    # 1. cancel the obsolete Delhi call
    banner("COORDINATION — cancel obsolete call")
    engine.cancel_call(delhi_call.call_id)
    print(f"Cancelled {delhi_call.call_id} "
          f"(status={engine.get_call(delhi_call.call_id).status.value})")

    # 2. update state (here: just the destination the user actually wants)
    current_destination = "Mumbai"
    print(f"Updated state: destination = {current_destination!r}")

    # 3. start the new, correct task
    banner("TOOL ENGINE — starting Mumbai search")
    mumbai_call = engine.create_call("search_flights", {"destination": current_destination})
    print(f"Created call: {mumbai_call.to_dict()}")

    # ---- Step 3: simulate results arriving, Delhi's arrives LATE -------
    banner("RESULTS ARRIVE (Delhi's stale result shows up late)")
    stale_result = {"flights": ["DL-101", "DL-202"]}
    completed_delhi = engine.complete_call(delhi_call.call_id, result=stale_result)
    print(f"Delhi call result arrived: {stale_result}")
    print(f"Delhi call status is still: {completed_delhi.status.value} -> IGNORED")

    real_result = {"flights": ["MU-501", "MU-777"]}
    completed_mumbai = engine.complete_call(mumbai_call.call_id, result=real_result)
    print(f"Mumbai call result arrived: {real_result}")
    print(f"Mumbai call status: {completed_mumbai.status.value} -> USED")

    # ---- Step 4: final response uses only the correct, current result --
    banner("FINAL RESPONSE")
    chosen_flight = completed_mumbai.result["flights"][0]
    print(f"Here are flights to {current_destination}: "
          f"{completed_mumbai.result['flights']}")
    print(f"(Delhi's stale result {completed_delhi.result} was correctly discarded.)")

    # ---- sanity assertions so this demo doubles as a smoke test --------
    assert completed_delhi.status.value == "cancelled"
    assert completed_mumbai.status.value == "completed"
    assert "Delhi" not in str(completed_mumbai.result)
    print("\n[demo] All interruption-handling assertions passed.")


if __name__ == "__main__":
    main()
