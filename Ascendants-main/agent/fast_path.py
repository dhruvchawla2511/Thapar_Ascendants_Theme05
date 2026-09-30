"""
fast_path.py — the "quick reflexes" of the agent.

FastPath's only job is to give the user an almost-instant acknowledgement
while the real work (Slow Path / tool calls) is still running. It must
NEVER claim that work is done unless it actually is — that's what makes
it safe to use: the user always knows the difference between
"I heard you, working on it" and "here is your actual result".

This module is intentionally independent of the event loop / controller.
It takes text in, returns text out. Shivansh's controller decides *when*
to call it (e.g. the moment a user message arrives, before the slow path
has produced anything).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import random


class IntentGuess(str, Enum):
    """A very lightweight, non-authoritative guess at what kind of thing
    the user is asking for. This is NOT the real intent classification —
    that belongs to the slow path / planner. FastPath only uses this to
    pick a more natural-sounding filler.
    """

    BOOKING = "booking"
    SEARCH = "search"
    QUESTION = "question"
    CORRECTION = "correction"
    UNKNOWN = "unknown"


_BOOKING_WORDS = {"book", "reserve", "order"}
_SEARCH_WORDS = {"find", "search", "look up", "show me"}
_CORRECTION_WORDS = {"actually", "wait", "no,", "instead", "sorry"}
_QUESTION_STARTERS = ("what", "when", "where", "who", "why", "how", "is", "are", "can")


def _guess_intent(user_text: str) -> IntentGuess:
    text = user_text.strip().lower()
    if not text:
        return IntentGuess.UNKNOWN
    if any(text.startswith(w) or f" {w} " in f" {text} " for w in _CORRECTION_WORDS):
        return IntentGuess.CORRECTION
    if any(w in text for w in _BOOKING_WORDS):
        return IntentGuess.BOOKING
    if any(w in text for w in _SEARCH_WORDS):
        return IntentGuess.SEARCH
    if text.startswith(_QUESTION_STARTERS) or text.endswith("?"):
        return IntentGuess.QUESTION
    return IntentGuess.UNKNOWN


_FILLERS: dict[IntentGuess, list[str]] = {
    IntentGuess.BOOKING: [
        "Got it — checking that for you.",
        "On it — looking into the booking now.",
        "Okay, working on that booking.",
    ],
    IntentGuess.SEARCH: [
        "Sure — searching now.",
        "Got it, let me look that up.",
        "Okay, searching for that.",
    ],
    IntentGuess.CORRECTION: [
        "Got it — switching to that instead.",
        "Understood, updating that now.",
        "No problem — changing that.",
    ],
    IntentGuess.QUESTION: [
        "Good question — let me check.",
        "One moment, looking into that.",
    ],
    IntentGuess.UNKNOWN: [
        "Got it — on it.",
        "Okay, working on that.",
        "Understood, one moment.",
    ],
}


@dataclass
class FastPath:
    """Generates fast acknowledgements without claiming completion.

    Keeps a small amount of state (`_last_filler`) purely so it doesn't
    repeat the exact same phrase twice in a row, which would feel robotic.
    This is cosmetic state only — it never affects correctness.
    """

    _last_filler: str | None = field(default=None, repr=False)
    _rng: random.Random = field(default_factory=random.Random, repr=False)

    def acknowledge(self, user_text: str) -> str:
        """Return a quick, honest acknowledgement of `user_text`.

        Never returns a completion claim (e.g. "booked", "done",
        "confirmed") — only an acknowledgement that work has started.
        """
        intent = _guess_intent(user_text)
        candidates = _FILLERS[intent]
        # Avoid repeating the exact same filler twice in a row.
        pool = [c for c in candidates if c != self._last_filler] or candidates
        choice = self._rng.choice(pool)
        self._last_filler = choice
        return choice

    def acknowledge_interruption(self, new_user_text: str) -> str:
        """Special-cased acknowledgement for when the user interrupts an
        in-flight task with a correction. Always phrased as a correction,
        regardless of what _guess_intent would otherwise say, because the
        controller is telling us explicitly that this is an interruption.
        """
        candidates = _FILLERS[IntentGuess.CORRECTION]
        pool = [c for c in candidates if c != self._last_filler] or candidates
        choice = self._rng.choice(pool)
        self._last_filler = choice
        return choice


_FORBIDDEN_COMPLETION_WORDS = (
    "booked",
    "confirmed",
    "done",
    "completed",
    "purchased",
    "reserved",
    "your flight is",
    "your order is",
)


def is_safe_fast_path_message(message: str) -> bool:
    """Guardrail helper (used by tests, and usable by the controller too):
    returns False if a proposed fast-path message would falsely claim
    completion. This lets the fast path stay a plain string generator
    while still being checkable/enforceable.
    """
    lowered = message.lower()
    return not any(bad in lowered for bad in _FORBIDDEN_COMPLETION_WORDS)
