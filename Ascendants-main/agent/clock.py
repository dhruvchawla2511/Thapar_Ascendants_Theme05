"""
clock.py — an injectable time source.

Spec §4 describes the hidden evaluation environment as running on a
"Virtual Clock Streaming Harness: Deterministic event replay...". Before
this module existed, every timestamp this codebase ever produced
(StateSnapshot.last_updated, every TraceRecorder entry, perception's
fallback event timestamp) called `time.time()` directly and irrevocably —
there was no way for a deterministic replay harness (or our own tests) to
control what "now" means, so two components could disagree on ordering
under real wall-clock jitter, and a replayed run could never produce
byte-identical trace timestamps run to run.

Clock is the seam: RealClock (the default everywhere in production)
behaves identically to calling time.time() directly — this is a pure
refactor, zero behavior change for real usage. VirtualClock is what makes
deterministic tests possible: it starts at a fixed instant and only moves
forward when explicitly told to, so every component sharing ONE
VirtualClock instance always agrees on "now" — and if a future harness
wants to drive this agent on ITS OWN virtual clock, it can construct one
and pass it to AgentRunner without any other code change.
"""

from __future__ import annotations

import time
from typing import Protocol


class Clock(Protocol):
    def now(self) -> float: ...


class RealClock:
    """Wall-clock time — identical behavior to calling time.time()
    directly. The default everywhere in production.
    """

    def now(self) -> float:
        return time.time()


class VirtualClock:
    """A fully controllable clock for deterministic tests and harness-
    driven event replay.

    Starts at `start` (default 0.0) and NEVER advances on its own — call
    `.tick(seconds)` or `.set(t)` to move it forward explicitly. Cannot
    move backwards (that would make "last_updated" timestamps meaningless
    as a freshness/ordering signal). Share ONE instance across every
    component in a session (StateManager, TraceRecorder, ...) to get
    fully reproducible, jitter-free timestamps for a replayed scenario.
    """

    def __init__(self, start: float = 0.0) -> None:
        if start < 0:
            raise ValueError("VirtualClock start must be >= 0")
        self._now = float(start)

    def now(self) -> float:
        return self._now

    def tick(self, seconds: float = 1.0) -> float:
        """Advance the clock by `seconds` (must be >= 0) and return the
        new "now".
        """
        if seconds < 0:
            raise ValueError("VirtualClock cannot tick backwards")
        self._now += seconds
        return self._now

    def set(self, t: float) -> float:
        """Jump the clock forward to an absolute time `t` (must be >=
        the current time) and return it.
        """
        if t < self._now:
            raise ValueError("VirtualClock cannot move backwards")
        self._now = float(t)
        return self._now


# A single shared RealClock instance is enough for every production
# default — RealClock is stateless, so there is no isolation concern in
# reusing one instance everywhere (unlike VirtualClock, which is always
# constructed fresh per session/test).
REAL_CLOCK = RealClock()
