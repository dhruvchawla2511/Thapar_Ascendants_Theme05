"""
core/interrupt_handler.py

Interruption recovery and task cancellation for a real-time conversational
agent. Tracks in-flight asyncio Tasks (e.g. tool calls, generation streams)
keyed by call_id so they can be cancelled cleanly when the user interrupts
(e.g. barge-in on a voice agent) or the session is torn down.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class InterruptHandler:
    """
    Tracks and cancels in-flight asyncio Tasks for a session.

    Not thread-safe by design (asyncio.Task registration/cancellation is
    expected to happen on the event loop), but safe across concurrent
    coroutines on the same loop since dict mutation and cancellation here
    are not split by an `await` that yields control mid-mutation.
    """

    def __init__(self) -> None:
        self.active_tasks: Dict[str, asyncio.Task] = {}

    def register_task(self, call_id: str, task: asyncio.Task) -> None:
        """Register a new background task under call_id."""
        if not call_id:
            raise ValueError("call_id must be a non-empty string")
        self.active_tasks[call_id] = task

    def unregister_task(self, call_id: str) -> None:
        """Remove a task from tracking (e.g. upon natural completion)."""
        self.active_tasks.pop(call_id, None)

    async def cancel_all_tasks(self) -> List[str]:
        """
        Cancel every registered task immediately.

        Calls .cancel() on each task, awaits them to let cancellation
        propagate (swallowing asyncio.CancelledError, which is the expected
        result of a successful cancel), clears the registry, and returns
        the list of call_ids that were cancelled.
        """
        # Snapshot to a plain list BEFORE any `await` in this method. A
        # cancelled task's own coroutine (e.g. ToolRunner.run_cancellable)
        # may call unregister_task() on ITS OWN call_id as part of handling
        # its cancellation — and since `await task` below yields control
        # back to the event loop, that unregister can happen while this
        # method is still iterating. Iterating a live `dict.items()` view
        # in that situation raises "dictionary changed size during
        # iteration"; iterating a snapshot list is immune to it, and the
        # final `.clear()` still leaves active_tasks empty either way.
        tasks_snapshot = list(self.active_tasks.items())
        cancelled_ids = [call_id for call_id, _task in tasks_snapshot]

        for call_id, task in tasks_snapshot:
            if not task.done():
                task.cancel()

        for call_id, task in tasks_snapshot:
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:
                # A task may raise a different exception if it caught
                # CancelledError internally and re-raised something else,
                # or had already failed before cancellation reached it.
                # Log and continue so one bad task doesn't block cleanup.
                logger.exception(
                    "Task %s raised during cancellation", call_id
                )

        self.active_tasks.clear()
        return cancelled_ids

    def format_cancellation_payload(self, call_id: str) -> Dict[str, Any]:
        """Build the standard cancellation message payload for a call_id."""
        return {"action": "cancel", "call_id": call_id}
