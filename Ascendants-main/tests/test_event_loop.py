import asyncio
import unittest

from agent.event_loop import EventProcessor
from agent.interrupt_handler import InterruptHandler
from agent.state_manager import StateManager
from agent.trace import TraceRecorder


class RecordingFastPath:
    def __init__(self, filler: str | None = "okay, working on it"):
        self.calls = []
        self.filler = filler

    async def handle_text_chunk(self, text, end_of_turn, **kwargs):
        self.calls.append((text, end_of_turn))
        return self.filler


class RecordingTurnHandler:
    def __init__(self):
        self.turns = []

    async def start_turn(self, user_text: str) -> str:
        self.turns.append(user_text)
        return f"turn_{len(self.turns)}"


def make_processor(turn_handler=None, trace=None, fast_path=None):
    input_q: "asyncio.Queue" = asyncio.Queue()
    output_q: "asyncio.Queue" = asyncio.Queue()
    state = StateManager()
    interrupts = InterruptHandler()
    fast_path = fast_path if fast_path is not None else RecordingFastPath()
    proc = EventProcessor(
        input_queue=input_q,
        output_queue=output_q,
        interrupt_handler=interrupts,
        state_manager=state,
        fast_path=fast_path,
        turn_handler=turn_handler,
        trace=trace,
    )
    return proc, input_q, output_q, state, interrupts, fast_path


class TestEventProcessor(unittest.IsolatedAsyncioTestCase):
    async def test_text_chunk_forwards_to_fast_path_and_updates_state(self):
        proc, _in_q, output_q, state, _interrupts, fast_path = make_processor()
        await proc._dispatch({"type": "text_chunk", "text": "hello", "end_of_turn": False})
        self.assertEqual(fast_path.calls, [("hello", False)])
        self.assertEqual(state.get_slot("last_text_chunk"), "hello")
        self.assertIsNone(state.get_slot("turn_complete"))

    async def test_fast_path_acknowledgement_is_emitted_as_a_filler_action(self):
        """Spec §3.1 lists 'spoken fillers' as a first-class output type,
        and Response Latency is scored from trace logs — an acknowledgement
        that never reaches the output queue is invisible to the evaluator.
        """
        fast_path = RecordingFastPath(filler="Got it, one moment.")
        proc, _in_q, output_q, *_rest = make_processor(fast_path=fast_path)
        await proc._dispatch({"type": "text_chunk", "text": "hello", "end_of_turn": False})

        action = output_q.get_nowait()
        self.assertEqual(action["action"], "filler")
        self.assertEqual(action["text"], "Got it, one moment.")
        self.assertTrue(action["call_id"].startswith("filler_"))

    async def test_fast_path_returning_none_emits_no_filler(self):
        fast_path = RecordingFastPath(filler=None)
        proc, _in_q, output_q, *_rest = make_processor(fast_path=fast_path)
        await proc._dispatch({"type": "text_chunk", "text": "hello", "end_of_turn": False})
        self.assertTrue(output_q.empty())

    async def test_filler_actions_get_unique_call_ids_across_chunks(self):
        fast_path = RecordingFastPath(filler="ack")
        proc, _in_q, output_q, *_rest = make_processor(fast_path=fast_path)
        await proc._dispatch({"type": "text_chunk", "text": "a", "end_of_turn": False})
        await proc._dispatch({"type": "text_chunk", "text": "b", "end_of_turn": False})
        first = output_q.get_nowait()
        second = output_q.get_nowait()
        self.assertNotEqual(first["call_id"], second["call_id"])

    async def test_emitted_actions_are_recorded_to_trace_when_provided(self):
        trace = TraceRecorder(session_id="s1")
        fast_path = RecordingFastPath(filler="ack")
        proc, _in_q, output_q, *_rest = make_processor(fast_path=fast_path, trace=trace)
        await proc._dispatch({"type": "text_chunk", "text": "hi", "end_of_turn": False})
        actions = [e.action for e in trace.events() if e.kind == "action"]
        self.assertIn("filler", actions)

    async def test_end_of_turn_triggers_turn_handler_with_full_buffered_text(self):
        turn_handler = RecordingTurnHandler()
        proc, *_rest, state, _interrupts, _fp = make_processor(turn_handler)

        await proc._dispatch({"type": "text_chunk", "text": "book a flight ", "end_of_turn": False})
        await proc._dispatch({"type": "text_chunk", "text": "to Mumbai", "end_of_turn": True})

        self.assertEqual(turn_handler.turns, ["book a flight to Mumbai"])
        self.assertTrue(state.get_slot("turn_complete"))
        # Buffer must be cleared after the turn is dispatched.
        self.assertEqual(proc._turn_buffer, [])

    async def test_interruption_does_not_wipe_session_slots(self):
        """Regression test: interruption used to call state_manager.reset(),
        silently discarding every slot collected so far. It must now only
        cancel in-flight work and record the interruption timestamp.
        """
        proc, *_rest, state, interrupts, _fp = make_processor()
        state.update_slots({"origin": "Delhi", "destination": "Mumbai"})

        await proc._dispatch({"type": "interruption", "timestamp": 123.0})

        self.assertEqual(state.get_slot("origin"), "Delhi")
        self.assertEqual(state.get_slot("destination"), "Mumbai")
        self.assertEqual(state.get_slot("last_interruption"), 123.0)

    async def test_interruption_cancels_in_flight_tasks_and_emits_cancellations(self):
        proc, _in_q, output_q, _state, interrupts, _fp = make_processor()

        async def never_ending():
            await asyncio.sleep(3600)

        task = asyncio.create_task(never_ending())
        interrupts.register_task("call_1", task)

        await proc._dispatch({"type": "interruption", "timestamp": 1.0})

        self.assertTrue(task.cancelled())
        action = output_q.get_nowait()
        self.assertEqual(action, {"action": "cancel", "call_id": "call_1"})

    async def test_interruption_clears_pending_turn_buffer(self):
        proc, *_rest = make_processor()
        await proc._dispatch({"type": "text_chunk", "text": "partial", "end_of_turn": False})
        self.assertEqual(proc._turn_buffer, ["partial"])
        await proc._dispatch({"type": "interruption", "timestamp": 1.0})
        self.assertEqual(proc._turn_buffer, [])

    async def test_tool_result_resolves_pending_future(self):
        proc, *_rest = make_processor()
        future: "asyncio.Future" = asyncio.get_event_loop().create_future()
        proc.pending_tool_calls["call_1"] = future

        await proc._dispatch({"type": "tool_result", "call_id": "call_1", "result": 42})

        self.assertEqual(await future, 42)
        self.assertNotIn("call_1", proc.pending_tool_calls)

    async def test_tool_result_with_error_sets_exception(self):
        proc, *_rest = make_processor()
        future: "asyncio.Future" = asyncio.get_event_loop().create_future()
        proc.pending_tool_calls["call_1"] = future

        await proc._dispatch({"type": "tool_result", "call_id": "call_1", "error": "boom"})

        with self.assertRaises(RuntimeError):
            await future

    async def test_raw_input_valid_text_flows_through_as_a_turn(self):
        turn_handler = RecordingTurnHandler()
        proc, *_rest = make_processor(turn_handler)

        await proc._dispatch(
            {"type": "raw_input", "payload": {"modality": "text", "text": "hi there"}}
        )

        self.assertEqual(turn_handler.turns, ["hi there"])

    async def test_raw_input_well_formed_audio_grounds_into_a_turn(self):
        """Regression coverage for the actual grounding wiring: a WAV clip
        with proper metadata must reach the turn handler as a described
        turn, not just get validated and silently dropped.
        """
        import struct

        turn_handler = RecordingTurnHandler()
        proc, *_rest = make_processor(turn_handler)

        sample_rate = 16000
        num_samples = 16000  # ~1s
        byte_rate = sample_rate * 2
        data = b"\x00" * (num_samples * 2)
        wav = (
            b"RIFF"
            + struct.pack("<I", 36 + len(data))
            + b"WAVE"
            + b"fmt "
            + struct.pack("<IHHIIHH", 16, 1, 1, sample_rate, byte_rate, 2, 16)
            + b"data"
            + struct.pack("<I", len(data))
            + data
        )

        await proc._dispatch(
            {
                "type": "raw_input",
                "payload": {"modality": "audio_wav", "data": wav, "sample_rate": sample_rate},
            }
        )

        self.assertEqual(len(turn_handler.turns), 1)
        self.assertIn("audio clip", turn_handler.turns[0])

    async def test_raw_input_ambiguous_audio_emits_clarification_not_a_turn(self):
        """Well-formed per perception (passes magic-byte checks) but
        missing the metadata grounding needs — must ask for clarification
        rather than silently guessing or dropping the input.
        """
        turn_handler = RecordingTurnHandler()
        proc, _in_q, output_q, *_rest = make_processor(turn_handler)

        await proc._dispatch(
            {
                "type": "raw_input",
                "payload": {"modality": "audio_wav", "data": b"RIFF" + b"\x00" * 40 + b"WAVE" + b"\x00" * 40},
            }
        )

        self.assertEqual(turn_handler.turns, [])
        action = output_q.get_nowait()
        self.assertEqual(action["action"], "clarification_request")

    async def test_raw_input_malformed_audio_emits_clarification_request(self):
        proc, _in_q, output_q, *_rest = make_processor()

        await proc._dispatch(
            {
                "type": "raw_input",
                "payload": {"modality": "audio_wav", "data": b"not a real wav"},
            }
        )

        action = output_q.get_nowait()
        self.assertEqual(action["action"], "clarification_request")

    async def test_raw_input_missing_payload_emits_clarification_request(self):
        proc, _in_q, output_q, *_rest = make_processor()
        await proc._dispatch({"type": "raw_input"})
        action = output_q.get_nowait()
        self.assertEqual(action["action"], "clarification_request")

    async def test_unknown_event_type_is_logged_and_does_not_raise(self):
        proc, *_rest = make_processor()
        await proc._dispatch({"type": "something_unexpected"})  # must not raise

    async def test_emit_action_rejects_non_dict(self):
        proc, *_rest = make_processor()
        with self.assertRaises(TypeError):
            await proc.emit_action("not a dict")


class RecordingManifestHandler:
    def __init__(self, errors=None):
        self.manifests = []
        self._errors = errors or []

    def register_manifest(self, manifest):
        self.manifests.append(manifest)

        class _Result:
            def __init__(self, errors):
                self.errors = errors

        return _Result(self._errors)


class TestEventProcessorManifest(unittest.IsolatedAsyncioTestCase):
    async def test_tool_manifest_event_delegates_to_manifest_handler(self):
        input_q: "asyncio.Queue" = asyncio.Queue()
        output_q: "asyncio.Queue" = asyncio.Queue()
        state = StateManager()
        interrupts = InterruptHandler()
        fast_path = RecordingFastPath()
        manifest_handler = RecordingManifestHandler()
        proc = EventProcessor(
            input_queue=input_q,
            output_queue=output_q,
            interrupt_handler=interrupts,
            state_manager=state,
            fast_path=fast_path,
            manifest_handler=manifest_handler,
        )
        manifest = [{"name": "search_flights", "parameters": []}]
        await proc._dispatch({"type": "tool_manifest", "manifest": manifest})
        self.assertEqual(manifest_handler.manifests, [manifest])

    async def test_tool_manifest_with_no_handler_does_not_raise(self):
        proc, *_rest = make_processor()
        await proc._dispatch({"type": "tool_manifest", "manifest": []})  # must not raise

    async def test_tool_manifest_errors_emit_clarification_request(self):
        input_q: "asyncio.Queue" = asyncio.Queue()
        output_q: "asyncio.Queue" = asyncio.Queue()
        state = StateManager()
        interrupts = InterruptHandler()
        fast_path = RecordingFastPath()
        manifest_handler = RecordingManifestHandler(errors=["bad entry"])
        proc = EventProcessor(
            input_queue=input_q,
            output_queue=output_q,
            interrupt_handler=interrupts,
            state_manager=state,
            fast_path=fast_path,
            manifest_handler=manifest_handler,
        )
        await proc._dispatch({"type": "tool_manifest", "manifest": [{}]})
        action = output_q.get_nowait()
        self.assertEqual(action["action"], "clarification_request")


if __name__ == "__main__":
    unittest.main()
