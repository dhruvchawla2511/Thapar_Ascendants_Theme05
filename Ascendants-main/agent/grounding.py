"""
grounding.py — turns a normalized PerceptionResult into something the LLM
reasoning loop can actually use, or a signal that clarification is needed.

perception.py (Dhruv's module) validates and normalizes raw input, but it
deliberately does NOT do real speech-to-text or computer vision — that's
out of scope for this hackathon environment (no GPU, no model weights, no
network egress guaranteed). Before this module existed, that meant a WAV
or PNG event never actually reached the reasoner in any form; perception
successfully validating it was a dead end.

GroundingAdapter is the seam: each modality gets an adapter with the same
tiny interface (`ground(PerceptionResult) -> GroundingResult`), so a real
Whisper/vision model can be dropped in later by implementing that one
method — nothing in event_loop.py, orchestrator.py, or the reasoner needs
to change. The adapters registered by default here are honest mocks: they
never claim to have transcribed speech or recognized image content (that
would be fabricating perception), they only describe the *metadata* the
input actually carries (duration, sample rate, dimensions) in a bounded,
deterministic string, and they signal "insufficient" when that metadata
is missing — which the controller turns into a clarification_request
instead of silently guessing.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Optional, Protocol

from agent.perception import Modality, PerceptionResult


@dataclass(frozen=True)
class GroundingResult:
    modality: str
    # Text description to fold into the reasoning turn, e.g.
    # "[audio: 2.3s, 16000Hz, mono]". None if grounding was insufficient.
    grounded_text: Optional[str]
    sufficient: bool
    reason: Optional[str] = None  # why insufficient, when sufficient=False


class GroundingAdapter(Protocol):
    def ground(self, result: PerceptionResult) -> GroundingResult: ...


class TextGroundingAdapter:
    """Text needs no grounding step — it already IS the turn text."""

    def ground(self, result: PerceptionResult) -> GroundingResult:
        return GroundingResult(
            modality=Modality.TEXT.value, grounded_text=result.payload, sufficient=True
        )


def _wav_duration_seconds(data: bytes) -> Optional[float]:
    """Best-effort duration from a canonical PCM WAV header. Returns None
    (never raises) if the header doesn't have what's needed — this is a
    metadata reader, not a decoder.
    """
    try:
        if len(data) < 44:
            return None
        byte_rate = struct.unpack_from("<I", data, 28)[0]
        data_chunk_size = struct.unpack_from("<I", data, 40)[0]
        if byte_rate <= 0:
            return None
        return data_chunk_size / byte_rate
    except (struct.error, IndexError):
        return None


class AudioGroundingAdapter:
    """Mock adapter for WAV audio. Does NOT transcribe speech — describes
    the clip's metadata only, and reports insufficient if the essentials
    (sample rate, a non-trivial duration) aren't present, exactly the kind
    of thing a real ASR front-end would also need before it could even
    attempt a transcription.
    """

    MIN_DURATION_SECONDS = 0.05

    def ground(self, result: PerceptionResult) -> GroundingResult:
        sample_rate = result.metadata.get("sample_rate")
        duration = _wav_duration_seconds(result.payload)

        if not sample_rate:
            return GroundingResult(
                modality=Modality.AUDIO_WAV.value,
                grounded_text=None,
                sufficient=False,
                reason="Audio clip is missing a sample rate.",
            )
        if duration is None or duration < self.MIN_DURATION_SECONDS:
            return GroundingResult(
                modality=Modality.AUDIO_WAV.value,
                grounded_text=None,
                sufficient=False,
                reason="Audio clip is too short or its duration could not be determined.",
            )
        return GroundingResult(
            modality=Modality.AUDIO_WAV.value,
            grounded_text=f"[audio clip: {duration:.2f}s at {sample_rate}Hz]",
            sufficient=True,
        )


class ImageGroundingAdapter:
    """Mock adapter for PNG frames. Does NOT run vision/OCR — describes
    dimensions only, and reports insufficient if they're missing (the
    frame is present but the caller told us nothing about it).
    """

    def ground(self, result: PerceptionResult) -> GroundingResult:
        width = result.metadata.get("width")
        height = result.metadata.get("height")
        if not width or not height:
            return GroundingResult(
                modality=Modality.IMAGE_PNG.value,
                grounded_text=None,
                sufficient=False,
                reason="Image frame is missing width/height.",
            )
        return GroundingResult(
            modality=Modality.IMAGE_PNG.value,
            grounded_text=f"[image frame: {width}x{height}px]",
            sufficient=True,
        )


_DEFAULT_ADAPTERS: dict[Modality, GroundingAdapter] = {
    Modality.TEXT: TextGroundingAdapter(),
    Modality.AUDIO_WAV: AudioGroundingAdapter(),
    Modality.IMAGE_PNG: ImageGroundingAdapter(),
}


def ground(
    result: PerceptionResult, adapters: Optional[dict[Modality, GroundingAdapter]] = None
) -> GroundingResult:
    """Ground one PerceptionResult using the adapter registered for its
    modality. `adapters` lets a caller (e.g. a future real-ASR wiring)
    override individual modalities without touching this module or its
    callers — pass a dict overlaying only the modalities you want to
    replace; anything else falls back to the built-in mock adapters.
    """
    table = dict(_DEFAULT_ADAPTERS)
    if adapters:
        table.update(adapters)
    adapter = table.get(result.modality)
    if adapter is None:
        return GroundingResult(
            modality=result.modality.value,
            grounded_text=None,
            sufficient=False,
            reason=f"No grounding adapter registered for modality {result.modality.value!r}.",
        )
    return adapter.ground(result)
