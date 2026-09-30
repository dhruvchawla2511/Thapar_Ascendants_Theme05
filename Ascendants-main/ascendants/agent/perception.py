"""
perception.py — turns raw multimodal input into one normalized shape.

The rest of the agent (fast path, tool engine, controller) shouldn't have
to care whether the user typed text, spoke into a mic, or sent a picture.
This module's only job is: take whatever came in, and hand back a
consistent `PerceptionResult` — or raise a typed `PerceptionError` instead
of crashing, if the input is malformed.

We deliberately do NOT do real speech-to-text or computer vision here.
That's future work. For now this is a normalization + validation layer.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Modality(str, Enum):
    TEXT = "text"
    AUDIO_WAV = "audio_wav"
    IMAGE_PNG = "image_png"


class PerceptionError(Exception):
    """Raised for malformed input. Callers should catch this instead of
    letting a bad input event crash the whole agent.
    """


@dataclass
class PerceptionResult:
    """The normalized shape every input type gets converted into."""

    modality: Modality
    payload: Any  # str for text; bytes for audio/image
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


# WAV files start with "RIFF"...."WAVE"
_WAV_RIFF_MAGIC = b"RIFF"
_WAV_FORMAT_MAGIC = b"WAVE"
# PNG files start with this fixed 8-byte signature
_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _normalize_text(event: dict[str, Any]) -> PerceptionResult:
    text = event.get("text")
    if not isinstance(text, str):
        raise PerceptionError("Text event missing a string 'text' field.")
    if text.strip() == "":
        raise PerceptionError("Text event has empty content.")
    return PerceptionResult(
        modality=Modality.TEXT,
        payload=text,
        metadata={"length": len(text)},
        timestamp=event.get("timestamp", time.time()),
    )


def _normalize_audio_wav(event: dict[str, Any]) -> PerceptionResult:
    data = event.get("data")
    if not isinstance(data, (bytes, bytearray)):
        raise PerceptionError("Audio event missing binary 'data' field.")
    if len(data) < 12 or data[0:4] != _WAV_RIFF_MAGIC or data[8:12] != _WAV_FORMAT_MAGIC:
        raise PerceptionError("Audio event data is not a valid WAV file (bad header).")
    return PerceptionResult(
        modality=Modality.AUDIO_WAV,
        payload=bytes(data),
        metadata={
            "size_bytes": len(data),
            "sample_rate": event.get("sample_rate"),
        },
        timestamp=event.get("timestamp", time.time()),
    )


def _normalize_image_png(event: dict[str, Any]) -> PerceptionResult:
    data = event.get("data")
    if not isinstance(data, (bytes, bytearray)):
        raise PerceptionError("Image event missing binary 'data' field.")
    if len(data) < 8 or bytes(data[0:8]) != _PNG_MAGIC:
        raise PerceptionError("Image event data is not a valid PNG file (bad signature).")
    return PerceptionResult(
        modality=Modality.IMAGE_PNG,
        payload=bytes(data),
        metadata={
            "size_bytes": len(data),
            "width": event.get("width"),
            "height": event.get("height"),
        },
        timestamp=event.get("timestamp", time.time()),
    )


_NORMALIZERS = {
    Modality.TEXT: _normalize_text,
    Modality.AUDIO_WAV: _normalize_audio_wav,
    Modality.IMAGE_PNG: _normalize_image_png,
}


def normalize(input_event: dict[str, Any]) -> PerceptionResult:
    """Normalize one raw input event into a PerceptionResult.

    `input_event` is expected to be a dict with at least a 'modality' key
    (one of "text", "audio_wav", "image_png") plus modality-specific
    fields (see the _normalize_* functions above).

    Raises PerceptionError on anything malformed — never lets a bad event
    propagate as an unhandled exception type, so the controller can catch
    just `PerceptionError` and keep the system running.
    """
    if not isinstance(input_event, dict):
        raise PerceptionError("Input event must be a dict.")

    raw_modality = input_event.get("modality")
    try:
        modality = Modality(raw_modality)
    except ValueError as exc:
        raise PerceptionError(
            f"Unknown or missing modality: {raw_modality!r}"
        ) from exc

    normalizer = _NORMALIZERS[modality]
    return normalizer(input_event)
