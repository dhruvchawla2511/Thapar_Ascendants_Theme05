import struct
import unittest

from agent.grounding import (
    AudioGroundingAdapter,
    GroundingResult,
    ImageGroundingAdapter,
    ground,
)
from agent.perception import normalize

PNG_HEADER = b"\x89PNG\r\n\x1a\n"


def make_canonical_wav(sample_rate=16000, channels=1, bits_per_sample=16, num_samples=8000):
    """Build a real, minimal, spec-correct 44-byte-header PCM WAV file so
    duration/sample-rate extraction has something genuine to read.
    """
    byte_rate = sample_rate * channels * bits_per_sample // 8
    block_align = channels * bits_per_sample // 8
    data = b"\x00" * (num_samples * block_align)
    header = (
        b"RIFF"
        + struct.pack("<I", 36 + len(data))
        + b"WAVE"
        + b"fmt "
        + struct.pack("<IHHIIHH", 16, 1, channels, sample_rate, byte_rate, block_align, bits_per_sample)
        + b"data"
        + struct.pack("<I", len(data))
    )
    return header + data


class TestAudioGrounding(unittest.TestCase):
    def test_well_formed_clip_is_sufficient(self):
        wav = make_canonical_wav(sample_rate=16000, num_samples=16000)  # ~1s
        result = normalize({"modality": "audio_wav", "data": wav, "sample_rate": 16000})
        g = ground(result)
        self.assertTrue(g.sufficient)
        self.assertIn("16000Hz", g.grounded_text)
        self.assertIn("1.00s", g.grounded_text)

    def test_missing_sample_rate_metadata_is_insufficient(self):
        wav = make_canonical_wav()
        result = normalize({"modality": "audio_wav", "data": wav})  # no sample_rate passed
        g = ground(result)
        self.assertFalse(g.sufficient)
        self.assertIsNone(g.grounded_text)
        self.assertIn("sample rate", g.reason)

    def test_too_short_clip_is_insufficient(self):
        wav = make_canonical_wav(sample_rate=16000, num_samples=1)  # basically 0s
        result = normalize({"modality": "audio_wav", "data": wav, "sample_rate": 16000})
        g = ground(result)
        self.assertFalse(g.sufficient)

    def test_minimal_riff_wave_only_header_is_insufficient(self):
        # Passes perception's magic-byte check but has no real fmt/data
        # chunks for the grounding adapter to read a duration from.
        data = b"RIFF" + b"\x00\x00\x00\x00" + b"WAVE" + b"\x00" * 100
        result = normalize({"modality": "audio_wav", "data": data, "sample_rate": 16000})
        g = ground(result)
        self.assertFalse(g.sufficient)


class TestImageGrounding(unittest.TestCase):
    def test_well_formed_frame_is_sufficient(self):
        data = PNG_HEADER + b"\x00" * 50
        result = normalize({"modality": "image_png", "data": data, "width": 640, "height": 480})
        g = ground(result)
        self.assertTrue(g.sufficient)
        self.assertIn("640x480", g.grounded_text)

    def test_missing_dimensions_is_insufficient(self):
        data = PNG_HEADER + b"\x00" * 50
        result = normalize({"modality": "image_png", "data": data})
        g = ground(result)
        self.assertFalse(g.sufficient)
        self.assertIn("width/height", g.reason)


class TestTextGrounding(unittest.TestCase):
    def test_text_is_always_sufficient_and_passthrough(self):
        result = normalize({"modality": "text", "text": "hello there"})
        g = ground(result)
        self.assertTrue(g.sufficient)
        self.assertEqual(g.grounded_text, "hello there")


class TestGroundingOverride(unittest.TestCase):
    def test_custom_adapter_can_be_plugged_in_without_changing_callers(self):
        """Proves a real ASR/vision model could be dropped in later just
        by supplying an adapter dict — no controller changes needed.
        """

        class AlwaysSufficientAudio:
            def ground(self, result):
                return GroundingResult(
                    modality="audio_wav", grounded_text="[real transcription]", sufficient=True
                )

        from agent.perception import Modality

        result = normalize({"modality": "audio_wav", "data": b"RIFFxxxxWAVE"})
        g = ground(result, adapters={Modality.AUDIO_WAV: AlwaysSufficientAudio()})
        self.assertTrue(g.sufficient)
        self.assertEqual(g.grounded_text, "[real transcription]")


if __name__ == "__main__":
    unittest.main()
