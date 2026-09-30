import unittest

from agent.perception import Modality, PerceptionError, normalize

WAV_HEADER = b"RIFF" + b"\x00\x00\x00\x00" + b"WAVE"
PNG_HEADER = b"\x89PNG\r\n\x1a\n"


class TestPerception(unittest.TestCase):
    def test_accepts_text(self):
        result = normalize({"modality": "text", "text": "Book me a flight to Mumbai."})
        self.assertEqual(result.modality, Modality.TEXT)
        self.assertEqual(result.payload, "Book me a flight to Mumbai.")
        self.assertEqual(result.metadata["length"], len("Book me a flight to Mumbai."))

    def test_accepts_wav_audio(self):
        data = WAV_HEADER + b"\x00" * 100
        result = normalize({"modality": "audio_wav", "data": data, "sample_rate": 16000})
        self.assertEqual(result.modality, Modality.AUDIO_WAV)
        self.assertEqual(result.payload, data)
        self.assertEqual(result.metadata["sample_rate"], 16000)
        self.assertEqual(result.metadata["size_bytes"], len(data))

    def test_accepts_png_image(self):
        data = PNG_HEADER + b"\x00" * 50
        result = normalize({"modality": "image_png", "data": data, "width": 640, "height": 480})
        self.assertEqual(result.modality, Modality.IMAGE_PNG)
        self.assertEqual(result.payload, data)
        self.assertEqual(result.metadata["width"], 640)
        self.assertEqual(result.metadata["height"], 480)

    def test_empty_text_raises_perception_error(self):
        with self.assertRaises(PerceptionError):
            normalize({"modality": "text", "text": "   "})

    def test_missing_text_field_raises_perception_error(self):
        with self.assertRaises(PerceptionError):
            normalize({"modality": "text"})

    def test_malformed_wav_header_raises_perception_error(self):
        with self.assertRaises(PerceptionError):
            normalize({"modality": "audio_wav", "data": b"not a wav file"})

    def test_malformed_png_header_raises_perception_error(self):
        with self.assertRaises(PerceptionError):
            normalize({"modality": "image_png", "data": b"not a png file"})

    def test_unknown_modality_raises_perception_error(self):
        with self.assertRaises(PerceptionError):
            normalize({"modality": "video_mp4", "data": b"\x00"})

    def test_explicit_event_timestamp_always_wins_over_the_clock(self):
        from agent.clock import VirtualClock

        clock = VirtualClock(start=500.0)
        result = normalize({"modality": "text", "text": "hi", "timestamp": 42.0}, clock=clock)
        self.assertEqual(result.timestamp, 42.0)

    def test_clock_is_used_as_fallback_when_event_has_no_timestamp(self):
        from agent.clock import VirtualClock

        clock = VirtualClock(start=500.0)
        result = normalize({"modality": "text", "text": "hi"}, clock=clock)
        self.assertEqual(result.timestamp, 500.0)

    def test_non_dict_input_raises_perception_error_not_crash(self):
        with self.assertRaises(PerceptionError):
            normalize("just a string, not an event dict")  # type: ignore[arg-type]

    def test_missing_binary_data_field_raises_perception_error(self):
        with self.assertRaises(PerceptionError):
            normalize({"modality": "audio_wav", "data": "not bytes"})


if __name__ == "__main__":
    unittest.main()
