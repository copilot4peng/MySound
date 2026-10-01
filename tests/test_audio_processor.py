import importlib.util
from pathlib import Path
import shutil
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import wave

from core.audio_processor import AudioProcessor, _choose_boundary, probe_media


class BoundaryTests(unittest.TestCase):
    def test_uses_natural_pause_after_target(self):
        speech = [{"start": 0, "end": 31}, {"start": 35, "end": 60}]
        self.assertEqual(_choose_boundary(speech, 60, 30, 60), 33)

    def test_avoids_cutting_long_speech_after_earlier_pause(self):
        speech = [{"start": 0, "end": 23}, {"start": 25, "end": 60}]
        self.assertEqual(_choose_boundary(speech, 60, 30, 60), 24)

    def test_no_pause_obeys_hard_limit(self):
        self.assertEqual(_choose_boundary([{"start": 0, "end": 60}], 60, 30, 60), 60)

    def test_short_tail_and_silence_advance(self):
        self.assertEqual(_choose_boundary([], 60, 30, 60), 60)
        self.assertEqual(_choose_boundary([{"start": 0, "end": 10}], 12, 30, 60), 12)

    def test_invalid_chunk_size(self):
        with self.assertRaises(ValueError):
            AudioProcessor(target_seconds=60, max_seconds=30)

    def test_ffprobe_metadata_failure_keeps_filesystem_facts(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "meeting.wav"
            source.write_bytes(b"not media")
            with patch("core.audio_processor.shutil.which", return_value=None):
                info = probe_media(source)
            self.assertEqual(info["name"], "meeting.wav")
            self.assertEqual(info["path"], str(source))
            self.assertEqual(info["size_bytes"], 9)
            self.assertIsNone(info["duration"])
            self.assertIn("probe_error", info)


@unittest.skipUnless(importlib.util.find_spec("numpy"), "NumPy is required for audio tests")
class ChunkTests(unittest.TestCase):
    def test_bounded_chunks_preserve_source_offsets_and_cleanup(self):
        import numpy as np

        processor = AudioProcessor(target_seconds=0.5, max_seconds=1)
        processor._vad = object()
        sizes = []

        def speech_timestamps(audio, model, **kwargs):
            sizes.append(len(audio))
            return [{"start": 0, "end": len(audio)}]

        processor._get_timestamps = speech_timestamps
        normalized_paths = []

        def normalize(source, destination, cancel_event, on_status):
            normalized_paths.append(destination)
            shutil.copyfile(source, destination)

        processor._normalize = normalize
        fake_torch = types.SimpleNamespace(from_numpy=lambda value: value)
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.wav"
            with wave.open(str(source), "wb") as output:
                output.setnchannels(1)
                output.setsampwidth(2)
                output.setframerate(16000)
                output.writeframes(np.full(40000, 1000, dtype="<i2").tobytes())
            with patch.dict(sys.modules, {"torch": fake_torch}):
                with processor.process(source) as iterator:
                    chunks = list(iterator)
                    self.assertTrue(normalized_paths[0].exists())
            self.assertFalse(normalized_paths[0].exists())
        self.assertEqual(sizes, [16000, 16000, 8000])
        self.assertEqual([(chunk.start, chunk.end) for chunk in chunks], [(0, 1), (1, 2), (2, 2.5)])
        self.assertTrue(all(chunk.audio.dtype == np.float32 for chunk in chunks))

    def test_silence_emits_skip_event_without_allocating_an_asr_chunk(self):
        import numpy as np
        processor = AudioProcessor(target_seconds=0.5, max_seconds=1)
        processor._vad = object()
        processor._get_timestamps = lambda *args, **kwargs: []
        processor._normalize = lambda source, destination, cancel_event, on_status: shutil.copyfile(source, destination)
        events = []
        fake_torch = types.SimpleNamespace(from_numpy=lambda value: value)
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "silence.wav"
            with wave.open(str(source), "wb") as output:
                output.setnchannels(1); output.setsampwidth(2); output.setframerate(16000)
                output.writeframes(np.zeros(16000, dtype="<i2").tobytes())
            with patch.dict(sys.modules, {"torch": fake_torch}):
                with processor.process(source, on_event=events.append) as chunks:
                    self.assertEqual(list(chunks), [])
        skipped = [event for event in events if event["kind"] == "chunk"]
        self.assertEqual(len(skipped), 1)
        self.assertEqual(skipped[0]["chunk"]["status"], "skipped")
        self.assertEqual(skipped[0]["chunk"]["reason"], "silence")
        media = [event for event in events if event["kind"] == "media_info"]
        self.assertGreaterEqual(len(media), 2)
        self.assertEqual(media[-1]["metadata"]["normalized_duration"], 1.0)


if __name__ == "__main__":
    unittest.main()
