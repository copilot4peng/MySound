import importlib.util
import unittest
from unittest.mock import MagicMock

from core.mic_streamer import MicStreamer


@unittest.skipUnless(importlib.util.find_spec("numpy"), "NumPy is required for audio tests")
class MicStreamerTests(unittest.TestCase):
    def setUp(self):
        import numpy as np
        self.np = np

    def make_streamer(self):
        from core.model_manager import TranscriptSegment

        class FakeModel:
            def __init__(self):
                self.calls = []

            def transcribe(self, audio, sample_rate=16000, language=None):
                self.calls.append(audio.copy())
                return [TranscriptSegment(0, len(audio) / sample_rate, "测试")]

        model = FakeModel()
        segments, errors, done = [], [], []
        streamer = MicStreamer(model, on_segment=segments.append, on_error=errors.append,
                               on_done=lambda: done.append(True))
        streamer._np = self.np
        return streamer, model, segments, errors, done

    def test_stop_drains_last_short_utterance(self):
        streamer, model, segments, errors, done = self.make_streamer()
        streamer._queue.put((16000, self.np.full(320, 0.1, dtype=self.np.float32)))
        streamer.stop()
        streamer._run()
        self.assertEqual(len(model.calls), 1)
        self.assertAlmostEqual(segments[0].start, 1.0)
        self.assertAlmostEqual(segments[0].end, 1.02)
        self.assertEqual(errors, [])
        self.assertEqual(done, [True])

    def test_silence_does_not_generate_hallucinated_transcript(self):
        streamer, model, segments, errors, done = self.make_streamer()
        streamer._queue.put((0, self.np.zeros(320, dtype=self.np.float32)))
        streamer.stop()
        streamer._run()
        self.assertEqual(model.calls, [])
        self.assertEqual(segments, [])
        self.assertEqual(done, [True])

    def test_partial_replacement_and_stop_finalize_the_entire_utterance(self):
        streamer, model, segments, errors, done = self.make_streamer()
        partials = []

        def receive_partial(text):
            partials.append(text)
            if text:
                streamer.stop()

        streamer.on_partial = receive_partial
        for index in range(115):
            streamer._queue.put((index * 320, self.np.full(320, 0.1, dtype=self.np.float32)))
        streamer._run()
        self.assertEqual(partials, ["测试", ""])
        self.assertEqual([len(audio) for audio in model.calls], [32000, 36800])
        self.assertEqual(len(segments), 1)
        self.assertAlmostEqual(segments[0].end, 2.3)
        self.assertEqual(errors, [])

    def test_queue_overload_stops_explicitly(self):
        streamer, model, segments, errors, done = self.make_streamer()

        class FakeSoundDevice:
            class CallbackStop(Exception):
                pass

        streamer._sd = FakeSoundDevice
        samples = self.np.zeros((320, 1), dtype=self.np.float32)
        for _ in range(streamer._queue.maxsize):
            streamer._capture(samples, 320, None, None)
        with self.assertRaises(FakeSoundDevice.CallbackStop):
            streamer._capture(samples, 320, None, None)
        self.assertTrue(streamer._stop_requested.is_set())
        self.assertIsNotNone(streamer._pending_error)
        self.assertEqual(streamer._queue.qsize(), streamer._queue.maxsize)

    def test_open_capture_retries_native_rates_and_reports_capture_info(self):
        class FakeStream:
            samplerate = 48000
            device = 7
            def start(self): pass
            def stop(self): pass
            def close(self): pass

        class FakeSD:
            CallbackStop = RuntimeError
            def __init__(self):
                self.checked = []
            def query_devices(self, device, kind):
                return {"name": "USB Mic", "max_input_channels": 1, "default_samplerate": 48000}
            def check_input_settings(self, **kwargs):
                self.checked.append(kwargs["samplerate"])
                if kwargs["samplerate"] == 16000:
                    raise RuntimeError("Error opening InputStream: Invalid sample rate [PaErrorCode -9997]")
            def InputStream(self, **kwargs):
                self.open_kwargs = kwargs
                return FakeStream()

        info = []
        streamer = MicStreamer(MagicMock(), device=7, on_capture_info=info.append)
        streamer._sd = FakeSD()
        streamer._open_capture()
        self.assertEqual(streamer._sd.checked[:2], [16000, 48000])
        self.assertEqual(streamer.capture_sample_rate, 48000)
        self.assertEqual(streamer._sd.open_kwargs["blocksize"], 960)
        # _run emits the callback; starting it directly avoids microphone I/O.
        streamer.stop()
        streamer._run()
        self.assertEqual(info[0]["capture_sample_rate"], 48000)
        self.assertEqual(info[0]["sample_rate"], 16000)

    def test_decode_resamples_one_complete_utterance_and_uses_native_clock(self):
        streamer, model, segments, errors, done = self.make_streamer()
        streamer.capture_sample_rate = 48000
        streamer.sample_rate = 16000
        streamer._decode([self.np.full(480, 0.1, dtype=self.np.float32)], 4800, True)
        self.assertEqual(model.calls[0].shape[0], 160)
        self.assertAlmostEqual(segments[0].start, 0.1)
        self.assertAlmostEqual(segments[0].end, 0.11, places=3)

    def test_non_rate_open_error_is_not_retried_and_stream_is_closed(self):
        class FakeStream:
            samplerate = 16000
            def __init__(self): self.closed = 0
            def start(self): raise PermissionError("device busy")
            def stop(self): self.closed += 1
            def close(self): self.closed += 1
        class FakeSD:
            CallbackStop = RuntimeError
            def __init__(self): self.checked = []; self.stream = None
            def query_devices(self, device, kind): return {"name": "Mic", "max_input_channels": 1, "default_samplerate": 48000}
            def check_input_settings(self, **kwargs): self.checked.append(kwargs["samplerate"])
            def InputStream(self, **kwargs): self.stream = FakeStream(); return self.stream
        streamer = MicStreamer(MagicMock())
        streamer._sd = FakeSD()
        with self.assertRaises(RuntimeError) as caught:
            streamer._open_capture()
        self.assertEqual(streamer._sd.checked, [16000])
        self.assertEqual(streamer._sd.stream.closed, 2)
        self.assertIn("microphone", caught.exception.message_en)
        self.assertIn("device busy", caught.exception.details)

    def test_all_invalid_rates_reports_attempts_and_cleans_stream(self):
        class FakeSD:
            CallbackStop = RuntimeError
            def __init__(self): self.checked = []
            def query_devices(self, device, kind): return {"name": "Mic", "max_input_channels": 1, "default_samplerate": 44100}
            def check_input_settings(self, **kwargs):
                self.checked.append(kwargs["samplerate"])
                raise RuntimeError("PortAudio error -9997")
            def InputStream(self, **kwargs): raise AssertionError("check should reject before opening")
        streamer = MicStreamer(MagicMock())
        streamer._sd = FakeSD()
        with self.assertRaises(RuntimeError) as caught:
            streamer._open_capture()
        self.assertEqual(caught.exception.args[0].startswith("麦克风不支持"), True)
        self.assertIn("44100", caught.exception.details)
        self.assertIsNone(streamer._stream)


if __name__ == "__main__":
    unittest.main()
