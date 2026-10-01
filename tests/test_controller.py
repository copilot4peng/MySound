from contextlib import contextmanager
from copy import deepcopy
import threading
import unittest
from unittest.mock import Mock, patch

from core.audio_processor import AudioChunk
from core.controller import TranscriptionController
from core.model_manager import TranscriptSegment


class FakeHistory:
    def __init__(self):
        self.records = []

    def add(self, **data):
        record = {"id": str(len(self.records) + 1), **deepcopy(data)}
        self.records.append(record)
        return record


class FakeProcessor:
    def __init__(self, chunks):
        self.chunks = chunks
        self.closed = False

    @contextmanager
    def process(self, path, cancel_event=None, on_status=None):
        try:
            yield iter(self.chunks)
        finally:
            self.closed = True


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.history = FakeHistory()
        self.events = []
        self.done = threading.Event()

        def emit(event):
            self.events.append(event)
            if event["kind"] == "done":
                self.done.set()

        self.controller = TranscriptionController(self.history, emit)
        self.model = Mock()
        self.controller.manager = Mock()
        self.controller.manager.load.return_value = self.model
        self.controller.manager.transcribe = self.model.transcribe
        self.config = {"model_type": "whisper", "model_path": "/models/test.pt", "language": "zh"}

    def await_done(self):
        self.assertTrue(self.done.wait(timeout=3), "The controller failed to finish")
        self.assertFalse(self.controller.busy)
        return [event for event in self.events if event["kind"] == "done"][-1]

    def test_file_offsets_are_absolute_clamped_and_saved(self):
        processor = FakeProcessor([
            AudioChunk("first", 10, 20), AudioChunk("second", 30, 35),
        ])
        self.model.transcribe.side_effect = [
            [TranscriptSegment(1, 3, "第一句")],
            [TranscriptSegment(-1, 9, "第二句"), TranscriptSegment(8, 9, "越界")],
        ]
        with patch("core.audio_processor.AudioProcessor", return_value=processor):
            self.controller.start_file("meeting.mp4", self.config)
            done = self.await_done()
        self.assertFalse(done["failed"])
        self.assertFalse(done["cancelled"])
        self.assertTrue(processor.closed)
        self.assertEqual(len(self.history.records), 1)
        record = self.history.records[0]
        self.assertEqual(record["source"], "meeting.mp4")
        self.assertEqual(record["text"], "第一句\n第二句")
        self.assertEqual(record["segments"], [
            {"start": 11, "end": 13, "text": "第一句"},
            {"start": 30, "end": 35, "text": "第二句"},
        ])
        self.model.transcribe.assert_any_call("first", sample_rate=16000, language="zh")

    def test_file_events_report_task_media_chunks_and_real_completion_progress(self):
        processor = FakeProcessor([AudioChunk("first", 10, 20), AudioChunk("second", 30, 35)])
        self.model.transcribe.side_effect = [
            [TranscriptSegment(0, 1, "一")], [TranscriptSegment(0, 1, "二")]
        ]
        processor.process = lambda path, cancel_event=None, on_status=None, on_event=None: self._fake_process(processor, on_event)
        with patch("core.audio_processor.AudioProcessor", return_value=processor):
            self.controller.start_file("meeting.wav", {**self.config, "vad_target_seconds": 30, "vad_max_seconds": 60})
            done = self.await_done()
        kinds = [event["kind"] for event in self.events]
        self.assertEqual(kinds[0], "task_started")
        self.assertIn("media_info", kinds)
        chunks = [event["chunk"] for event in self.events if event["kind"] == "chunk"]
        self.assertEqual([chunk["status"] for chunk in chunks], ["pending", "processing", "done", "pending", "processing", "done"])
        self.assertEqual(done["completed_chunks"], 2)
        self.assertEqual(done["total_chunks"], 2)
        self.assertEqual(done["progress"], 1.0)
        self.assertTrue(all("timestamp" in event for event in self.events))

    def test_file_metadata_is_emitted_before_slow_model_load_and_probe_runs_once(self):
        processor = FakeProcessor([])
        seen = []
        def load(*args, **kwargs):
            seen.append([event["kind"] for event in self.events])
            return self.model

        self.controller.manager.load.side_effect = load
        with patch("core.audio_processor.AudioProcessor", return_value=processor), \
             patch("core.audio_processor.probe_media", return_value={"name": "meeting.wav", "path": "meeting.wav", "size_bytes": 7, "duration": 9.0, "format": "wav", "codec": "pcm_s16le", "sample_rate": 16000, "channels": 1}) as probe:
            self.controller.start_file("meeting.wav", self.config)
            done = self.await_done()
        self.assertEqual(probe.call_count, 1)
        self.assertIn("media_info", seen[0])
        self.assertEqual(done["duration"], 9.0)

    @staticmethod
    def _fake_process(processor, on_event):
        @contextmanager
        def context():
            if on_event:
                on_event({"kind": "media_info", "metadata": {"duration": 35, "name": "meeting.wav"}})
            def iterator():
                for chunk in processor.chunks:
                    if on_event:
                        on_event({"kind": "chunk", "chunk": {"id": chunk.id or 1, "start": chunk.start, "end": chunk.end, "status": "pending", "reason": chunk.reason}})
                    yield chunk
            yield iterator()
        return context()

    def test_file_cancellation_preserves_current_result_and_skips_next_chunk(self):
        processor = FakeProcessor([AudioChunk("first", 0, 2), AudioChunk("second", 2, 4)])
        entered, release = threading.Event(), threading.Event()

        def transcribe(*args, **kwargs):
            entered.set()
            if not release.wait(timeout=3):
                raise RuntimeError("Test did not release inference")
            return [TranscriptSegment(0, 2, "已识别")]

        self.model.transcribe.side_effect = transcribe
        with patch("core.audio_processor.AudioProcessor", return_value=processor):
            self.controller.start_file("meeting.wav", self.config)
            try:
                self.assertTrue(entered.wait(timeout=3))
                self.controller.stop()
                self.assertTrue(self.controller.busy)
            finally:
                release.set()
            done = self.await_done()
        self.assertTrue(done["cancelled"])
        self.assertFalse(done["failed"])
        self.assertEqual(self.model.transcribe.call_count, 1)
        self.assertEqual(self.history.records[0]["text"], "已识别")
        self.assertTrue(processor.closed)
        self.assertEqual(done["completed_chunks"], 1)
        self.assertEqual(done["discovered_chunks"], 1)
        self.assertIsNone(done["total_chunks"])
        self.assertTrue(done["progress"] is None or done["progress"] < 1.0)

    def test_ffprobe_warning_is_visible_before_model_load_but_does_not_fail_task(self):
        processor = FakeProcessor([])
        with patch("core.audio_processor.AudioProcessor", return_value=processor), \
             patch("core.audio_processor.probe_media", return_value={"name": "meeting.wav", "path": "meeting.wav", "size_bytes": None, "duration": None, "probe_error": "ffprobe timed out"}):
            self.controller.start_file("meeting.wav", self.config)
            done = self.await_done()
        self.assertFalse(done["failed"])
        warnings = [event for event in self.events if event["kind"] == "status" and "媒体信息检测失败" in event["text"]]
        self.assertEqual(len(warnings), 1)
        self.assertIn("ffprobe timed out", warnings[0]["values"]["detail"])

    def test_mic_stop_saves_final_tail_before_completing(self):
        started = threading.Event()
        instances = []

        class FakeMic:
            def __init__(self, **callbacks):
                self.callbacks = callbacks
                instances.append(self)

            def start(self):
                self.callbacks["on_partial"]("未确认")
                started.set()

            def stop(self):
                self.callbacks["on_segment"](TranscriptSegment(1.5, 2, "最后一句"))
                self.callbacks["on_done"]()

        with patch("core.mic_streamer.MicStreamer", FakeMic):
            self.controller.start_mic(self.config)
            self.assertTrue(started.wait(timeout=3))
            self.assertTrue(self.controller.busy)
            self.assertEqual(self.history.records, [])
            self.controller.stop()
            done = self.await_done()
            instances[0].callbacks["on_done"]()  # Duplicate shutdown cannot save twice.
        self.assertFalse(done["failed"])
        self.assertEqual(len(self.history.records), 1)
        self.assertEqual(self.history.records[0]["source"], "麦克风")
        self.assertEqual(self.history.records[0]["text"], "最后一句")
        kinds = [event["kind"] for event in self.events]
        self.assertLess(kinds.index("segment"), kinds.index("done"))
        self.assertEqual(kinds.count("done"), 1)

    def test_model_failure_restores_idle_and_allows_retry(self):
        self.controller.manager.load.side_effect = RuntimeError("权重损坏")
        with self.assertLogs("core.controller", level="ERROR"):
            self.controller.start_file("meeting.wav", self.config)
            done = self.await_done()
        self.assertTrue(done["failed"])
        self.assertIsNone(done["record"])
        self.assertIn("权重损坏", next(event["text"] for event in self.events if event["kind"] == "error"))

        self.done.clear()
        self.controller.manager.load.side_effect = None
        self.model.transcribe.return_value = [TranscriptSegment(0, 1, "恢复成功")]
        with patch("core.audio_processor.AudioProcessor", return_value=FakeProcessor([AudioChunk("audio", 0, 1)])):
            self.controller.start_file("meeting.wav", self.config)
            done = self.await_done()
        self.assertFalse(done["failed"])
        self.assertEqual(self.history.records[0]["text"], "恢复成功")

    def test_cancelling_during_mic_model_load_never_starts_capture(self):
        entered, release = threading.Event(), threading.Event()

        def load(*args, **kwargs):
            entered.set()
            if not release.wait(timeout=3):
                raise RuntimeError("Test did not release model load")
            return self.model

        self.controller.manager.load.side_effect = load
        with patch("core.mic_streamer.MicStreamer") as mic_class:
            self.controller.start_mic(self.config)
            try:
                self.assertTrue(entered.wait(timeout=3))
                self.controller.stop()
            finally:
                release.set()
            done = self.await_done()
            mic_class.assert_not_called()
        self.assertTrue(done["cancelled"])
        self.assertFalse(done["failed"])
        self.assertEqual(self.history.records, [])


if __name__ == "__main__":
    unittest.main()
