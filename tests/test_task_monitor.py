import unittest

from ui.i18n import Translator
from ui.task_monitor import (
    ActivityLog,
    TaskState,
    format_bytes,
    format_duration,
    format_timestamp,
    timeline_bins,
)


class TaskStateTests(unittest.TestCase):
    def test_reduces_events_and_keeps_unknown_total_unknown(self):
        state = TaskState()
        state.handle_event({"kind": "task_started", "mode": "file", "source": "a.wav",
                            "model": "qwen", "target_seconds": 30, "max_seconds": 60})
        state.handle_event({"kind": "media_info", "metadata": {"name": "a.wav", "duration": 90}})
        state.handle_event({"kind": "chunk", "chunk": {"id": 1, "start": 0, "end": 12, "status": "processing"}})
        state.handle_event({"kind": "chunk", "chunk": {"id": 1, "status": "done"}})
        state.handle_event({"kind": "chunk", "chunk": {"id": "silence-2", "start": 12, "end": 20,
                                                              "status": "skipped", "reason": "silence"}})
        self.assertEqual(state.status, "running")
        self.assertEqual(state.discovered_count, 2)
        self.assertIsNone(state.final_count)
        self.assertEqual(state.chunks[0]["end"], 12)
        state.handle_event({"kind": "done", "discovered_chunks": 4})
        self.assertEqual(state.final_count, 4)

    def test_new_task_resets_previous_chunks(self):
        state = TaskState()
        state.handle_event({"type": "task_started", "mode": "mic", "source": "Mic"})
        state.handle_event({"type": "chunk", "chunk": {"id": 1, "start": 0, "end": 1}})
        state.handle_event({"type": "task_started", "mode": "file", "source": "next.wav"})
        self.assertEqual(state.mode, "file")
        self.assertEqual(state.source, "next.wav")
        self.assertEqual(state.chunks, [])

    def test_cancelled_task_still_reports_discovered_count(self):
        state = TaskState()
        state.handle_event({"kind": "task_started", "mode": "file"})
        state.handle_event({"kind": "done", "cancelled": True, "discovered_chunks": 3})
        self.assertEqual(state.status, "cancelled")
        self.assertIsNone(state.final_count)
        self.assertEqual(state.discovered_count, 3)

    def test_status_counts_do_not_treat_discovered_as_completed(self):
        state = TaskState()
        state.handle_event({"kind": "task_started", "mode": "file"})
        for ident, status in ((1, "done"), (2, "skipped"), (3, "pending"), (4, "error")):
            state.handle_event({"kind": "chunk", "chunk": {
                "id": ident, "start": ident, "end": ident + 1, "status": status,
            }})
        self.assertEqual(state.discovered_count, 4)
        self.assertEqual(state.completed_count, 1)
        self.assertEqual(state.skipped_count, 1)
        self.assertEqual(state.pending_count, 1)
        self.assertEqual(state.error_count, 1)

    def test_normalized_duration_falls_back_when_missing_or_null(self):
        state = TaskState()
        state.handle_event({"kind": "media_info", "metadata": {"duration": 90,
                                                                   "normalized_duration": None}})
        self.assertEqual(state.duration, 90)
        state.handle_event({"kind": "media_info", "metadata": {"normalized_duration": 84}})
        self.assertEqual(state.duration, 84)

    def test_current_segment_uses_chunk_id_after_silence_chunks(self):
        state = TaskState()
        state.handle_event({"kind": "task_started", "mode": "file"})
        state.handle_event({"kind": "chunk", "chunk": {"id": "silence-0", "status": "skipped",
                                                              "start": 0, "end": 30}})
        state.handle_event({"kind": "chunk", "chunk": {"id": 42, "status": "processing",
                                                              "start": 30, "end": 60}})
        self.assertEqual(state.current[0], 42)


class FormattingTests(unittest.TestCase):
    def test_format_helpers(self):
        self.assertEqual(format_duration(0), "0:00")
        self.assertEqual(format_duration(3661), "1:01:01")
        self.assertEqual(format_duration(None), "—")
        self.assertEqual(format_bytes(1024), "1.0 KB")
        self.assertEqual(format_bytes(5), "5 B")
        self.assertEqual(format_timestamp("12:34:56"), "12:34:56")

    def test_timeline_bins_are_bounded_and_keep_real_time(self):
        chunks = [{"id": i, "start": i, "end": i + 1, "status": "done"} for i in range(1000)]
        chunks[300]["status"] = "error"
        result = timeline_bins(chunks, 1000, width=100)
        self.assertLessEqual(len(result), 100)
        self.assertTrue(any(item["status"] == "error" for item in result))
        self.assertTrue(all(item["start"] < item["end"] for item in result))

    def test_single_short_chunk_does_not_fill_unknown_tail(self):
        result = timeline_bins([{"id": 1, "start": 0, "end": 30, "status": "done"}], 90, width=90)
        self.assertTrue(result)
        self.assertLessEqual(max(item["end"] for item in result), 30)

    def test_activity_log_keeps_only_recent_entries_without_gui(self):
        log = ActivityLog.__new__(ActivityLog)
        log.entries = []
        for index in range(ActivityLog.MAX_ENTRIES + 7):
            log.append(f"message-{index}")
        self.assertEqual(len(log.entries), ActivityLog.MAX_ENTRIES)
        self.assertEqual(log.entries[0]["message"], "message-7")


class TranslationTests(unittest.TestCase):
    def test_monitor_messages_are_registered(self):
        translator = Translator("en_US")
        self.assertEqual(translator.tr("任务监测"), "Task monitor")
        self.assertEqual(translator.tr("发现 {count} 段", count=3), "3 segments found")


if __name__ == "__main__":
    unittest.main()
