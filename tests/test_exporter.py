from pathlib import Path
import tempfile
import unittest

from core.exporter import export_record


class ExporterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "transcript.srt"
        self.record = {
            "source": "demo.wav", "text": "原文一\n原文二",
            "segments": [
                {"start": 1.125, "end": 3.5, "text": "原文一"},
                {"start": 5, "end": 7.75, "text": "原文二"},
            ],
        }

    def test_original_subtitles_keep_absolute_timestamps(self):
        export_record(self.record, self.path)
        content = self.path.read_text(encoding="utf-8")
        self.assertIn("00:00:01,125 --> 00:00:03,500\n原文一", content)
        self.assertIn("2\n00:00:05,000 --> 00:00:07,750\n原文二", content)

    def test_edited_lines_replace_subtitles_at_existing_times(self):
        self.record["text"] = "改正一\n改正二"
        export_record(self.record, self.path)
        content = self.path.read_text(encoding="utf-8")
        self.assertNotIn("原文", content)
        self.assertIn("00:00:05,000 --> 00:00:07,750\n改正二", content)

    def test_edited_paragraph_uses_single_full_span_cue(self):
        self.record["text"] = "重新组织的段落。"
        export_record(self.record, self.path)
        self.assertEqual(
            self.path.read_text(encoding="utf-8"),
            "1\n00:00:01,125 --> 00:00:07,750\n重新组织的段落。\n",
        )

    def test_edits_to_word_spacing_are_preserved(self):
        self.record["segments"][0]["text"] = "helloworld"
        self.record["text"] = "hello world\n原文二"
        export_record(self.record, self.path)
        content = self.path.read_text(encoding="utf-8")
        self.assertIn("hello world", content)
        self.assertNotIn("helloworld", content)

    def test_srt_requires_real_timestamps(self):
        with self.assertRaises(ValueError):
            export_record({"text": "A transcript without timing"}, self.path)
        self.assertFalse(self.path.exists())

    def test_txt_and_markdown_use_edited_text(self):
        self.record["text"] = "手动修正"
        txt_path = self.path.with_suffix(".txt")
        md_path = self.path.with_suffix(".md")
        export_record(self.record, txt_path)
        export_record(self.record, md_path)
        self.assertEqual(txt_path.read_text(encoding="utf-8"), "手动修正\n")
        self.assertEqual(md_path.read_text(encoding="utf-8"), "# demo.wav\n\n手动修正\n")


if __name__ == "__main__":
    unittest.main()
