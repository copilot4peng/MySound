import gc
import unittest
import weakref

from ui.i18n import EN_US, Translator, register_messages


class FakeWidget:
    def __init__(self):
        self.options = {}
        self.exists = True

    def configure(self, **options):
        self.options.update(options)

    def winfo_exists(self):
        return self.exists


class TranslationTests(unittest.TestCase):
    def test_language_switch_refreshes_all_registered_options(self):
        translator = Translator()
        button, entry = FakeWidget(), FakeWidget()
        self.assertIs(translator.bind(button, "保存修改"), button)
        translator.bind(entry, "搜索历史记录…", option="placeholder_text")
        translator.set_language("en_US")
        self.assertEqual(button.options["text"], "Save changes")
        self.assertEqual(entry.options["placeholder_text"], "Search history…")
        translator.set_language("zh_CN")
        self.assertEqual(button.options["text"], "保存修改")

    def test_rebinding_updates_source_and_format_values(self):
        translator = Translator()
        label = FakeWidget()
        translator.bind(label, "已导出：{path}", path="/音频/old.txt")
        translator.bind(label, "已导出：{path}", path="/音频/新文本.txt")
        translator.set_language("en_US")
        self.assertEqual(label.options["text"], "Exported: /音频/新文本.txt")

    def test_unbind_preserves_transcript_when_language_changes(self):
        translator = Translator()
        label = FakeWidget()
        translator.bind(label, "实时听写将在这里显示…")
        translator.unbind(label)
        label.configure(text="正在识别的用户原话")
        translator.set_language("en_US")
        self.assertEqual(label.options["text"], "正在识别的用户原话")

    def test_runtime_worker_messages_match_templates_without_changing_values(self):
        translator = Translator("en_US")
        self.assertEqual(
            translator.tr("正在转写第 2 段 · 30.0–59.5 秒"),
            "Transcribing segment 2 · 30.0–59.5 seconds",
        )
        self.assertEqual(
            translator.tr("音频文件不存在：/中文/{音频}.wav"),
            "Audio file does not exist: /中文/{音频}.wav",
        )
        self.assertEqual(translator.tr("设备 3（当前未发现）"), "Device 3 (not currently detected)")
        self.assertEqual(
            translator.tr("正在转写第 {index} 段 · {start:.1f}–{end:.1f} 秒", index=3, start=60, end=90.25),
            "Transcribing segment 3 · 60.0–90.2 seconds",
        )

    def test_unknown_text_model_names_and_paths_are_preserved(self):
        translator = Translator("en_US")
        for value in ("Qwen3-ASR-1.7B", "/模型/中文权重", "这是用户转写的原文", "literal {braces}"):
            self.assertEqual(translator.tr(value), value)

    def test_deleted_widgets_are_not_kept_alive_or_updated(self):
        translator = Translator()
        widget = FakeWidget()
        translator.bind(widget, "删除")
        widget.exists = False
        translator.set_language("en_US")
        self.assertEqual(widget.options["text"], "删除")
        live = FakeWidget()
        translator.bind(live, "停止")
        reference = weakref.ref(live)
        del live
        gc.collect()
        self.assertIsNone(reference())

    def test_unsupported_locale_uses_chinese(self):
        translator = Translator("unknown")
        self.assertEqual(translator.language, "zh_CN")
        self.assertEqual(translator.tr("保存修改"), "保存修改")

    def test_catalogue_formats_preserve_the_same_named_values(self):
        from string import Formatter
        formatter = Formatter()
        for source, english in EN_US.items():
            source_keys = {field for _, field, _, _ in formatter.parse(source) if field is not None}
            english_keys = {field for _, field, _, _ in formatter.parse(english) if field is not None}
            self.assertEqual(source_keys, english_keys, source)

    def test_feature_catalogue_registration_updates_runtime_templates(self):
        translator = Translator("en_US")
        register_messages({"测试任务已完成：{count}": "Test jobs completed: {count}"})
        self.assertEqual(translator.tr("测试任务已完成：4"), "Test jobs completed: 4")
        self.assertEqual(translator.tr("测试任务已完成：{count}", count=5), "Test jobs completed: 5")


if __name__ == "__main__":
    unittest.main()
