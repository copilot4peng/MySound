"""Opt-in desktop regression tests: MYSOUND_GUI_TESTS=1 python -m unittest ...

All settings/history use temporary directories. No model or microphone is loaded.
"""
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch


@unittest.skipUnless(os.environ.get("MYSOUND_GUI_TESTS") == "1", "requires a desktop display; opt in with MYSOUND_GUI_TESTS=1")
class DesktopTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import customtkinter as ctk
        from storage import BootstrapStore, DataDirectorySelection
        from ui.app import MySoundApp

        cls.ctk = ctk
        cls.temp = tempfile.TemporaryDirectory(prefix="mysound-ui-tests-")
        cls.root_path = Path(cls.temp.name)
        cls.bootstrap = BootstrapStore(cls.root_path / "bootstrap")
        cls.app = MySoundApp(
            data_selection=DataDirectorySelection(cls.root_path / "data", "default", False),
            bootstrap=cls.bootstrap,
        )
        cls.errors = []
        cls.app.report_callback_exception = lambda *args: cls.errors.append(args[1])
        cls.app.lift()
        cls.pump(0.5)

    @classmethod
    def pump(cls, seconds=0.15):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            cls.app.update()
            time.sleep(0.01)

    @classmethod
    def tearDownClass(cls):
        cls.ctk.set_widget_scaling(1.0)
        cls.app.destroy()
        cls.temp.cleanup()

    def setUp(self):
        self.app._set_busy(False)
        self.app.apply_preferences("dark", "blue", 14, "zh_CN")
        self.app._settings.font_size_var.set("14")
        self.app._settings.ui_language_var.set("简体中文")
        self.app.show_transcription()

    def tearDown(self):
        self.pump()
        self.assertEqual(self.errors, [], f"Tk callback errors: {self.errors}")

    def test_pages_preserve_editor_and_never_create_a_modal(self):
        app = self.app
        app._replace_text("正在编辑的文本，切换页面时必须保留。")
        editor = app.editor
        for _ in range(4):
            app._open_settings()
            self.pump(0.05)
            self.assertTrue(app._settings.winfo_ismapped())
            self.assertFalse(app.transcription_page.winfo_ismapped())
            self.assertIsNone(app.grab_current())
            app.show_transcription()
        self.assertIs(app.editor, editor)
        self.assertEqual(app._text(), "正在编辑的文本，切换页面时必须保留。")
        app._set_busy(True)
        app._open_settings()
        self.assertEqual(app.settings_button.cget("state"), "normal")
        for section, button in app._settings._save_buttons.items():
            self.assertEqual(button.cget("state"), "normal" if section == "外观" else "disabled")
        app.events.put({"kind": "segment", "segment": {"start": 0, "end": 2, "text": "后台追加的文字"}})
        self.pump()
        app.show_transcription()
        self.assertIn("后台追加的文字", app._text())

    def test_all_themes_update_existing_widgets(self):
        app = self.app
        app._replace_text("主题切换不能删除文本。\nTheme changes preserve edits.")
        record = app.history_store.add("很长的中文音频文件名用于检查侧栏是否会按宽度截断而不是挤出窗口.wav", "qwen3", "主题检查", [{"start": 0, "end": 2, "text": "主题检查"}])
        app._refresh_history()
        self.pump()
        self.assertIn("很长", app._history_buttons[0][0].cget("text"))
        old_editor = app.editor
        for mode in ("dark", "light"):
            for color in ("blue", "green", "purple", "orange"):
                app._settings.appearance_var.set({"dark": "深色", "light": "浅色"}[mode])
                app._settings.color_var.set({"blue": "蓝色", "green": "绿色", "purple": "紫色", "orange": "橙色"}[color])
                self.assertTrue(app._settings._save_appearance())
                self.pump(0.08)
                self.assertEqual(app.editor.cget("fg_color"), app.theme.palette["input"])
                self.assertEqual(app.horizontal.cget("bg"), app.theme.palette["border"])
                self.assertEqual(app._settings.tabs.cget("text_color"), app.theme.palette["text"])
                self.assertEqual(app.file_button.cget("fg_color"), app.theme.palette["accent"])
                self.assertEqual(app.config_store.load()["color_theme"], color)
                self.assertIs(old_editor, app.editor)
                self.assertIn("Theme changes preserve edits.", app._text())
                self._screenshot(f"transcription-{mode}-{color}")
        app._open_settings()
        for name in ("模型", "目录", "音频", "外观"):
            app._settings.tabs.set(name)
            self.pump()
            self._screenshot(f"settings-{name}")
        app.history_store.delete(record["id"])
        app._refresh_history()

    def test_draggable_sashes_and_layout_storage(self):
        app = self.app
        app.geometry("1180x780")
        self.pump()
        for pane, key, delta in ((app.horizontal, "sidebar_ratio", 50), (app.vertical, "editor_ratio", 45)):
            x, y = pane.sash_coord(0)
            horizontal = pane is app.horizontal
            x, y = (x + 3, 80) if horizontal else (80, y + 3)
            before = pane.sash_coord(0)[0 if horizontal else 1]
            pane.event_generate("<ButtonPress-1>", x=x, y=y)
            self.pump(0.05)
            pane.event_generate("<B1-Motion>", x=x + (delta if horizontal else 0), y=y + (0 if horizontal else delta))
            self.pump(0.05)
            pane.event_generate("<ButtonRelease-1>", x=x + (delta if horizontal else 0), y=y + (0 if horizontal else delta))
            self.pump()
            after = pane.sash_coord(0)[0 if horizontal else 1]
            self.assertGreater(after, before + 20)
            app._save_layout()
            self.assertAlmostEqual(app.config_store.load()["window_layout"][key], app._layout[key])
        app.reset_layout()

    def test_history_refreshes_without_resizing_and_keeps_scroll_and_rows(self):
        app = self.app
        app.current_record = None
        app._replace_text("")
        records = [app.history_store.add(f"会议记录-{index:02d}.wav", "whisper", f"第 {index} 条记录", []) for index in range(45)]
        try:
            app._refresh_history()
            self.pump()
            rows = dict(app._history_items)
            self.assertTrue(all(button.cget("text").strip() for button in rows.values()))
            canvas = app.history_list._parent_canvas
            canvas.yview_moveto(1.0)
            self.pump()
            before = canvas.yview()[0]
            geometry = app.geometry()
            for record in records[:4]:
                app._history_items[record["id"]].invoke()
                self.pump()
                self.assertEqual(app._text(), record["text"])
                self.assertEqual(app.geometry(), geometry)
                self.assertAlmostEqual(canvas.yview()[0], before, delta=0.025)
                self.assertIs(app._history_items[record["id"]], rows[record["id"]])
                self.assertTrue(all(button.cget("text").strip() for button in app._history_items.values()))
            app.search.insert(0, "会议记录-44")
            app._refresh_history()
            self.pump()
            self.assertEqual(len(app._history_buttons), 1)
            button = app._history_buttons[0][0]
            self.assertIn("会议记录", button.cget("text"))
            self.assertTrue(button.winfo_ismapped())
            self.assertAlmostEqual(canvas.yview()[0], 0, delta=0.01)
            self.assertGreater(button.winfo_width(), 40)
            self._screenshot("history-refreshed-without-resize")
        finally:
            app.search.delete(0, "end")
            app.current_record = None
            app._replace_text("")
            for record in records:
                app.history_store.delete(record["id"])
            app._refresh_history()

    def test_live_font_and_language_preserve_transcript_and_recognition_settings(self):
        app = self.app
        page = app._settings
        app.config_store.update(language="Chinese")
        app._replace_text("原始转写文本\nKeep this transcript unchanged.")
        editor, editor_font = app.editor, app.editor.cget("font")
        page.path_var.set("/draft/model/path")
        app._set_busy(True)
        app.events.put({"kind": "partial", "text": "正在识别的真实内容"})
        self.pump()
        app._open_settings()
        page.tabs.set("外观")
        page.font_size_var.set("24")
        page.ui_language_var.set("English")
        self.assertTrue(page._save_appearance())
        self.pump()
        self.assertEqual(app.ui_language, "en_US")
        self.assertEqual(page.tabs.get(), "Appearance")
        self.assertEqual(app.file_button.cget("text"), "Choose file")
        self.assertEqual(app.search.cget("placeholder_text"), "Search history…")
        self.assertEqual(editor_font.cget("size"), 28)
        self.assertIs(editor, app.editor)
        self.assertIs(editor_font, app.editor.cget("font"))
        self.assertEqual(app.partial_label.cget("text"), "正在识别的真实内容")
        self.assertEqual(app._text(), "原始转写文本\nKeep this transcript unchanged.")
        self.assertEqual(page.path_var.get(), "/draft/model/path")
        config = app.config_store.load()
        self.assertEqual((config["ui_language"], config["font_size"], config["language"]), ("en_US", 24, "Chinese"))
        self.assertTrue(page.save_current())
        app.events.put({"kind": "status", "text": "正在转写第 {index} 段 · {start:.1f}–{end:.1f} 秒", "values": {"index": 2, "start": 30.0, "end": 50.0}})
        self.pump()
        self.assertEqual(app.status.cget("text"), "Transcribing segment 2 · 30.0–50.0 seconds")
        app.geometry("900x620")
        self.pump()
        for section, name in (("模型", "Models"), ("目录", "Folders"), ("音频", "Audio"), ("外观", "Appearance")):
            page.tabs.set(name)
            self.pump()
            self._screenshot(f"large-font-en-{name}")
            for widget in (page.back_button, page._save_buttons[section]):
                self.assertTrue(widget.winfo_ismapped())
                self.assertLessEqual(widget.winfo_rootx() + widget.winfo_width(), app.winfo_rootx() + app.winfo_width())
        page.ui_language_var.set("简体中文")
        page.font_size_var.set("18")
        self.assertTrue(page._save_appearance())
        self.pump()
        self.assertEqual(page.tabs.get(), "外观")
        self.assertEqual(app.file_button.cget("text"), "选择文件")
        self.assertEqual(editor_font.cget("size"), 22)
        app.reset_layout()

    def test_sizes_scaling_and_settings_save_visibility(self):
        app = self.app
        for scale in (1.0, 1.25, 1.5):
            self.ctk.set_widget_scaling(scale)
            # CTk temporarily pins root min/max for 1000 ms during rescaling.
            self.pump(1.1)
            for width, height in ((900, 620), (1180, 780)):
                app.geometry(f"{width}x{height}")
                app._open_settings()
                self.pump()
                for name in ("模型", "目录", "音频", "外观"):
                    app._settings.tabs.set(name)
                    self.pump(0.2)
                    button = app._settings._save_buttons[name]
                    self.assertTrue(button.winfo_ismapped())
                    self.assertLessEqual(button.winfo_rooty() + button.winfo_height(), app.winfo_rooty() + app.winfo_height())
                    self.assertLessEqual(button.winfo_rootx() + button.winfo_width(), app.winfo_rootx() + app.winfo_width())
                app.show_transcription()
                self.pump()
                self.assertTrue(app.file_button.winfo_ismapped())
                self.assertGreater(app.editor.winfo_height(), 80, f"{width}x{height} at {scale}; actual {app.geometry()}")
                self._screenshot(f"layout-{width}-{height}-{scale}")
        self.ctk.set_widget_scaling(1.0)
        self.pump(1.1)
        app.reset_layout()
        self.pump(0.2)
        app.attributes("-zoomed", True)
        self.pump(0.6)
        self.assertTrue(app._is_maximized())
        app._save_layout()
        self.assertTrue(app.config_store.load()["window_layout"]["maximized"])
        app.attributes("-zoomed", False)
        self.pump(0.2)

    def test_extra_large_type_icons_and_hardware_navigation(self):
        app, page = self.app, self.app._settings
        app.geometry("900x620")
        app._open_settings()
        page.font_size_var.set("48")
        page.ui_language_var.set("English")
        self.assertTrue(page._save_appearance())
        self.pump()
        self.assertEqual(app.editor.cget("font").cget("size"), 52)
        for section, name in page._tab_names.items():
            page.tabs.set(name)
            self.pump()
            self.assertTrue(page._sections[section].winfo_ismapped())
            self.assertIsNotNone(page.tabs._buttons[name].cget("image"))
            widgets = [page.back_button]
            if section in page._save_buttons:
                widgets.append(page._save_buttons[section])
            for widget in widgets:
                self.assertTrue(widget.winfo_ismapped())
                self.assertLessEqual(widget.winfo_rootx() + widget.winfo_width(), app.winfo_rootx() + app.winfo_width())
                self.assertLessEqual(widget.winfo_rooty() + widget.winfo_height(), app.winfo_rooty() + app.winfo_height())
            self._screenshot(f"font48-{section}")
        self.assertTrue(page.save_current())  # This PC is read-only.
        self.assertEqual(page.system_panel.refresh_button.cget("state"), "normal")
        panel = page.system_panel
        self.assertEqual(len(panel._model_menu.cget("values")), len(panel.recommendations))
        visited = set()
        for _ in range(len(panel.recommendations)):
            visited.add(panel._selected_model_id)
            self.assertEqual(len(panel._recommendation_body.winfo_children()), 1)
            self.assertLess(page._sections["本机配置"].winfo_height(), 30_000)
            self.assertLess(panel.winfo_height(), 30_000)
            panel._step_recommendation(1)
            self.pump(0.05)
        self.assertEqual(visited, {model["id"] for model in panel.recommendations})
        self.assertIn("whisper-onnx-medium-int8", visited)
        self.assertIn("whisper-onnx-medium-fp32", visited)
        app.show_transcription()
        self.pump()
        self.assertGreaterEqual(app.editor.winfo_height(), 80)
        for button in (app.file_button, app.mic_button, app.save_button, app.export_button):
            self.assertIsNotNone(button.cget("image"))
            self.assertGreater(button.winfo_height(), 40)
        self._screenshot("font48-transcription")
        app.reset_layout()

    def test_hardware_and_model_guidance_show_budgets_without_loading_weights(self):
        from core.model_catalog import describe_model, recommend_models

        app, page = self.app, self.app._settings
        hardware = {"cpu_model": "Test CPU", "logical_cores": 8, "ram_total_gib": 32,
                    "ram_available_gib": 18, "system": "Ubuntu test", "gpus": [], "warnings": []}
        panel = page.system_panel
        panel._request += 1  # Ignore an earlier asynchronous detection result.
        panel.accept_result({"request": panel._request, "hardware": hardware, "recommendations": recommend_models(hardware)})
        page.model_info.model = describe_model("whisper", "/models/large-v3.pt")
        page.model_info._loading = False
        page.model_info.set_hardware(hardware)
        self.pump()
        self.assertEqual(page.model_info.model["cpu_ram_gib"], [10, 13])
        self.assertIsNone(app.controller.manager.model)
        self.assertTrue(any(item.get("top_recommendation") for item in panel.recommendations))
        app._open_settings()
        page.tabs.set("本机配置")
        self.pump()
        self._screenshot("hardware-overview")
        page._sections["本机配置"]._parent_canvas.yview_moveto(0.3)
        self.pump()
        self._screenshot("hardware-recommendations")
        page.tabs.set("模型")
        page._sections["模型"]._parent_canvas.yview_moveto(0.35)
        self.pump()
        self._screenshot("model-resource-budget")

    def test_work_directory_and_independent_data_selection(self):
        app = self.app
        work = self.root_path / "work"
        work.mkdir(exist_ok=True)
        app.config_store.update(work_dir=str(work))
        with patch("ui.app.filedialog.askopenfilename", return_value="") as choose:
            app._choose_file()
            self.assertEqual(choose.call_args.kwargs["initialdir"], str(work))
        app._replace_text("导出内容")
        with patch("ui.app.filedialog.asksaveasfilename", return_value="") as save:
            app._export()
            self.assertEqual(save.call_args.kwargs["initialdir"], str(work))
        current_path = app.config_store.path
        target = self.root_path / "other-library"
        app._settings.work_var.set(str(work))
        app._settings.data_var.set(str(target))
        app._settings._save_directories()
        self.assertEqual(app.config_store.path, current_path)
        self.assertEqual(app.bootstrap.load()["data_dir"], str(target))
        self.assertFalse((target / "history.json").exists())
        from storage import resolve_data_directory
        self.assertEqual(resolve_data_directory(bootstrap=app.bootstrap).path, target)

    def test_sherpa_scan_precision_paths_localization_and_cpu_resource_cards(self):
        app, page = self.app, self.app._settings
        model = self.root_path / "onnx-scan" / "sherpa-onnx-whisper-medium"
        model.mkdir(parents=True)
        for suffix in (".onnx", ".int8.onnx"):
            for part in ("encoder", "decoder"):
                (model / f"medium-{part}{suffix}").write_bytes(b"fake fixture; never loaded")
        (model / "medium-tokens.txt").write_text("token 0\n", encoding="utf-8")

        def wait_for(predicate):
            deadline = time.monotonic() + 5
            while not predicate() and time.monotonic() < deadline:
                self.pump(0.05)
            self.assertTrue(predicate(), "Expected settings background result did not arrive")

        def precision_is(value):
            return not page.model_info._loading and (page.model_info.model or {}).get("precision") == value

        with patch.object(app.controller.manager, "load", side_effect=AssertionError("GUI fixtures must never load weights")):
            app._open_settings()
            page.tabs.set("模型")
            page._change_type("OpenAI Whisper")
            page.type_var.set("OpenAI Whisper")
            page.device_var.set("NVIDIA CUDA")
            page.root_var.set(str(model.parent))
            page._scan()
            wait_for(lambda: any(item["path"] == str(model) for item in page._models))
            selected = next(item for item in page._models if item["path"] == str(model))
            self.assertEqual(selected["type"], "whisper_onnx")
            self.assertTrue(selected["ready"])
            page._select_model(selected)
            self.assertEqual(page.type_var.get(), "Sherpa-ONNX Whisper")
            self.assertEqual(page.device_var.get(), "CPU")
            self.assertTrue(page._save_model())
            wait_for(lambda: precision_is("int8"))
            self.assertEqual(app.config_store.load()["model_path"], str(model))

            fp32 = model / "medium-encoder.onnx"
            with patch("ui.settings.filedialog.askopenfilename", return_value=str(fp32)):
                page._browse_model_file()
            self.assertTrue(page._save_model())
            wait_for(lambda: precision_is("fp32"))
            page._change_type("Qwen3-ASR")
            page.type_var.set("Qwen3-ASR")
            page._change_type("Sherpa-ONNX Whisper")
            page.type_var.set("Sherpa-ONNX Whisper")
            self.assertEqual(page.path_var.get(), str(fp32))

            page.ui_language_var.set("English")
            self.assertTrue(page._save_appearance())
            self.pump()
            self.assertEqual(page.tabs.get(), "Models")
            self.assertEqual(page.path_var.get(), str(fp32))
            self.assertEqual(app.config_store.load()["model_paths"]["whisper_onnx"], str(fp32))
            page.model_info.set_hardware({"ram_available_gib": 32, "gpus": [{"name": "Unused CUDA GPU", "cuda": True, "free_vram_gib": 24}]})
            self.pump()
            self.assertEqual(page.model_info.fit["device"], "cpu")
            self.assertEqual(page.model_info.model["supported_devices"], ["cpu"])
            pending = [page.model_info.body]
            labels = []
            while pending:
                widget = pending.pop()
                pending.extend(widget.winfo_children())
                if isinstance(widget, self.ctk.CTkLabel):
                    labels.append(widget.cget("text"))
            self.assertEqual(labels.count("Not applicable (CPU only)"), 2)
            self.assertIn("CPU", labels)
            self.assertNotIn("CUDA GPU", labels)
            self.assertIsNone(app.controller.manager.model)
            self._screenshot("sherpa-fp32-settings-en")

    def test_task_monitor_file_events_and_timestamped_log_are_visible(self):
        app = self.app
        app.geometry("1180x780")
        app.reset_layout()
        app._replace_text("保留正文")
        app.activity_log.clear()
        stamp = "2026-10-01T04:05:06+00:00"
        for event in (
            {"kind": "task_started", "mode": "file", "source": "/tmp/会议.wav",
             "model": "Whisper small", "target_seconds": 30, "max_seconds": 60},
            {"kind": "media_info", "metadata": {"path": "/tmp/会议.wav", "name": "会议.wav",
             "size_bytes": 409600, "duration": 90, "format": "wav", "codec": "pcm_s16le",
             "sample_rate": 48000, "channels": 2}},
            {"kind": "chunk", "chunk": {"id": 1, "start": 0, "end": 30, "status": "done", "reason": "pause"}},
            {"kind": "chunk", "chunk": {"id": 2, "start": 30, "end": 60, "status": "processing", "reason": "limit"}},
        ):
            app.events.put({**event, "timestamp": stamp})
        self.pump(0.35)
        monitor = app.task_monitor
        self.assertIn("0:30", monitor.hover.cget("text"))
        self.assertIn("/tmp/会议.wav", monitor.detail.cget("text"))
        self.assertIn("400.0 KB", monitor.detail.cget("text"))
        self.assertIn("48000 Hz", monitor.detail.cget("text"))
        self.assertTrue(monitor.canvas.find_all())
        self.assertEqual(monitor.state.current[0], 2)
        self.assertIn("处理中", app.activity_log.textbox.get("1.0", "end"))
        self.assertRegex(app.activity_log.textbox.get("1.0", "end"), r"\[\d{2}:\d{2}:\d{2}\]")
        self.assertGreater(app.activity_log.textbox.winfo_height(), 20)
        for widget in (monitor.canvas, app.activity_log.textbox):
            self.assertTrue(widget.winfo_ismapped())
            self.assertLessEqual(widget.winfo_rooty() + widget.winfo_height(), app.winfo_rooty() + app.winfo_height())
        self._screenshot("file-progress-and-log")
        app._open_settings()
        app.apply_preferences("light", "purple", 18, "en_US")
        app.show_transcription()
        self.pump()
        self.assertEqual(monitor.state.current[0], 2)
        self.assertEqual(app._text(), "保留正文")
        self.assertEqual(monitor.title.cget("text"), "Task monitor")
        with patch("ui.app.messagebox.showerror"):
            app.events.put({"kind": "error", "text": "录音失败", "details": "48000 Hz: test failure", "timestamp": stamp})
            self.pump()
        self.assertIn("48000 Hz: test failure", app.activity_log.textbox.get("1.0", "end"))

    def _screenshot(self, name):
        directory = os.environ.get("MYSOUND_SCREENSHOTS")
        if not directory:
            return
        from desktop_capture import capture_window
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        app = self.app
        capture_window(app).save(path / f"{name}.png")


if __name__ == "__main__":
    unittest.main()
