"""Settings save/selection behavior without creating a Tk window."""

from pathlib import Path
from queue import Queue
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from storage import ConfigStore
from storage.bootstrap import BootstrapStore, DataDirectorySelection
from ui.i18n import Translator

try:
    from ui.settings import SettingsPage, validate_audio_values, validate_model_path
    from ui.system_panel import ModelInfoPanel, SystemPanel, format_gib, format_budget, localized_field, model_budget
except ImportError:
    SettingsPage = None


class Value:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


@unittest.skipIf(SettingsPage is None, "CustomTkinter/Tk is not installed")
class SettingsTests(unittest.TestCase):
    def make_page(self, directory):
        page = SettingsPage.__new__(SettingsPage)
        translator = Translator("zh_CN")
        page.app = SimpleNamespace(
            config_store=ConfigStore(Path(directory) / "data"),
            bootstrap=BootstrapStore(Path(directory) / "bootstrap"),
            data_selection=DataDirectorySelection(Path(directory) / "data", "default", False),
            _job_ui_active=False, status=MagicMock(),
            apply_preferences=MagicMock(), _update_model_label=MagicMock(),
            tr=translator.tr, translate_widget=translator.bind, translator=translator,
            tr_error=lambda exc: getattr(exc, "message_en", str(exc)) if translator.language == "en_US" else str(exc),
            ui_language="zh_CN",
        )
        page.config = page.app.config_store.load()
        page._notes = {name: MagicMock() for name in ("模型", "目录", "音频", "外观")}
        page._note_sources = {}
        page._choice_maps = {}
        page._choice_specs = []
        page._tab_names = {name: name for name in ("模型", "目录", "音频", "外观")}
        page._save_buttons = {}
        page._model_row_labels = []
        page._buttons = []
        page._inputs = []
        page._model_detail_source = ("", {})
        page._device_records = []
        page._devices_loaded = False
        page._data_source = "系统默认目录"
        page.data_info = MagicMock()
        page.busy_note = MagicMock()
        page.model_detail = MagicMock()
        page.font_size_var = Value("14")
        page.device_var = Value("自动选择")
        page.ui_language_var = Value("简体中文")
        page._busy = False
        page._data_overridden = False
        return page

    def test_unconfigured_model_does_not_block_audio_or_appearance(self):
        with TemporaryDirectory() as directory:
            page = self.make_page(directory)
            page.target_var, page.max_var, page.block_var = Value("20"), Value("40"), Value("2")
            page.mic_var, page._mic_devices = Value("系统默认"), {"系统默认": None}
            self.assertTrue(page._save_audio())
            page.appearance_var, page.color_var = Value("浅色"), Value("橙色")
            page._busy = page.app._job_ui_active = True
            self.assertFalse(page._save_audio())
            self.assertTrue(page._save_appearance())
            saved = page.app.config_store.load()
            self.assertEqual(saved["model_path"], "")
            self.assertEqual(saved["vad_target_seconds"], 20)
            self.assertEqual(saved["color_theme"], "orange")
            page.app.apply_preferences.assert_called_once_with("light", "orange", 14, "zh_CN")

    def test_type_switches_preserve_each_draft_path(self):
        with TemporaryDirectory() as directory:
            page = self.make_page(directory)
            page._active_type = "qwen3"
            page._model_paths = {"qwen3": "/qwen/old", "whisper": "/whisper/large.pt"}
            page.path_var = Value("/qwen/edited")
            page._change_type("OpenAI Whisper")
            self.assertEqual(page.path_var.get(), "/whisper/large.pt")
            page.path_var.set("/whisper/edited.pt")
            page._change_type("Qwen3-ASR")
            self.assertEqual(page.path_var.get(), "/qwen/edited")
            self.assertEqual(page._model_paths["whisper"], "/whisper/edited.pt")

    def test_scan_results_do_not_change_selection(self):
        with TemporaryDirectory() as directory:
            page = self.make_page(directory)
            page._request_ids = {"scan": 1}
            page._render_models = MagicMock()
            page._select_model = MagicMock()
            page.path_var = Value("/selected/model")
            page.accept_result({"operation": "scan", "request_id": 1, "result": [{"name": "Different model", "path": "/new/model", "type": "whisper"}]})
            page._render_models.assert_called_once()
            page._select_model.assert_not_called()
            self.assertEqual(page.path_var.get(), "/selected/model")

    def test_invalid_data_directory_does_not_partially_save_work_directory(self):
        with TemporaryDirectory() as directory:
            page = self.make_page(directory)
            invalid = Path(directory) / "file"
            invalid.write_text("not a directory")
            page.work_var, page.data_var = Value(directory), Value(str(invalid))
            self.assertFalse(page._save_directories())
            self.assertEqual(page.app.config_store.load()["work_dir"], "")
            self.assertEqual(page.app.bootstrap.load(), {})

    def test_directory_selection_only_changes_next_launch_pointer(self):
        with TemporaryDirectory() as directory:
            page = self.make_page(directory)
            new_data = Path(directory) / "new-data"
            page.work_var, page.data_var = Value(directory), Value(str(new_data))
            self.assertTrue(page._save_directories())
            self.assertEqual(page.app.bootstrap.load()["data_dir"], str(new_data))
            self.assertEqual(page.app.config_store.path.parent, Path(directory) / "data")
            self.assertFalse((new_data / "config.json").exists())

    def test_launch_override_can_save_work_directory_without_touching_pointer(self):
        with TemporaryDirectory() as directory:
            page = self.make_page(directory)
            page._data_overridden = True
            page.work_var, page.data_var = Value(directory), Value("")
            self.assertTrue(page._save_directories())
            self.assertEqual(page.app.config_store.load()["work_dir"], directory)
            self.assertEqual(page.app.bootstrap.load(), {})

    def test_invalid_audio_is_rejected(self):
        for values in (("nan", "60", "2"), ("30", "20", "2"), ("30", "60", "0")):
            with self.subTest(values=values), self.assertRaises(ValueError):
                validate_audio_values(*values)

    def test_label_wrapping_defers_and_converts_physical_width_without_recursion(self):
        with TemporaryDirectory() as directory:
            page = self.make_page(directory)
            page._widget = lambda widget, role: widget
            parent, label = MagicMock(), MagicMock()
            label.winfo_exists.return_value = True
            label.winfo_width.return_value = 300
            label._get_widget_scaling.return_value = 1.5
            current_wrap = [300]
            callbacks = []
            icon = MagicMock()
            icon.cget.return_value = (22, 22)
            label.cget.side_effect = lambda key: icon if key == "image" else current_wrap[0]
            label.after_idle.side_effect = lambda callback: callbacks.append(callback) or len(callbacks)
            def configure(**options):
                if "wraplength" not in options:
                    return
                current_wrap[0] = options["wraplength"]
                # Tk can emit another parent Configure when wrapping changes
                # the label height; it must be deferred and converge.
                parent.bind.call_args.args[1]()
            label.configure.side_effect = configure
            with patch("ui.settings.ctk.CTkLabel", return_value=label):
                page._label(parent, "Some wrapping text")
            self.assertEqual(label.bind.call_args.args[0], "<Destroy>")
            self.assertEqual(len(callbacks), 1)
            callbacks.pop(0)()
            callbacks.pop(0)()
            self.assertEqual(callbacks, [])
            wrap_calls = [call.kwargs for call in label.configure.call_args_list if "wraplength" in call.kwargs]
            self.assertEqual(wrap_calls, [{"wraplength": 160}])
            parent.bind.call_args.args[1]()
            label.bind.call_args.args[1]()
            label.after_cancel.assert_called_once()

    def test_appearance_saves_font_and_interface_language_without_changing_speech_language(self):
        with TemporaryDirectory() as directory:
            page = self.make_page(directory)
            page.app.config_store.update(language="Chinese")
            page.appearance_var, page.color_var = Value("浅色"), Value("紫色")
            page.font_size_var, page.ui_language_var = Value("24"), Value("English")
            self.assertTrue(page._save_appearance())
            saved = page.app.config_store.load()
            self.assertEqual((saved["font_size"], saved["ui_language"], saved["language"]), (24, "en_US", "Chinese"))
            page.app.apply_preferences.assert_called_once_with("light", "purple", 24, "en_US")
            page.font_size_var.set("13")
            self.assertFalse(page._save_appearance())
            self.assertEqual(page.app.config_store.load()["font_size"], 24)

    def test_theme_preset_and_largest_font_persist_without_loading_model(self):
        with TemporaryDirectory() as directory:
            page = self.make_page(directory)
            page.appearance_var, page.color_var = Value("深色"), Value("蓝色")
            page._select_theme("green")
            page.font_size_var.set("48")
            self.assertTrue(page._save_appearance())
            saved = page.app.config_store.load()
            self.assertEqual((saved["font_size"], saved["color_theme"]), (48, "green"))
            page.app.apply_preferences.assert_called_once_with("dark", "green", 48, "zh_CN")
            page.font_size_var.set("56")
            self.assertFalse(page._save_appearance())
            self.assertEqual(page.app.config_store.load()["font_size"], 48)

    def test_hardware_section_is_read_only_even_during_transcription(self):
        with TemporaryDirectory() as directory:
            page = self.make_page(directory)
            page._tab_names["本机配置"] = "This PC"
            page.tabs = SimpleNamespace(get=lambda: "This PC")
            page._busy = page.app._job_ui_active = True
            before = page.app.config_store.load()
            with patch.object(page.app.config_store, "update") as update:
                self.assertTrue(page.save_current())
                update.assert_not_called()
            self.assertEqual(page.app.config_store.load(), before)

    def test_model_file_browser_keeps_selected_qwen_type_and_normalizes_path(self):
        with TemporaryDirectory() as directory:
            page = self.make_page(directory)
            selected_file = Path(directory) / "config.json"
            selected_file.write_text("{}")
            page.type_var, page.path_var, page.root_var = Value("Qwen3-ASR"), Value(""), Value(directory)
            page.winfo_toplevel = MagicMock()
            with patch("ui.settings.filedialog.askopenfilename", return_value=str(selected_file)) as picker, \
                 patch("ui.settings.resolve_model_path", return_value=Path(directory)) as resolver:
                page._browse_model_file()
            self.assertEqual(page.type_var.get(), "Qwen3-ASR")
            self.assertEqual(page.path_var.get(), directory)
            resolver.assert_called_once_with(str(selected_file), "qwen3")
            patterns = picker.call_args.kwargs["filetypes"][0][1]
            self.assertIn("*.safetensors", patterns)
            self.assertNotIn("*.pt", patterns)

    def test_onnx_file_browser_preserves_precision_and_scan_selection_saves_independent_type(self):
        with TemporaryDirectory() as directory:
            page = self.make_page(directory)
            onnx_file = Path(directory) / "medium-encoder.onnx"
            onnx_file.touch()
            page._active_type = "whisper"
            page._model_paths = {"whisper": "/models/medium.pt", "whisper_onnx": ""}
            page.type_var, page.path_var, page.root_var = Value("OpenAI Whisper"), Value("/models/medium.pt"), Value(directory)
            page.device_var, page.language_var = Value("NVIDIA CUDA"), Value("自动识别")
            page.winfo_toplevel = MagicMock()
            with patch("ui.settings.resolve_model_path", return_value=onnx_file):
                page._select_model({"type": "whisper_onnx", "path": str(onnx_file), "name": "Whisper medium FP32"})
                self.assertEqual(page.type_var.get(), "Sherpa-ONNX Whisper")
                self.assertEqual(page.device_var.get(), "CPU")
                self.assertTrue(page._save_model())
                with patch("ui.settings.filedialog.askopenfilename", return_value=str(onnx_file)) as picker:
                    page._browse_model_file()
            self.assertEqual(picker.call_args.kwargs["filetypes"][0][1], "*.onnx")
            saved = page.app.config_store.load()
            self.assertEqual((saved["model_type"], saved["model_path"], saved["device"]), ("whisper_onnx", str(onnx_file), "cpu"))
            self.assertEqual(saved["model_paths"]["whisper"], "/models/medium.pt")

    def test_model_folder_browser_requires_existing_folder_and_shows_candidates_on_failure(self):
        with TemporaryDirectory() as directory:
            page = self.make_page(directory)
            page.type_var, page.path_var, page.root_var = Value("Confucius4-R2T2"), Value(""), Value(directory)
            page.winfo_toplevel = MagicMock()
            page._background = MagicMock()
            with patch("ui.settings.filedialog.askdirectory", return_value=directory) as picker, \
                 patch("ui.settings.resolve_model_path", side_effect=ValueError("请选择具体模型目录")):
                page._browse_model_folder()
            self.assertTrue(picker.call_args.kwargs["mustexist"])
            self.assertEqual(page.path_var.get(), directory)
            self.assertEqual(page.type_var.get(), "Confucius4-R2T2")
            self.assertEqual(page._background.call_args.args[0], "scan")
            self.assertIn("请选择具体模型目录", page.model_detail.configure.call_args.kwargs["text"])

    def test_retranslation_keeps_selected_tab_and_unsaved_choices_and_paths(self):
        class Tabs:
            def __init__(self):
                self.current = "外观"
                self.names = ["模型", "目录", "音频", "外观"]

            def get(self):
                return self.current

            def rename(self, old, new):
                self.names[self.names.index(old)] = new

            def set(self, name):
                self.current = name

        with TemporaryDirectory() as directory:
            page = self.make_page(directory)
            page.tabs = Tabs()
            page.path_var = Value("/unsaved/model")
            page.appearance_var, page.color_var = Value("浅色"), Value("橙色")
            page._choice_specs = [(MagicMock(), page.appearance_var, {"深色": "dark", "浅色": "light"}),
                                  (MagicMock(), page.color_var, {"蓝色": "blue", "橙色": "orange"})]
            page._choice_maps = {id(var): dict(values) for _, var, values in page._choice_specs}
            page._mic_devices = {"系统默认": None, "设备 3": 3}
            page.mic_var, page.mic_menu = Value("设备 3"), MagicMock()
            page.app.translator.set_language("en_US")
            page.app.ui_language = "en_US"
            page.retranslate()
            self.assertEqual(page.tabs.get(), "Appearance")
            self.assertEqual(page.appearance_var.get(), "Light")
            self.assertEqual(page.color_var.get(), "Orange")
            self.assertEqual(page.path_var.get(), "/unsaved/model")
            self.assertEqual(page._mic_devices[page.mic_var.get()], 3)
            page._save_appearance = MagicMock(return_value=True)
            self.assertTrue(page.save_current())
            page._save_appearance.assert_called_once()
            page.app.translator.set_language("zh_CN")
            page.app.ui_language = "zh_CN"
            page.retranslate()
            self.assertEqual(page.tabs.get(), "外观")
            self.assertEqual(page.color_var.get(), "橙色")
            self.assertEqual(page.path_var.get(), "/unsaved/model")


@unittest.skipIf(SettingsPage is None, "CustomTkinter/Tk is not installed")
class SystemPanelTests(unittest.TestCase):
    def panel(self, cls=SystemPanel):
        panel = cls.__new__(cls)
        translator = Translator("en_US")
        panel.app = SimpleNamespace(events=Queue(), ui_language="en_US", tr=translator.tr)
        panel._request = 2
        panel._loading = True
        panel._error = None
        panel.hardware = {"cpu_model": "Previous CPU"}
        panel.recommendations = []
        panel.model = None
        panel.fit = None
        panel.refresh_button = MagicMock()
        panel.on_detect = MagicMock()
        panel.retranslate = MagicMock()
        return panel

    def test_resource_formatting_keeps_unknown_distinct_from_zero(self):
        tr = Translator("en_US").tr
        for value in (None, float("nan"), float("inf"), -1, "bad"):
            self.assertEqual(format_gib(value, tr), "Unknown")
        self.assertEqual(format_gib(0, tr), "0.0 GiB")
        self.assertEqual(format_budget([2.5, 4], tr), "2.5–4 GiB")
        self.assertEqual(format_budget([None, 4], tr), "Unknown")
        cpu_model = {"supported_devices": ["cpu"], "cpu_ram_gib": [3, 6], "gpu_vram_gib": None}
        self.assertEqual(model_budget(cpu_model, "gpu_vram_gib", tr), "Not applicable (CPU only)")
        self.assertEqual(model_budget(cpu_model, "cpu_ram_gib", tr), "3–6 GiB")
        self.assertEqual(model_budget({"gpu_vram_gib": None}, "gpu_vram_gib", tr), "Unknown")

    def test_hardware_result_ignores_stale_requests_and_forwards_current_snapshot(self):
        panel = self.panel()
        panel.accept_result({"request": 1, "hardware": {"cpu_model": "stale"}, "recommendations": []})
        self.assertEqual(panel.hardware["cpu_model"], "Previous CPU")
        panel.on_detect.assert_not_called()
        current = {"cpu_model": "Current CPU", "ram_available_gib": 12}
        panel.accept_result({"request": 2, "hardware": current, "recommendations": [{"id": "whisper-base"}]})
        self.assertIs(panel.hardware, current)
        panel.on_detect.assert_called_once_with(current)
        self.assertFalse(panel._loading)
        panel.refresh_button.configure.assert_called_once_with(state="normal")

    def test_detection_worker_only_posts_result_and_error_is_recoverable(self):
        panel = self.panel()
        hardware = {"cpu_model": "Fake CPU", "ram_available_gib": 8}
        with patch("ui.system_panel.threading.Thread") as thread, \
             patch("core.hardware_info.detect_hardware", return_value=hardware), \
             patch("core.model_catalog.recommend_models", return_value=[]) as recommend:
            panel.refresh()
            # Dispatching a job must not run expensive detection on the Tk thread.
            recommend.assert_not_called()
            thread.call_args.kwargs["target"]()
        event = panel.app.events.get_nowait()
        self.assertEqual(event["hardware"], hardware)
        self.assertEqual(panel.hardware["cpu_model"], "Previous CPU")
        panel.accept_result({"request": panel._request, "error": RuntimeError("probe failed")})
        self.assertFalse(panel._loading)
        self.assertEqual(panel.hardware["cpu_model"], "Previous CPU")
        panel.accept_result(event)
        self.assertIsNone(panel._error)
        self.assertEqual(panel.hardware, hardware)

    def test_model_metadata_stale_result_cannot_replace_current_path(self):
        panel = self.panel(ModelInfoPanel)
        panel.model = {"id": "current-model"}
        panel.accept_result({"request": 1, "model": {"id": "stale-model"}})
        self.assertEqual(panel.model["id"], "current-model")
        assessed = {"fit": "unknown", "reason_en": "Model size is unrecognized"}
        current = {"id": "unknown-model", "cpu_ram_gib": None}
        with patch("core.model_catalog.assess_model", return_value=assessed) as assess:
            panel.accept_result({"request": 2, "model": current})
        assess.assert_called_once_with(current, panel.hardware)
        self.assertIs(panel.fit, assessed)
        self.assertEqual(localized_field(assessed, "reason", panel.app), "Model size is unrecognized")

    def test_clearing_model_path_discards_pending_result_without_reading_metadata(self):
        panel = self.panel(ModelInfoPanel)
        panel.model = {"id": "previous-model"}
        with patch("ui.system_panel.threading.Thread") as thread:
            panel.refresh("qwen3", "  ")
            thread.assert_not_called()
        self.assertIsNone(panel.model)
        self.assertFalse(panel._loading)
        panel.accept_result({"request": 2, "model": {"id": "stale-model"}})
        self.assertIsNone(panel.model)
        panel.retranslate.assert_called_once()

    def test_theme_changes_reuse_hardware_widgets_but_language_changes_rebuild(self):
        panel = self.panel()
        panel._loading = False
        panel.hardware = None
        panel.body = MagicMock()
        panel._clear = MagicMock()
        panel._text = MagicMock()
        SystemPanel.retranslate(panel)
        SystemPanel.retranslate(panel)
        panel._clear.assert_called_once()
        panel.app.ui_language = "zh_CN"
        SystemPanel.retranslate(panel)
        self.assertEqual(panel._clear.call_count, 2)

    def test_recommendation_navigation_keeps_every_variant_accessible(self):
        panel = self.panel()
        panel.recommendations = [{"id": "whisper-small", "name": "Whisper small"},
                                 {"id": "whisper-onnx-medium-int8", "name": "Sherpa INT8"},
                                 {"id": "whisper-onnx-medium-fp32", "name": "Sherpa FP32"}]
        panel._selected_model_id = "whisper-small"
        panel._render_selected_recommendation = MagicMock()
        visited = set()
        for _ in panel.recommendations:
            visited.add(panel._selected_model_id)
            panel._step_recommendation(1)
        self.assertEqual(visited, {model["id"] for model in panel.recommendations})
        self.assertEqual(panel._selected_model_id, "whisper-small")
        panel._choose_recommendation("Sherpa FP32")
        self.assertEqual(panel._selected_model_id, "whisper-onnx-medium-fp32")


if __name__ == "__main__":
    unittest.main()
