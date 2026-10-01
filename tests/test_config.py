import json
from pathlib import Path
import tempfile
import unittest

from storage import ConfigStore, DEFAULT_CONFIG


class ConfigTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.store = ConfigStore(self.directory)

    def test_legacy_selected_model_path_is_migrated_without_rewriting_on_read(self):
        legacy = {"model_type": "whisper", "model_path": "/models/whisper.pt", "device": "cpu"}
        self.store.path.write_text(json.dumps(legacy), encoding="utf-8")
        loaded = self.store.load()
        self.assertEqual(loaded["model_paths"]["whisper"], "/models/whisper.pt")
        self.assertEqual(loaded["model_paths"]["qwen3"], "")
        self.assertEqual(loaded["model_paths"]["whisper_onnx"], "")
        self.assertEqual(loaded["appearance_mode"], "dark")
        self.assertEqual(loaded["window_layout"]["editor_ratio"], 0.40)
        self.assertEqual(json.loads(self.store.path.read_text()), legacy)

    def test_changing_model_restores_each_saved_path(self):
        self.store.update(model_path="/models/qwen")
        self.store.update(model_type="whisper", model_path="/models/whisper.pt")
        selected = self.store.update(model_type="qwen3")
        self.assertEqual(selected["model_path"], "/models/qwen")
        self.assertEqual(selected["model_paths"]["whisper"], "/models/whisper.pt")
        self.assertEqual(self.store.load()["model_path"], "/models/qwen")

    def test_onnx_precision_file_path_is_saved_independently_from_pytorch_whisper(self):
        self.store.update(model_type="whisper", model_path="/models/medium.pt")
        onnx_file = "/models/sherpa-medium/medium-encoder.onnx"
        self.store.update(model_type="whisper_onnx", model_path=onnx_file, device="cpu")
        self.assertEqual(self.store.update(model_type="whisper")["model_path"], "/models/medium.pt")
        restored = self.store.update(model_type="whisper_onnx")
        self.assertEqual(restored["model_path"], onnx_file)
        self.assertEqual(ConfigStore(self.directory).load()["model_paths"]["whisper_onnx"], onnx_file)

    def test_new_mapping_updates_active_legacy_field(self):
        self.store.update(model_paths={"qwen3": "/models/qwen", "confucius": "/models/confucius"})
        self.assertEqual(self.store.load()["model_path"], "/models/qwen")
        changed = self.store.update(model_type="confucius")
        self.assertEqual(changed["model_path"], "/models/confucius")
        cleared = self.store.update(model_paths={"confucius": ""})
        self.assertEqual(cleared["model_path"], "")
        self.assertEqual(self.store.load()["model_path"], "")

    def test_mapping_is_authoritative_over_legacy_path_when_loading(self):
        data = {"model_type": "qwen3", "model_path": "/old/path", "model_paths": {"qwen3": ""}}
        self.store.path.write_text(json.dumps(data), encoding="utf-8")
        self.assertEqual(self.store.load()["model_path"], "")

    def test_partial_nested_updates_keep_other_values_and_defaults(self):
        self.store.update(window_layout={"width": 1350, "sidebar_ratio": 0.3})
        updated = self.store.update(window_layout={"height": 900}, appearance_mode="light", color_theme="purple", work_dir="/media/work")
        self.assertEqual(updated["window_layout"], {
            "width": 1350, "height": 900, "maximized": False,
            "sidebar_ratio": 0.3, "editor_ratio": 0.40,
        })
        self.assertEqual(updated["model_paths"], DEFAULT_CONFIG["model_paths"])
        self.assertEqual(updated["work_dir"], "/media/work")
        self.assertEqual(ConfigStore(self.directory).load(), updated)

    def test_returned_nested_defaults_do_not_mutate_other_loads(self):
        first = self.store.load()
        first["window_layout"]["width"] = 999
        first["model_paths"]["whisper"] = "/unexpected"
        again = self.store.load()
        self.assertEqual(again["window_layout"]["width"], 1180)
        self.assertEqual(again["model_paths"]["whisper"], "")

    def test_interface_language_and_font_size_persist_independently_from_asr_language(self):
        updated = self.store.update(ui_language="en_US", font_size=20, language="Chinese")
        self.assertEqual(updated["ui_language"], "en_US")
        self.assertEqual(updated["font_size"], 20)
        loaded = ConfigStore(self.directory).load()
        self.assertEqual(loaded["language"], "Chinese")
        self.assertEqual(loaded["ui_language"], "en_US")
        self.assertEqual(loaded["font_size"], 20)

    def test_old_or_invalid_presentation_values_are_normalized(self):
        self.assertEqual(self.store.load()["ui_language"], "zh_CN")
        self.assertEqual(self.store.load()["font_size"], 14)
        self.store.path.write_text(json.dumps({"ui_language": "unsupported", "font_size": 100}), encoding="utf-8")
        loaded = self.store.load()
        self.assertEqual(loaded["ui_language"], "zh_CN")
        self.assertEqual(loaded["font_size"], 48)
        updated = self.store.update(ui_language="unsupported", font_size="invalid")
        self.assertEqual(updated["ui_language"], "zh_CN")
        self.assertEqual(updated["font_size"], 14)


if __name__ == "__main__":
    unittest.main()
