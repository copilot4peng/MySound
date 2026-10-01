import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from storage import BootstrapStore, ConfigStore, HistoryStore, resolve_data_directory, validate_data_directory


class BootstrapTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.bootstrap = BootstrapStore(self.root / "bootstrap")
        environment = patch.dict(os.environ, {
            "MYSOUND_DATA_DIR": "",
            "XDG_CONFIG_HOME": str(self.root / "xdg-config"),
            "XDG_DATA_HOME": str(self.root / "xdg-data"),
        })
        environment.start()
        self.addCleanup(environment.stop)

    def test_default_location_preserves_old_application_data(self):
        selection = resolve_data_directory(bootstrap=self.bootstrap)
        self.assertEqual(selection.path, self.root / "xdg-data" / "mysound")
        self.assertEqual(selection.source, "default")
        self.assertFalse(selection.overridden)
        self.assertEqual(BootstrapStore().path, self.root / "xdg-config" / "mysound" / "bootstrap.json")

    def test_selection_only_saves_pointer_and_creates_directory(self):
        destination = self.root / "new-data"
        selected = self.bootstrap.select_data_dir(destination)
        self.assertEqual(selected, destination)
        self.assertEqual(self.bootstrap.load(), {"data_dir": str(destination)})
        self.assertEqual(json.loads(self.bootstrap.path.read_text()), {"data_dir": str(destination)})
        self.assertEqual(list(destination.iterdir()), [])
        next_launch = resolve_data_directory(bootstrap=BootstrapStore(self.root / "bootstrap"))
        self.assertEqual(next_launch.path, destination)
        self.assertEqual(next_launch.source, "bootstrap")
        self.assertFalse(next_launch.overridden)

    def test_cli_and_environment_override_saved_pointer(self):
        saved = self.root / "saved"
        self.bootstrap.select_data_dir(saved)
        with patch.dict(os.environ, {"MYSOUND_DATA_DIR": str(self.root / "environment")}):
            environment = resolve_data_directory(bootstrap=self.bootstrap)
            explicit = resolve_data_directory(self.root / "cli", self.bootstrap)
        self.assertEqual(environment.path, self.root / "environment")
        self.assertEqual(environment.source, "env")
        self.assertTrue(environment.overridden)
        self.assertEqual(explicit.path, self.root / "cli")
        self.assertEqual(explicit.source, "cli")
        self.assertTrue(explicit.overridden)
        self.assertEqual(self.bootstrap.load()["data_dir"], str(saved))

    def test_switching_keeps_existing_target_and_source_data_independent(self):
        source, target = self.root / "source", self.root / "target"
        old_config, old_history = ConfigStore(source), HistoryStore(source)
        old_config.update(model_path="/old-model")
        old_record = old_history.add("old.wav", "qwen3", "原目录")
        ConfigStore(target).update(model_path="/target-model")
        target_record = HistoryStore(target).add("new.wav", "qwen3", "目标目录")
        self.bootstrap.select_data_dir(target)
        selected = resolve_data_directory(bootstrap=self.bootstrap)
        self.assertEqual(ConfigStore(selected.path).load()["model_path"], "/target-model")
        self.assertEqual(HistoryStore(selected.path).list()[0]["id"], target_record["id"])
        self.assertEqual(old_history.list()[0]["id"], old_record["id"])
        self.assertEqual(old_config.load()["model_path"], "/old-model")

    def test_invalid_directory_does_not_replace_previous_pointer(self):
        previous = self.bootstrap.select_data_dir(self.root / "previous")
        occupied = self.root / "file.txt"
        occupied.write_text("keep me", encoding="utf-8")
        with self.assertRaises((OSError, ValueError)):
            self.bootstrap.select_data_dir(occupied)
        with self.assertRaises(ValueError):
            self.bootstrap.select_data_dir("")
        self.assertEqual(self.bootstrap.load()["data_dir"], str(previous))
        self.assertEqual(occupied.read_text(), "keep me")

    def test_unwritable_directory_does_not_replace_pointer(self):
        previous = self.bootstrap.select_data_dir(self.root / "previous")
        with patch("storage.bootstrap.os.access", return_value=False):
            with self.assertRaises(PermissionError):
                self.bootstrap.select_data_dir(self.root / "denied")
        self.assertEqual(self.bootstrap.load()["data_dir"], str(previous))

    def test_validation_does_not_change_selected_directory(self):
        previous = self.bootstrap.select_data_dir(self.root / "previous")
        candidate = self.root / "candidate"
        self.assertEqual(validate_data_directory(candidate), candidate)
        self.assertEqual(self.bootstrap.load()["data_dir"], str(previous))
        self.assertEqual(list(candidate.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
