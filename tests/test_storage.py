import concurrent.futures
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from storage import ConfigStore, HistoryStore, StorageError, default_data_dir


class StorageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)

    def test_config_defaults_merge_and_round_trip(self):
        config = ConfigStore(self.directory)
        self.assertEqual(config.load()["model_type"], "qwen3")
        config.update(model_path="/models/语音", device="cpu")
        config.update(language="Chinese")
        loaded = ConfigStore(self.directory).load()
        self.assertEqual(loaded["model_path"], "/models/语音")
        self.assertEqual(loaded["device"], "cpu")
        self.assertEqual(loaded["language"], "Chinese")
        self.assertEqual(loaded["vad_max_seconds"], 60)

    def test_data_directory_override(self):
        with patch.dict(os.environ, {"MYSOUND_DATA_DIR": str(self.directory)}):
            self.assertEqual(default_data_dir(), self.directory)
            ConfigStore().update(device="cpu")
        self.assertTrue((self.directory / "config.json").exists())

    def test_history_search_edit_delete_and_order(self):
        history = HistoryStore(self.directory)
        first = history.add("first.wav", "whisper", "你好", [{"start": 0, "end": 1, "text": "你好"}])
        second = history.add("second.wav", "qwen3", "WORLD")
        self.assertEqual([record["id"] for record in history.list()], [second["id"], first["id"]])
        self.assertEqual(history.list("world")[0]["id"], second["id"])
        updated = history.update(first["id"], "您好")
        self.assertEqual(updated["text"], "您好")
        self.assertEqual(updated["segments"][0]["text"], "你好")
        self.assertEqual(history.get(first["id"])["text"], "您好")
        history.delete(second["id"])
        self.assertIsNone(history.get(second["id"]))

    def test_corruption_is_preserved_and_reported(self):
        path = self.directory / "history.json"
        broken = '{"broken":'
        path.write_text(broken, encoding="utf-8")
        history = HistoryStore(self.directory)
        with self.assertRaises(StorageError):
            history.list()
        backups = list(self.directory.glob("history.json.corrupt-*"))
        self.assertEqual(len(backups), 1)
        self.assertEqual(backups[0].read_text(encoding="utf-8"), broken)
        self.assertFalse(path.exists())
        self.assertEqual(history.list(), [])

    def test_failed_atomic_replacement_preserves_previous_data(self):
        config = ConfigStore(self.directory)
        config.update(device="cpu")
        with patch("storage.json_store.os.replace", side_effect=OSError("disk unavailable")):
            with self.assertRaises(OSError):
                config.update(device="cuda")
        self.assertEqual(config.load()["device"], "cpu")
        self.assertEqual(list(self.directory.glob("*.tmp")), [])

    def test_concurrent_instances_do_not_lose_history_records(self):
        def add_record(index):
            return HistoryStore(self.directory).add(str(index), "test", str(index))

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
            list(pool.map(add_record, range(25)))
        records = HistoryStore(self.directory).list()
        self.assertEqual(len(records), 25)
        self.assertEqual({record["text"] for record in records}, {str(index) for index in range(25)})
        self.assertEqual(len(json.loads((self.directory / "history.json").read_text())), 25)


if __name__ == "__main__":
    unittest.main()
