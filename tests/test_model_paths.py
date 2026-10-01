"""Resolve genuine filesystem selections without loading ML dependencies."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from core.model_paths import ModelPathError, infer_model_type, resolve_model_path, resolve_whisper_onnx_files, scan_models


def model_folder(parent, name="Qwen3-ASR"):
    path = Path(parent) / name
    path.mkdir(parents=True)
    (path / "config.json").write_text('{"model_type":"qwen3_asr"}')
    for filename in ("model.safetensors", "preprocessor_config.json", "tokenizer_config.json", "tokenizer.json"):
        (path / filename).write_text("{}")
    if "confucius" in name.lower():
        (path / "README.md").write_text("Confucius4-R2T2")
    return path


def onnx_folder(parent, name="sherpa-onnx-whisper-medium", precisions=("int8", "fp32")):
    path = Path(parent) / name
    path.mkdir(parents=True)
    for precision in precisions:
        suffix = ".int8.onnx" if precision == "int8" else ".onnx"
        for part in ("encoder", "decoder"):
            (path / ("medium-" + part + suffix)).touch()
    (path / "medium-tokens.txt").write_text("tokens")
    return path


class ModelPathTests(unittest.TestCase):
    def test_onnx_directory_prefers_complete_int8_and_is_not_native_whisper(self):
        with TemporaryDirectory() as directory:
            model = onnx_folder(directory)
            self.assertEqual(infer_model_type(model), "whisper_onnx")
            files = resolve_whisper_onnx_files(model)
            self.assertEqual(files["precision"], "int8")
            self.assertEqual(files["encoder"], model / "medium-encoder.int8.onnx")
            self.assertEqual(resolve_model_path(directory, "whisper_onnx"), model)
            found = scan_models(directory)
            self.assertEqual(len(found), 1)
            self.assertEqual(found[0]["type"], "whisper_onnx")
            self.assertTrue(found[0]["ready"])

    def test_explicit_onnx_file_keeps_precision_and_path(self):
        with TemporaryDirectory() as directory:
            model = onnx_folder(directory)
            for filename, precision in [("medium-encoder.onnx", "fp32"), ("medium-decoder.onnx", "fp32"), ("medium-decoder.int8.onnx", "int8")]:
                selected = model / filename
                self.assertEqual(resolve_model_path(selected, "whisper_onnx"), selected)
                self.assertEqual(resolve_whisper_onnx_files(selected)["precision"], precision)

    def test_incomplete_int8_falls_back_as_a_pair_but_explicit_file_does_not(self):
        with TemporaryDirectory() as directory:
            model = onnx_folder(directory)
            (model / "medium-decoder.int8.onnx").unlink()
            files = resolve_whisper_onnx_files(model)
            self.assertEqual(files["precision"], "fp32")
            self.assertEqual(files["encoder"].name, "medium-encoder.onnx")
            with self.assertRaises(ModelPathError) as caught:
                resolve_model_path(model / "medium-encoder.int8.onnx", "whisper_onnx")
            self.assertIn("medium-decoder.int8.onnx", str(caught.exception))
            self.assertIn("medium-decoder.int8.onnx", caught.exception.message_en)

    def test_missing_tokens_is_incomplete_without_requesting_hf_config(self):
        with TemporaryDirectory() as directory:
            model = onnx_folder(directory)
            (model / "medium-tokens.txt").unlink()
            found = scan_models(directory)
            self.assertFalse(found[0]["ready"])
            self.assertIn("medium-tokens.txt", found[0]["status"])
            self.assertNotIn("config.json", found[0]["status"])

    def test_encoder_and_decoder_different_precision_are_not_mixed(self):
        with TemporaryDirectory() as directory:
            model = onnx_folder(directory)
            (model / "medium-decoder.int8.onnx").unlink()
            (model / "medium-encoder.onnx").unlink()
            with self.assertRaises(ModelPathError):
                resolve_model_path(model, "whisper_onnx")

    def test_onnx_pair_is_recognized_after_directory_is_renamed(self):
        with TemporaryDirectory() as directory:
            model = onnx_folder(directory, "my-download")
            self.assertEqual(infer_model_type(model), "whisper_onnx")
            self.assertEqual(resolve_model_path(model, "whisper_onnx"), model)

    def test_selecting_metadata_or_weights_returns_containing_model_folder(self):
        with TemporaryDirectory() as directory:
            model = model_folder(directory)
            for filename in ("config.json", "model.safetensors", "tokenizer.json"):
                with self.subTest(filename=filename):
                    self.assertEqual(resolve_model_path(model / filename, "qwen3"), model)

    def test_parent_with_one_matching_nested_model_is_resolved(self):
        with TemporaryDirectory() as directory:
            model = model_folder(Path(directory) / "wrapper" / "downloads", "Confucius4-R2T2")
            (Path(directory) / "large.pt").touch()
            self.assertEqual(resolve_model_path(directory, "confucius"), model)

    def test_multiple_models_require_an_explicit_choice(self):
        with TemporaryDirectory() as directory:
            first = model_folder(directory, "Qwen3-one")
            second = model_folder(directory, "Qwen3-two")
            with self.assertRaises(ValueError) as caught:
                resolve_model_path(directory, "qwen3")
            self.assertIn(str(first), str(caught.exception))
            self.assertIn(str(second), str(caught.exception))

    def test_partial_directory_is_not_marked_ready(self):
        with TemporaryDirectory() as directory:
            model = Path(directory) / "Qwen3-ASR"
            model.mkdir()
            (model / "config.json.incomplete").write_text("partial")
            found = scan_models(directory)
            self.assertFalse(found[0]["ready"])
            self.assertIn("config.json", found[0]["status"])
            with self.assertRaisesRegex(ValueError, "尚未下载完成"):
                resolve_model_path(model, "qwen3")

    def test_missing_sharded_weight_is_reported(self):
        with TemporaryDirectory() as directory:
            model = model_folder(directory)
            (model / "model.safetensors").unlink()
            (model / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"one": "model-00001.safetensors", "two": "model-00002.safetensors"}}))
            (model / "model-00001.safetensors").touch()
            self.assertFalse(scan_models(directory)[0]["ready"])
            with self.assertRaisesRegex(ValueError, "model-00002.safetensors"):
                resolve_model_path(model, "qwen3")

    def test_archive_is_reported_without_modifying_it(self):
        with TemporaryDirectory() as directory:
            archive = Path(directory) / "Qwen3-ASR.zip"
            archive.write_bytes(b"archive-placeholder")
            before = archive.read_bytes()
            with self.assertRaisesRegex(ValueError, "解压"):
                resolve_model_path(directory, "qwen3")
            self.assertEqual(archive.read_bytes(), before)
            self.assertEqual(list(Path(directory).iterdir()), [archive])


if __name__ == "__main__":
    unittest.main()
