"""Adapter contracts without downloading models or loading the ML runtime."""

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
import weakref
from unittest.mock import MagicMock, patch

from core.model_manager import (
    BaseASRModel, ConfuciusASRModel, ModelManager, Qwen3ASRModel,
    TranscriptSegment, WhisperASRModel, WhisperOnnxASRModel, _whisper_onnx_language, scan_models,
)
from tests.test_model_paths import onnx_folder


class FakeModel(BaseASRModel):
    def load(self):
        self._backend = object()

    def transcribe(self, audio, sample_rate=16000, language=None):
        return []


def complete_hf_model(directory):
    path = Path(directory)
    (path / "config.json").write_text('{"model_type":"qwen3_asr"}')
    for name in ("model.safetensors", "preprocessor_config.json", "tokenizer_config.json", "tokenizer.json"):
        (path / name).write_text("{}")


class ModelTests(unittest.TestCase):
    def test_onnx_unknown_language_is_rejected_before_native_config_or_decode(self):
        model = WhisperOnnxASRModel("/local/onnx", "cpu")
        model._variant = "medium"
        model._backend = MagicMock()
        with patch("core.model_manager._audio", return_value=[0.0] * 160):
            for language in ("not-a-language", "yue", "Cantonese"):
                with self.subTest(language=language), self.assertRaises(ValueError) as caught:
                    model.transcribe([], language=language)
                self.assertIn("does not support", caught.exception.message_en)
        model._backend.recognizer.set_config.assert_not_called()
        model._backend.create_stream.assert_not_called()
        model._backend.decode_stream.assert_not_called()

    def test_onnx_language_validation_respects_variant(self):
        for language in ("zh", "en", "ja", "ko", "de", "haw"):
            self.assertEqual(_whisper_onnx_language(language, "medium"), language)
        for language in (None, "auto", "自动", ""):
            self.assertEqual(_whisper_onnx_language(language, "medium"), "")
            self.assertEqual(_whisper_onnx_language(language, "medium.en"), "")
        self.assertEqual(_whisper_onnx_language("English", "medium.en"), "en")
        self.assertEqual(_whisper_onnx_language("yue", "large-v3"), "yue")
        self.assertEqual(_whisper_onnx_language("yue", "large-v3-turbo"), "yue")
        for variant, language in (("medium.en", "zh"), ("small.en", "ja"), ("large-v2", "yue"), ("unknown", "yue")):
            with self.subTest(variant=variant, language=language), self.assertRaises(ValueError):
                _whisper_onnx_language(language, variant)

    def test_onnx_load_uses_cpu_int8_and_imports_no_torch(self):
        backend = MagicMock()
        sdk = SimpleNamespace(OfflineRecognizer=SimpleNamespace(from_whisper=MagicMock(return_value=backend)))
        with TemporaryDirectory() as directory:
            model_path = onnx_folder(directory)
            with patch("core.model_manager._dependency", return_value=sdk) as dependency:
                manager = ModelManager()
                model = manager.load("whisper_onnx", str(model_path))
                self.assertIs(model, manager.load("whisper_onnx", str(model_path)))
            dependency.assert_called_once_with("sherpa_onnx", "pip install sherpa-onnx==1.13.8")
            kwargs = sdk.OfflineRecognizer.from_whisper.call_args.kwargs
            self.assertEqual(kwargs["provider"], "cpu")
            self.assertEqual(kwargs["language"], "")
            self.assertTrue(kwargs["encoder"].endswith("medium-encoder.int8.onnx"))
            self.assertTrue(kwargs["decoder"].endswith("medium-decoder.int8.onnx"))
            self.assertEqual(model.device, "cpu")
            manager.unload()
            self.assertIsNone(model._backend)

    def test_onnx_explicit_fp32_file_is_used(self):
        sdk = MagicMock()
        with TemporaryDirectory() as directory:
            selected = onnx_folder(directory) / "medium-decoder.onnx"
            with patch("core.model_manager._dependency", return_value=sdk):
                model = WhisperOnnxASRModel(str(selected), "cpu")
                model.load()
            kwargs = sdk.OfflineRecognizer.from_whisper.call_args.kwargs
            self.assertTrue(kwargs["encoder"].endswith("medium-encoder.onnx"))
            self.assertTrue(kwargs["decoder"].endswith("medium-decoder.onnx"))

    def test_onnx_explicit_cuda_is_rejected_before_loading_weights(self):
        with patch("core.model_manager._dependency") as dependency:
            with self.assertRaises(RuntimeError) as caught:
                WhisperOnnxASRModel("/any/path", "cuda").load()
            dependency.assert_not_called()
            self.assertIn("CPU", str(caught.exception))
            self.assertIn("CPU", caught.exception.message_en)

    def test_onnx_splits_long_audio_and_updates_language_without_reload(self):
        model = WhisperOnnxASRModel("/local/onnx", "cpu")
        backend = MagicMock()
        streams = []
        for index in range(3):
            stream = MagicMock()
            stream.result = SimpleNamespace(text=f"  part {index}  ", timestamps=[])
            streams.append(stream)
        backend.create_stream.side_effect = streams
        model._backend = backend
        waveform = [0.0] * (61 * 16000)
        with patch("core.model_manager._audio", return_value=waveform):
            result = model.transcribe([], language="Chinese")
        self.assertEqual(result, [TranscriptSegment(0, 29, "part 0"), TranscriptSegment(29, 58, "part 1"), TranscriptSegment(58, 61, "part 2")])
        self.assertEqual([len(stream.accept_waveform.call_args.args[1]) for stream in streams], [29 * 16000, 29 * 16000, 3 * 16000])
        self.assertEqual(backend.config.model_config.whisper.language, "zh")
        backend.recognizer.set_config.assert_called_once_with(backend.config)

    def test_onnx_language_auto_reset_and_blank_chunks(self):
        model = WhisperOnnxASRModel("/local/onnx", "cpu")
        backend = MagicMock()
        backend.create_stream.return_value.result.text = "   "
        model._backend = backend
        with patch("core.model_manager._audio", return_value=[0.0] * 1600):
            self.assertEqual(model.transcribe([], language="en"), [])
            self.assertEqual(backend.config.model_config.whisper.language, "en")
            self.assertEqual(model.transcribe([], language="auto"), [])
            self.assertEqual(backend.config.model_config.whisper.language, "")

    def test_scan_detects_confucius_from_metadata_and_unprepared_files(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            confucius = root / "netease-youdao"
            confucius.mkdir()
            complete_hf_model(confucius)
            (confucius / "README.md").write_text("# Confucius4-R2T2")
            qwen = root / "Qwen3-ASR"
            qwen.mkdir()
            (qwen / "Qwen3-ASR-1.7B.zip").touch()
            whisper = root / "Whisper"
            whisper.mkdir()
            (whisper / "large.pt.incomplete").touch()
            models = scan_models(root)
            self.assertEqual([item["type"] for item in models], ["confucius", "whisper", "qwen3"])
            self.assertTrue(models[0]["ready"])
            self.assertFalse(models[1]["ready"])
            self.assertFalse(models[2]["ready"])

    def test_manager_reuses_model_and_releases_on_switch(self):
        with TemporaryDirectory() as directory:
            complete_hf_model(directory)
            manager = ModelManager()
            manager.adapters = {"qwen3": FakeModel, "confucius": FakeModel}
            first = manager.load("qwen3", directory)
            self.assertIs(first, manager.load("qwen3", directory))
            second = manager.load("confucius", directory)
            self.assertIsNone(first._backend)
            self.assertIs(second, manager.model)
            manager.unload()
            self.assertIsNone(manager.model)

    def test_manager_rejects_archive_without_importing_torch(self):
        with TemporaryDirectory() as directory:
            archive = Path(directory) / "Qwen3.zip"
            archive.touch()
            with self.assertRaisesRegex(ValueError, "解压"):
                ModelManager().load("qwen3", str(archive))

    def test_qwen_local_loading_and_chunk_timestamps(self):
        fake_torch = SimpleNamespace(
            float32="float32", cuda=SimpleNamespace(is_available=lambda: False)
        )
        backend = MagicMock()
        backend.transcribe.return_value = [SimpleNamespace(text="  你好世界。  ")]
        sdk = SimpleNamespace(Qwen3ASRModel=SimpleNamespace(from_pretrained=MagicMock(return_value=backend)))
        with TemporaryDirectory() as directory:
            complete_hf_model(directory)
            with patch("core.model_manager._dependency", side_effect=[fake_torch, sdk]):
                model = Qwen3ASRModel(directory)
                model.load()
            kwargs = sdk.Qwen3ASRModel.from_pretrained.call_args.kwargs
            self.assertTrue(kwargs["local_files_only"])
            self.assertFalse(kwargs["trust_remote_code"])
            self.assertEqual(kwargs["device_map"], "cpu")
            waveform = [0.1] * 8000
            with patch("core.model_manager._audio", return_value=waveform):
                segments = model.transcribe(waveform, language="zh")
            self.assertEqual(segments, [TranscriptSegment(0, 0.5, "你好世界。")])
            self.assertEqual(backend.transcribe.call_args.kwargs["language"], "Chinese")
            self.assertFalse(backend.transcribe.call_args.kwargs["return_time_stamps"])

    def test_confucius_uses_the_shared_official_transformers_backend(self):
        self.assertTrue(issubclass(ConfuciusASRModel, Qwen3ASRModel))
        self.assertEqual(ConfuciusASRModel.model_type, "confucius")

    def test_native_whisper_returns_relative_segments_and_disables_cpu_fp16(self):
        model = WhisperASRModel("/local/large.pt", "cpu")
        model._native = True
        model._backend = MagicMock()
        model._backend.transcribe.return_value = {
            "segments": [{"start": 0.1, "end": 0.4, "text": " 测试 "}]
        }
        with patch("core.model_manager._audio", return_value=[0.0] * 8000):
            self.assertEqual(model.transcribe([], language="zh"), [TranscriptSegment(0.1, 0.4, "测试")])
        self.assertFalse(model._backend.transcribe.call_args.kwargs["fp16"])
        self.assertEqual(model._backend.transcribe.call_args.kwargs["language"], "chinese")

    def test_whisper_loads_checkpoint_on_cpu_and_halves_before_gpu_transfer(self):
        class LayerNorm:
            float = MagicMock()
        fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True), nn=SimpleNamespace(LayerNorm=LayerNorm))
        backend = MagicMock()
        normalization = LayerNorm()
        backend.modules.return_value = [normalization]
        sdk = SimpleNamespace(load_model=MagicMock(return_value=backend))
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "large.pt"
            checkpoint.touch()
            with patch("core.model_manager._dependency", side_effect=[fake_torch, sdk]):
                model = WhisperASRModel(str(checkpoint), "auto")
                model.load()
            sdk.load_model.assert_called_once_with(str(checkpoint), device="cpu")
            self.assertEqual([call[0] for call in backend.mock_calls], ["half", "modules", "to", "eval"])
            normalization.float.assert_called_once()
            backend.to.assert_called_once_with("cuda:0")

    def test_auto_load_oom_retries_cpu_after_freeing_failed_loader_frames(self):
        attempts, weak_references = [], []
        class Resource:
            pass
        class LoadOOM(FakeModel):
            def load(self):
                attempts.append(self.device)
                if self.device == "auto":
                    temporary_checkpoint = Resource()
                    weak_references.append(weakref.ref(temporary_checkpoint))
                    self.device = "cuda:0"
                    raise RuntimeError("CUDA out of memory. Tried to allocate")
                self.assert_released = weak_references[0]() is None
                super().load()
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "large.pt"
            checkpoint.touch()
            status = MagicMock()
            manager = ModelManager(on_status=status)
            manager.adapters = {"whisper": LoadOOM}
            with self.assertLogs("core.model_manager", level="WARNING") as logged:
                result = manager.load("whisper", str(checkpoint))
            self.assertIn("Tried to allocate", logged.output[0])
            self.assertIn("Traceback", logged.output[0])
            self.assertEqual(attempts, ["auto", "cpu"])
            self.assertTrue(result.assert_released)
            self.assertEqual(result.device, "cpu")
            status.assert_called_once()
            self.assertIn("CPU", status.call_args.args[0])
            self.assertIs(manager.load("whisper", str(checkpoint)), result)

    def test_explicit_cuda_oom_is_actionable_and_does_not_fallback(self):
        attempts = []
        class LoadOOM(FakeModel):
            def load(self):
                attempts.append(self.device)
                raise RuntimeError("CUDA out of memory")
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "large.pt"
            checkpoint.touch()
            manager = ModelManager()
            manager.adapters = {"whisper": LoadOOM}
            with self.assertLogs("core.model_manager", level="WARNING"):
                with self.assertRaisesRegex(RuntimeError, "请选择自动模式"):
                    manager.load("whisper", str(checkpoint), "cuda")
            self.assertEqual(attempts, ["cuda"])
            self.assertIsNone(manager.model)

    def test_auto_inference_oom_retries_same_audio_once_and_reuses_cpu(self):
        attempts = []
        class InferenceOOM(FakeModel):
            def load(self):
                super().load()
                if self.device == "auto":
                    self.device = "cuda:0"
            def transcribe(self, audio, sample_rate=16000, language=None):
                attempts.append((self.device, audio, language))
                if self.device.startswith("cuda"):
                    raise RuntimeError("CUDA out of memory")
                return [TranscriptSegment(0, 1, "hello")]
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "large.pt"
            checkpoint.touch()
            manager = ModelManager(on_status=MagicMock())
            manager.adapters = {"whisper": InferenceOOM}
            old_model = manager.load("whisper", str(checkpoint))
            with self.assertLogs("core.model_manager", level="WARNING") as logged:
                segments = manager.transcribe([1, 2, 3], language="English")
            self.assertIn("CUDA out of memory", logged.output[0])
            self.assertEqual(segments[0].text, "hello")
            self.assertEqual([item[0] for item in attempts], ["cuda:0", "cpu"])
            self.assertEqual(attempts[0][1:], attempts[1][1:])
            self.assertIsNone(old_model._backend)
            self.assertIs(manager.load("whisper", str(checkpoint)), manager.model)
            manager.transcribe([4])
            self.assertEqual(attempts[-1][0], "cpu")
            manager.on_status.assert_called_once()

    def test_unrelated_load_error_never_falls_back(self):
        attempts = []
        class Broken(FakeModel):
            def load(self):
                attempts.append(self.device)
                raise RuntimeError("malformed checkpoint")
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "large.pt"
            checkpoint.touch()
            manager = ModelManager()
            manager.adapters = {"whisper": Broken}
            with self.assertRaisesRegex(RuntimeError, "malformed"):
                manager.load("whisper", str(checkpoint))
            self.assertEqual(attempts, ["auto"])


if __name__ == "__main__":
    unittest.main()
