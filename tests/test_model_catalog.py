"""Resource advice depends on available memory and acknowledges uncertainty."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from core.model_catalog import assess_model, describe_model, recommend_models


class ModelCatalogTests(unittest.TestCase):
    @staticmethod
    def _onnx_pair(folder, precision, *, prefix="medium"):
        suffix = ".int8.onnx" if precision == "int8" else ".onnx"
        for role, megabytes in (("encoder", 1 if precision == "int8" else 4), ("decoder", 2 if precision == "int8" else 6)):
            with (folder / f"{prefix}-{role}{suffix}").open("wb") as stream:
                stream.truncate(megabytes * 1024 ** 2)
        (folder / f"{prefix}-tokens.txt").write_text("token 0\n", encoding="utf-8")

    def test_low_available_memory_downgrades_despite_large_total(self):
        hardware = {"ram_total_gib": 128, "ram_available_gib": 3, "gpus": [{"name": "Busy GPU", "cuda": True, "total_vram_gib": 24, "free_vram_gib": 0.1}]}
        choices = recommend_models(hardware)
        self.assertEqual(choices[0]["id"], "whisper-tiny")
        self.assertEqual(choices[0]["device"], "cpu")
        self.assertEqual(choices[0]["fit"], "recommended")
        large = next(item for item in choices if item["id"] == "whisper-large")
        self.assertEqual(large["fit"], "insufficient")
        self.assertIn("gpu_host_ram_low", large["reason_codes"])

    def test_cpu_choice_uses_core_count_and_available_ram(self):
        for cores, available, expected in [(2, 32, "whisper-tiny"), (4, 32, "whisper-base"), (8, 32, "whisper-small"), (16, 3, "whisper-base"), (16, 2.5, "whisper-tiny")]:
            with self.subTest(cores=cores, available=available):
                choices = recommend_models({"logical_cores": cores, "ram_available_gib": available})
                self.assertEqual(choices[0]["id"], expected)
                self.assertTrue(choices[0]["top_recommendation"])
                self.assertIn("heuristic", choices[0]["preference_reason_en"])

    def test_large_gpu_prefers_turbo_and_larger_qwen_with_headroom(self):
        choices = recommend_models({"logical_cores": 16, "ram_available_gib": 32, "gpus": [{"cuda": True, "free_vram_gib": 24}]})
        self.assertEqual(choices[0]["id"], "whisper-turbo")
        self.assertEqual(choices[0]["device"], "cuda")
        self.assertTrue(next(item for item in choices if item["id"] == "qwen3-1.7b")["preferred"])
        self.assertEqual(sum(item["top_recommendation"] for item in choices), 1)

    def test_second_gpu_cannot_make_cuda_zero_recommendation_fit(self):
        hardware = {"logical_cores": 8, "ram_available_gib": 32, "gpus": [
            {"cuda": True, "cuda_ordinal": 0, "free_vram_gib": 0.1},
            {"cuda": True, "cuda_ordinal": 1, "free_vram_gib": 48},
        ]}
        choices = recommend_models(hardware)
        self.assertTrue(all(item["device"] == "cpu" for item in choices))
        self.assertEqual(choices[0]["id"], "whisper-small")

    def test_logical_cuda_zero_order_takes_precedence_over_physical_order(self):
        choices = recommend_models({"ram_available_gib": 32, "gpus": [
            {"name": "Physical zero", "cuda": True, "cuda_ordinal": 1, "free_vram_gib": 0.1},
            {"name": "Physical one", "cuda": True, "cuda_ordinal": 0, "free_vram_gib": 24},
        ]})
        self.assertEqual(choices[0]["id"], "whisper-turbo")
        self.assertEqual(choices[0]["gpu_name"], "Physical one")

    def test_unverified_multi_gpu_order_does_not_claim_memory_fit(self):
        choices = recommend_models({"logical_cores": 8, "ram_available_gib": 32, "gpus": [
            {"cuda": True, "cuda_ordinal": 0, "cuda_order_verified": False, "free_vram_gib": 48},
            {"cuda": True, "cuda_ordinal": 1, "cuda_order_verified": False, "free_vram_gib": 0.1},
        ]})
        self.assertTrue(all(item["device"] == "cpu" for item in choices))
        self.assertIn("cuda_order_unknown", choices[0]["reason_codes"])

    def test_unknown_available_memory_never_means_zero_or_total_capacity(self):
        choices = recommend_models({"ram_total_gib": 128, "ram_available_gib": None, "gpus": []})
        self.assertTrue(all(item["fit"] == "unknown" for item in choices))
        self.assertFalse(any(item["preferred"] for item in choices))
        self.assertIn("ram_available_unknown", choices[0]["reason_codes"])

    def test_whisper_gpu_recommendation_checks_cpu_staging_memory(self):
        hardware = {"ram_available_gib": 3, "gpus": [{"name": "Free GPU", "cuda": True, "free_vram_gib": 40}]}
        model = assess_model(describe_model("whisper", "large-v3.pt"), hardware)
        self.assertEqual(model["fit"], "insufficient")
        self.assertIn("gpu_host_ram_low", model["reason_codes"])

    def test_other_gpu_is_not_recommended_as_cuda(self):
        choices = recommend_models({"ram_available_gib": 32, "gpus": [{"name": "AMD", "cuda": False, "free_vram_gib": 64}]})
        self.assertTrue(all(item["device"] == "cpu" for item in choices))

    def test_large_free_vram_with_unknown_host_ram_remains_unknown(self):
        model = assess_model(describe_model("whisper", "tiny.pt"), {"ram_available_gib": None, "gpus": [{"cuda": True, "free_vram_gib": 40}]})
        self.assertEqual(model["fit"], "unknown")

    def test_real_metadata_wins_over_misleading_whisper_folder_name(self):
        with TemporaryDirectory() as directory:
            folder = Path(directory) / "large-custom"
            folder.mkdir()
            (folder / "config.json").write_text(json.dumps({"model_type": "whisper", "d_model": 512, "decoder_layers": 6}))
            self.assertEqual(describe_model("whisper", str(folder))["id"], "whisper-base")

    def test_qwen_small_variant_is_identified_from_config(self):
        with TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / "config.json").write_text(json.dumps({"thinker_config": {"text_config": {"hidden_size": 1024}}}))
            result = describe_model("qwen3", str(folder))
            self.assertEqual(result["id"], "qwen3-0.6b")
            self.assertIsNone(result["official_vram_gb"])

    def test_qwen_metadata_takes_precedence_over_directory_name(self):
        with TemporaryDirectory() as directory:
            folder = Path(directory) / "Qwen3-ASR-0.6B"
            folder.mkdir()
            (folder / "config.json").write_text(json.dumps({"thinker_config": {"text_config": {"hidden_size": 2048}}}))
            self.assertEqual(describe_model("qwen3", str(folder))["id"], "qwen3-1.7b")

    def test_unknown_checkpoint_does_not_get_large_model_estimates(self):
        result = describe_model("whisper", "/some/custom-checkpoint.pt")
        self.assertFalse(result["identified"])
        self.assertIsNone(result["cpu_ram_gib"])
        self.assertIsNone(result["gpu_vram_gib"])

    def test_official_reference_and_application_estimates_are_separate(self):
        result = describe_model("whisper", "large-v3-turbo.pt")
        self.assertEqual(result["id"], "whisper-turbo")
        self.assertEqual(result["official_vram_gb"], 6)
        self.assertEqual(result["official_speed_relative"], 8)
        self.assertTrue(result["estimate"])
        self.assertIn("不是官方最低配置", result["budget_note"])
        self.assertIn("A100", result["speed_note"])

    def test_onnx_folder_uses_int8_budget_and_only_selected_files(self):
        with TemporaryDirectory() as directory:
            folder = Path(directory)
            self._onnx_pair(folder, "int8")
            self._onnx_pair(folder, "fp32")
            model = describe_model("whisper_onnx", str(folder))
            self.assertEqual(model["id"], "whisper-onnx-medium-int8")
            self.assertEqual(model["family"], "whisper_onnx")
            self.assertEqual(model["precision"], "int8")
            self.assertEqual(model["cpu_ram_gib"], [3, 6])
            self.assertEqual(len(model["selected_files"]), 3)
            self.assertGreater(model["folder_disk_gib"], model["disk_gib"])
            self.assertTrue(all(".int8." in item for item in model["selected_files"] if item.endswith(".onnx")))

    def test_explicit_onnx_file_keeps_fp32_budget_when_int8_also_exists(self):
        with TemporaryDirectory() as directory:
            folder = Path(directory)
            self._onnx_pair(folder, "int8")
            self._onnx_pair(folder, "fp32")
            for role in ("encoder", "decoder"):
                selected = folder / f"medium-{role}.onnx"
                model = describe_model("whisper_onnx", str(selected))
                self.assertEqual(model["id"], "whisper-onnx-medium-fp32")
                self.assertEqual(model["cpu_ram_gib"], [6, 10])
                self.assertEqual(model["precision"], "fp32")
                self.assertEqual(model["resolved_path"], str(selected))
                self.assertFalse(any(".int8." in item for item in model["selected_files"]))

    def test_onnx_cpu_advice_does_not_borrow_native_staging_or_gpu_speed(self):
        with TemporaryDirectory() as directory:
            folder = Path(directory)
            self._onnx_pair(folder, "int8")
            model = assess_model(describe_model("whisper_onnx", str(folder)), {
                "ram_available_gib": 8, "gpus": [{"cuda": True, "free_vram_gib": 80}],
            })
            self.assertEqual(model["device"], "cpu")
            self.assertEqual(model["supported_devices"], ["cpu"])
            self.assertEqual(model["fit"], "recommended")
            self.assertIn("backend_cpu_only", model["reason_codes"])
            for key in ("gpu_vram_gib", "gpu_host_ram_gib", "official_vram_gb", "official_speed_relative"):
                self.assertIsNone(model[key])
            self.assertIn("不是官方最低配置", model["budget_note"])
            self.assertIn("不经过原生 Whisper .pt", model["budget_note"])

    def test_onnx_suggestions_preserve_smaller_native_whisper_cpu_preference(self):
        models = recommend_models({"logical_cores": 8, "ram_available_gib": 16})
        self.assertEqual(models[0]["id"], "whisper-small")
        quantized = next(model for model in models if model["id"] == "whisper-onnx-medium-int8")
        self.assertTrue(quantized["preferred"])
        self.assertFalse(quantized["top_recommendation"])
        self.assertIn("not a promise", quantized["preference_reason_en"])
        self.assertEqual(quantized["device"], "cpu")

    def test_directory_with_turbo_checkpoint_uses_turbo_profile(self):
        with TemporaryDirectory() as directory:
            folder = Path(directory) / "Whisper-large-v3-turbo"
            folder.mkdir()
            (folder / "large-v3-turbo.pt").write_bytes(b"metadata test only")
            self.assertEqual(describe_model("whisper", str(folder))["id"], "whisper-turbo")

    def test_three_file_qwen_package_is_described_from_model_config(self):
        with TemporaryDirectory() as directory:
            folder = Path(directory) / "Qwen3-ASR-0.6B"
            folder.mkdir()
            (folder / "config.json").write_text(json.dumps({"thinker_config": {"text_config": {"hidden_size": 1024}}}))
            (folder / "tokenizer.json").write_text("{}")
            (folder / "model.safetensors").write_bytes(b"metadata test only")
            for selection in (folder, folder / "config.json", folder / "model.safetensors"):
                self.assertEqual(describe_model("qwen3", str(selection))["id"], "qwen3-0.6b")


if __name__ == "__main__":
    unittest.main()
