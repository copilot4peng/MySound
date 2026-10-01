"""Documented model facts and conservative, explicitly estimated app budgets.

See docs/model-resources.md for sources, assumptions and recommendation rules.
This module neither downloads anything nor imports torch or model weights.
"""

from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
import re


WHISPER_SOURCE = "https://github.com/openai/whisper#available-models-and-languages"
QWEN_SOURCE = "https://github.com/QwenLM/Qwen3-ASR"
CONFUCIUS_SOURCE = "https://huggingface.co/netease-youdao/Confucius4-R2T2"
SHERPA_WHISPER_SOURCE = "https://k2-fsa.github.io/sherpa/onnx/pretrained_models/whisper/export-onnx.html"
SHERPA_MEDIUM_SOURCE = "https://huggingface.co/csukuangfj/sherpa-onnx-whisper-medium/tree/main"
ONNX_QUANTIZATION_SOURCE = "https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html"

_BUDGET_ZH = "RAM/显存区间为本程序单任务、CPU FP32 / CUDA FP16或BF16的规划估算，含加载余量；不是官方最低配置或本机实测。音频长度和后台占用会影响结果。"
_BUDGET_EN = "RAM/VRAM ranges are planning estimates for one task with CPU FP32 or CUDA FP16/BF16, including loading headroom. They are not official minimums or local benchmarks; audio length and other processes affect usage."
_WHISPER_SPEED_ZH = "官方相对速度来自 A100 英语测试，以 large=1；不代表本机或 CPU 倍速。当前程序未做本机测速，小模型通常更适合低延迟 CPU 听写。"
_WHISPER_SPEED_EN = "Official relative speed is from English speech on an A100, with large=1; it is not a local or CPU speed measurement. This app has not benchmarked this machine; smaller models generally suit lower-latency CPU dictation."


def _model(identifier, family, name, parameters, cpu, gpu, host, source, speed_note, speed_en, official_vram=None, official_speed=None):
    return {
        "id": identifier, "family": family, "name": name, "parameters": parameters,
        "disk_gib": None, "cpu_ram_gib": list(cpu), "gpu_vram_gib": list(gpu) if gpu is not None else None,
        "gpu_host_ram_gib": list(host) if host is not None else None, "official_vram_gb": official_vram,
        "official_speed_relative": official_speed,
        "speed_note": speed_note, "speed_note_en": speed_en,
        "budget_note": _BUDGET_ZH, "budget_note_en": _BUDGET_EN,
        "sources": [{"title": "Official model documentation", "url": source}],
        "identified": True, "estimate": True, "supported_devices": ["cpu", "cuda"],
    }


def _onnx_medium(precision, cpu):
    model = _model(
        f"whisper-onnx-medium-{precision}", "whisper_onnx",
        f"Sherpa-ONNX Whisper medium · {precision.upper()}", "769 M", cpu, None, None,
        SHERPA_WHISPER_SOURCE,
        "当前使用 Sherpa-ONNX 的 CPU 离线推理。INT8 的速度收益取决于 CPU 指令集和运行库；medium 仍可能较慢，没有本机倍速或实时保证。",
        "This adapter uses Sherpa-ONNX offline inference on CPU. INT8 speed gains depend on CPU instructions and the runtime; medium may still be slow. No local speed or real-time guarantee is available.",
    )
    model.update(
        supported_devices=["cpu"], backend="sherpa-onnx", precision=precision,
        budget_note=(
            f"此范围是 Sherpa-ONNX medium {precision.upper()} 单任务 CPU 推理的主存规划估算，包含 ONNX Runtime 会话、图优化与音频缓存余量，未经本机实测，也不是官方最低配置。"
            "它不经过原生 Whisper .pt 的 PyTorch FP32 模型加检查点暂存流程；本程序的此适配器只使用 CPU，CUDA 显存/主存预算不适用。"
        ),
        budget_note_en=(
            f"This is an unmeasured planning estimate for single-task Sherpa-ONNX medium {precision.upper()} CPU inference, including ONNX Runtime sessions, graph optimization, and audio-buffer headroom; it is not an official minimum. "
            "It does not use native Whisper .pt staging of a PyTorch FP32 model plus checkpoint. This adapter is CPU-only; CUDA VRAM and staging-RAM budgets do not apply."
        ),
        sources=[
            {"title": "Sherpa-ONNX Whisper export and model format", "url": SHERPA_WHISPER_SOURCE},
            {"title": "Sherpa-ONNX medium model files", "url": SHERPA_MEDIUM_SOURCE},
            {"title": "ONNX Runtime quantization and hardware considerations", "url": ONNX_QUANTIZATION_SOURCE},
            {"title": "OpenAI Whisper model sizes", "url": WHISPER_SOURCE},
        ],
    )
    return model


MODEL_CATALOG = [
    _model("whisper-tiny", "whisper", "Whisper tiny", "39 M", (1, 2), (1, 1.5), (1, 2), WHISPER_SOURCE, _WHISPER_SPEED_ZH, _WHISPER_SPEED_EN, 1, 10),
    _model("whisper-base", "whisper", "Whisper base", "74 M", (1.5, 2.5), (1, 2), (1.5, 2.5), WHISPER_SOURCE, _WHISPER_SPEED_ZH, _WHISPER_SPEED_EN, 1, 7),
    _model("whisper-small", "whisper", "Whisper small", "244 M", (2.5, 4), (2, 3), (2.5, 4), WHISPER_SOURCE, _WHISPER_SPEED_ZH, _WHISPER_SPEED_EN, 2, 4),
    _model("whisper-medium", "whisper", "Whisper medium", "769 M", (6, 8), (3.5, 5.5), (6, 8), WHISPER_SOURCE, _WHISPER_SPEED_ZH, _WHISPER_SPEED_EN, 5, 2),
    _model("whisper-large", "whisper", "Whisper large / large-v3", "1550 M", (10, 13), (6, 10), (10, 13), WHISPER_SOURCE, _WHISPER_SPEED_ZH, _WHISPER_SPEED_EN, 10, 1),
    _model("whisper-turbo", "whisper", "Whisper large-v3-turbo", "809 M", (6, 9), (4, 6.5), (6, 9), WHISPER_SOURCE, _WHISPER_SPEED_ZH, _WHISPER_SPEED_EN, 6, 8),
    _model("qwen3-0.6b", "qwen3", "Qwen3-ASR-0.6B", "0.6B variant · ~0.9B checkpoint", (5, 7), (3, 4.5), (2, 4),
           "https://huggingface.co/Qwen/Qwen3-ASR-0.6B", "Qwen3 中较小的版本，通常更省资源；没有适用于本机的官方 CPU 倍速或延迟数据，需实际试用。", "The smaller Qwen3 variant generally uses fewer resources. No official CPU speed or latency figure applies to this machine; measure with your own audio."),
    _model("qwen3-1.7b", "qwen3", "Qwen3-ASR-1.7B", "1.7B variant · ~2B checkpoint", (10, 14), (5.5, 8), (4, 7),
           "https://huggingface.co/Qwen/Qwen3-ASR-1.7B", "大模型更依赖资源；CPU 听写可能跟不上录音速度。本程序没有本机测速，不把官方高并发吞吐当作桌面实时性能。", "This larger model needs more resources; CPU dictation may lag behind recording. This app has no local benchmark and does not treat official concurrent throughput as desktop latency."),
    _model("confucius-r2t2", "confucius", "Confucius4-R2T2", "~2B checkpoint", (10, 14), (5.5, 8), (4, 7), CONFUCIUS_SOURCE,
           "当前使用 Transformers 分句推理；官方 vLLM 流式延迟不适用于本程序。CPU 听写可能较慢，需要本机实测。", "This app uses utterance-based Transformers inference; official vLLM streaming latency does not apply. CPU dictation may be slow and needs a local benchmark."),
    _onnx_medium("int8", (3, 6)),
    _onnx_medium("fp32", (6, 10)),
]

REASON_TEXT = {
    "resources_fit": ("当前可用资源留有估算余量", "Currently available resources leave estimated headroom"),
    "ram_available_unknown": ("可用内存未知，无法判断", "Available RAM is unknown"),
    "ram_low": ("可用内存低于估算区间，优先选择更小模型", "Available RAM is below the estimated range; prefer a smaller model"),
    "ram_tight": ("可用内存余量偏紧", "Available RAM headroom is tight"),
    "vram_free_unknown": ("空闲显存未知", "Free VRAM is unknown"),
    "vram_low": ("空闲显存低于估算区间", "Free VRAM is below the estimated range"),
    "vram_tight": ("空闲显存余量偏紧", "Free VRAM headroom is tight"),
    "gpu_host_ram_low": ("GPU 加载也需要更多可用内存", "GPU loading also needs more available host RAM"),
    "gpu_host_ram_tight": ("GPU 加载时主存余量偏紧", "Host RAM headroom during GPU loading is tight"),
    "gpu_host_ram_unknown": ("GPU 加载所需的可用主存未知", "Available host RAM for GPU loading is unknown"),
    "cpu_only": ("当前按 CPU 方案评估", "Assessed for CPU execution"),
    "backend_cpu_only": ("本程序的此模型适配器仅支持 CPU", "This model adapter only supports CPU in this app"),
    "cuda_hidden": ("NVIDIA GPU 被启动环境隐藏", "The launch environment hides the NVIDIA GPU"),
    "cuda_order_unknown": ("多张 GPU 的 CUDA 顺序未验证，未用其他卡的显存推断 cuda:0；可用 CUDA_VISIBLE_DEVICES=GPU-UUID 指定", "CUDA device order is unverified; other cards' VRAM is not assumed available to cuda:0. Select by CUDA_VISIBLE_DEVICES=GPU-UUID"),
    "model_unknown": ("无法识别模型大小，未套用固定预算", "Model size is unrecognized; no fixed budget was applied"),
}


def _valid_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0


def _fit(available, budget, resource):
    if budget is None:
        return "unknown", "model_unknown"
    unknown = {"ram": "ram_available_unknown", "vram": "vram_free_unknown", "gpu_host_ram": "gpu_host_ram_unknown"}
    if not _valid_number(available):
        return "unknown", unknown[resource]
    if available < budget[0]:
        return "insufficient", resource + "_low"
    if available < budget[1] * 1.15:
        return "tight", resource + "_tight"
    return "recommended", "resources_fit"


_FIT_ORDER = {"recommended": 0, "tight": 1, "unknown": 2, "insufficient": 3}


def assess_model(model: dict, hardware: dict) -> dict:
    """Compare against AVAILABLE RAM/VRAM, never substituting total capacity."""
    result = deepcopy(model)
    ram = hardware.get("ram_available_gib")
    cpu_fit, cpu_reason = _fit(ram, model.get("cpu_ram_gib"), "ram")
    cuda_supported = "cuda" in model.get("supported_devices", ["cpu", "cuda"])
    options = [{"fit": cpu_fit, "device": "cpu", "reason_codes": [cpu_reason, "cpu_only" if cuda_supported else "backend_cpu_only"]}]
    visible_cuda = [gpu for gpu in hardware.get("gpus", []) if cuda_supported and gpu.get("cuda") and gpu.get("visible_to_process") is not False]
    # The current backend uses torch.device("cuda"), i.e. logical cuda:0.
    # A larger second card cannot justify a recommendation for that backend.
    visible_cuda.sort(key=lambda gpu: gpu.get("cuda_ordinal") if isinstance(gpu.get("cuda_ordinal"), int) else 999)
    for gpu in visible_cuda[:1]:
        free_vram = gpu.get("free_vram_gib") if gpu.get("cuda_order_verified") is not False else None
        vram_fit, vram_reason = _fit(free_vram, model.get("gpu_vram_gib"), "vram")
        host_fit, host_reason = _fit(ram, model.get("gpu_host_ram_gib"), "gpu_host_ram")
        gpu_fit = max((vram_fit, host_fit), key=_FIT_ORDER.get)
        options.append({"fit": gpu_fit, "device": "cuda", "device_index": 0, "gpu_name": gpu.get("name"), "reason_codes": list(dict.fromkeys([vram_reason, host_reason]))})
        if gpu.get("cuda_order_verified") is False:
            options[-1]["reason_codes"].append("cuda_order_unknown")
    choice = min(options, key=lambda item: (_FIT_ORDER[item["fit"]], item["device"] != "cuda"))
    reasons = list(choice["reason_codes"])
    if choice["device"] == "cpu" and visible_cuda:
        gpu_option = min(options[1:], key=lambda item: _FIT_ORDER[item["fit"]])
        reasons.extend(code for code in gpu_option["reason_codes"] if code != "resources_fit")
    if cuda_supported and any(gpu.get("cuda") and gpu.get("visible_to_process") is False for gpu in hardware.get("gpus", [])):
        reasons.append("cuda_hidden")
    reasons = list(dict.fromkeys(reasons))
    result.update(choice)
    result["reason_codes"] = reasons
    result["reason"] = "；".join(REASON_TEXT[code][0] for code in reasons)
    result["reason_en"] = "; ".join(REASON_TEXT[code][1] for code in reasons)
    return result


def recommend_models(hardware: dict) -> list[dict]:
    """Return all variants with explicit, heuristic family and overall choices.

    Memory fit alone does not imply useful CPU latency. CPU Whisper choices are
    capped at tiny/base/small using core count; CUDA favors turbo's throughput
    balance, then a family variant that has estimated allocation headroom.
    """
    models = [assess_model(model, hardware) for model in MODEL_CATALOG]
    for item in models:
        item.update(preferred=False, top_recommendation=False, preference_reason="", preference_reason_en="")
    by_id = {item["id"]: item for item in models}

    def choose(identifiers, device, chinese, english):
        for identifier in identifiers:
            item = by_id[identifier]
            if item["fit"] == "recommended" and item["device"] == device:
                item.update(preferred=True,
                            preference_reason=chinese + "；这是启发式建议，未在本机测速。",
                            preference_reason_en=english + "; this is a heuristic, not a local benchmark.")
                return item
        return None

    whisper = choose(["whisper-" + size for size in ("turbo", "large", "medium", "small", "base", "tiny")], "cuda",
                     "显存和加载主存有估算余量，优先 turbo 平衡吞吐，否则选择可容纳的较大 Whisper",
                     "Available VRAM and loading RAM leave estimated headroom; favor turbo for throughput balance, otherwise a larger fitting Whisper")
    if whisper is None:
        cores = hardware.get("logical_cores")
        cpu_sizes = ["tiny"]
        if _valid_number(cores) and cores >= 4:
            cpu_sizes.insert(0, "base")
        if _valid_number(cores) and cores >= 8:
            cpu_sizes.insert(0, "small")
        whisper = choose(["whisper-" + size for size in cpu_sizes], "cpu",
                         "按逻辑核心数和可用内存保守选择较小 Whisper；内存充足不代表能实时听写",
                         "Conservative smaller Whisper choice based on logical cores and available RAM; sufficient RAM does not imply real-time dictation")
    qwen = choose(["qwen3-1.7b", "qwen3-0.6b"], "cuda",
                  "在 Qwen3 系列中选择可用资源有估算余量的较大版本",
                  "Choose the larger Qwen3 variant whose available resources leave estimated headroom")
    if qwen is None:
        qwen = choose(["qwen3-0.6b"], "cpu", "CPU 优先较小 Qwen3 以降低资源和延迟压力，仍需实测",
                      "Prefer the smaller Qwen3 on CPU to reduce resource and latency pressure; measure locally")
    confucius = choose(["confucius-r2t2"], "cuda", "Confucius 的 GPU 与主存有估算余量，当前后端仍需实测延迟",
                       "Confucius has estimated GPU and host-RAM headroom; current-backend latency still needs measurement")
    if confucius is None:
        confucius = choose(["confucius-r2t2"], "cpu", "Confucius 的 CPU 内存预算有余量，但听写可能较慢",
                           "Confucius has estimated CPU RAM headroom, but dictation may be slow")
    onnx = choose(["whisper-onnx-medium-int8"], "cpu",
                  "Sherpa-ONNX medium 的 INT8 主存预算有余量；只作为该 CPU 后端的候选，量化不保证实时或比小型 Whisper 更快",
                  "Sherpa-ONNX medium INT8 has estimated RAM headroom; it is a candidate for this CPU backend, not a promise of real-time speed or an improvement over smaller Whisper models")
    family_choices = [item for item in (whisper, qwen, confucius, onnx) if item is not None]
    if family_choices:
        top = min(family_choices, key=lambda item: (item["device"] != "cuda", item["family"] != "whisper", item["family"] != "qwen3"))
        top["top_recommendation"] = True
    return sorted(models, key=lambda item: (not item["top_recommendation"], not item["preferred"], _FIT_ORDER[item["fit"]], item["device"] != "cuda", item["cpu_ram_gib"][1]))


def _selection(path):
    if not path:
        return None
    selected = Path(path).expanduser()
    if selected.is_file():
        return selected if selected.suffix.lower() == ".pt" else selected.parent
    if selected.is_dir() and not (selected / "config.json").is_file():
        try:
            children = list(selected.iterdir())
        except OSError:
            return selected
        matches = [child for child in children if (child.is_file() and child.suffix.lower() == ".pt") or (child.is_dir() and (child / "config.json").is_file())]
        if len(matches) == 1:
            return matches[0]
    return selected


def _config(path):
    if path is None or path.is_file():
        return {}
    try:
        with (path / "config.json").open(encoding="utf-8") as stream:
            result = json.loads(stream.read(128_000))
        return result if isinstance(result, dict) else {}
    except (OSError, ValueError, UnicodeError):
        return {}


def _disk_size(path):
    if path is None or not path.exists():
        return None
    try:
        if path.is_file():
            return round(path.stat().st_size / 1024 ** 3, 3)
        # Only model-folder files, no recursive traversal or archive double count.
        return round(sum(item.stat().st_size for item in path.iterdir() if item.is_file() and item.suffix.lower() not in {".zip", ".gz", ".7z"}) / 1024 ** 3, 3)
    except OSError:
        return None


def describe_model(model_type: str, model_path: str = "") -> dict:
    """Identify a variant from small metadata/name, without opening .pt weights."""
    onnx_files = None
    if model_type == "whisper_onnx":
        selected = Path(model_path).expanduser() if model_path else None
        if selected is not None:
            # Use exactly the same precision/pair choice as the ASR loader;
            # never turn an explicit FP32 .onnx selection back into INT8.
            from core.model_paths import resolve_whisper_onnx_files
            onnx_files = resolve_whisper_onnx_files(selected)
    else:
        selected = _selection(model_path)
    config = _config(selected)
    name = selected.name.lower() if selected else ""
    identifier = None
    if model_type == "whisper_onnx":
        if onnx_files and onnx_files["variant"] in {"medium", "medium.en"}:
            identifier = "whisper-onnx-medium-" + onnx_files["precision"]
    elif model_type == "confucius":
        identifier = "confucius-r2t2"
    elif model_type == "qwen3":
        thinker = config.get("thinker_config")
        text = thinker.get("text_config") if isinstance(thinker, dict) else None
        hidden = text.get("hidden_size") if isinstance(text, dict) else None
        if hidden == 1024:
            identifier = "qwen3-0.6b"
        elif hidden == 2048:
            identifier = "qwen3-1.7b"
        elif "0.6b" in name:
            identifier = "qwen3-0.6b"
        elif "1.7b" in name:
            identifier = "qwen3-1.7b"
    elif model_type == "whisper":
        hidden, layers = config.get("d_model"), config.get("decoder_layers")
        variants = {(384, 4): "tiny", (512, 6): "base", (768, 12): "small", (1024, 24): "medium", (1280, 32): "large", (1280, 4): "turbo"}
        variant = variants.get((hidden, layers))
        if variant is None:
            variant = next((size for size in ("turbo", "tiny", "base", "small", "medium", "large") if re.search(r"(?<![a-z])" + size + r"(?![a-z])", name)), None)
        if variant:
            identifier = "whisper-" + variant
    result = next((deepcopy(model) for model in MODEL_CATALOG if model["id"] == identifier), None)
    if result is None:
        result = {"id": "unknown", "family": model_type, "name": selected.name if selected else model_type,
                  "parameters": None, "cpu_ram_gib": None, "gpu_vram_gib": None, "gpu_host_ram_gib": None,
                  "official_vram_gb": None, "official_speed_relative": None,
                  "speed_note": "未识别具体模型大小，请查看模型官方说明并进行本机测试。",
                  "speed_note_en": "The exact model size is unrecognized. Check its official documentation and benchmark locally.",
                  "budget_note": _BUDGET_ZH, "budget_note_en": _BUDGET_EN,
                  "sources": [], "identified": False, "estimate": True}
    if model_type == "whisper" and result["identified"]:
        result["budget_note"] += " 原生 .pt 即使用 GPU，也先在 CPU 加载，需同时为 FP32 模型与检查点留主存。"
        result["budget_note_en"] += " Native .pt loading stages on CPU even with a GPU; host RAM must hold the FP32 model and checkpoint together."
    result["disk_gib"] = _disk_size(selected)
    if model_type == "whisper_onnx":
        result.update(supported_devices=["cpu"], backend="sherpa-onnx")
        if not result["identified"]:
            result["budget_note"] = "此 Sherpa-ONNX 模型未识别到已有资源档位，未套用原生 .pt 或其他精度预算。当前适配器仅使用 CPU；请按所选模型文件进行本机测量。"
            result["budget_note_en"] = "No resource profile is available for this Sherpa-ONNX variant. Native .pt or other-precision budgets were not reused. This adapter is CPU-only; measure the selected model locally."
        if onnx_files:
            files = [onnx_files[key] for key in ("encoder", "decoder", "tokens")]
            result["precision"] = onnx_files["precision"]
            result["selected_files"] = [str(path) for path in files]
            result["folder_disk_gib"] = _disk_size(files[0].parent)
            try:
                result["disk_gib"] = round(sum(path.stat().st_size for path in files) / 1024 ** 3, 3)
            except OSError:
                result["disk_gib"] = None
    result["resolved_path"] = str(selected) if selected else None
    return result
