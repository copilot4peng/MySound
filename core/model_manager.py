"""Local-only ASR adapters. Heavy ML libraries are imported only on model load.

Qwen3 and Confucius use the official qwen-asr Transformers backend. Confucius'
official R2T2ASRModel.transcribe delegates to this same implementation; its
special streaming methods require vLLM, so the desktop app instead supplies
short microphone utterances to the common transcribe interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import gc
import importlib
import logging
import os
from pathlib import Path
import sys
import threading
import traceback
from typing import TYPE_CHECKING, Any, Callable

from .model_paths import resolve_model_path, resolve_whisper_onnx_files, scan_models

log = logging.getLogger(__name__)

if TYPE_CHECKING:
    import numpy as np


MODEL_LABELS = {
    "whisper": "OpenAI Whisper",
    "whisper_onnx": "Sherpa-ONNX Whisper",
    "qwen3": "Qwen3-ASR",
    "confucius": "Confucius4-R2T2",
}


@dataclass(frozen=True)
class TranscriptSegment:
    """A transcript span with times relative to the input audio, in seconds."""

    start: float
    end: float
    text: str


def _dependency(module: str, install: str) -> Any:
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise RuntimeError(
            f"无法导入 {module}：{exc}。请在运行程序的 Python 环境执行：{install}"
        ) from exc


def _offline() -> None:
    # Set before importing Hugging Face libraries: neither models nor processors
    # are allowed to download missing files during desktop inference.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"


def _device(torch: Any, requested: str) -> str:
    if requested == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("已选择 CUDA，但 PyTorch 未检测到可用的 NVIDIA GPU。请选择 CPU 或安装对应的 CUDA 版 PyTorch。")
    return requested


def _audio(audio: Any, sample_rate: int) -> Any:
    if sample_rate != 16000:
        raise ValueError("模型输入必须为 16000 Hz；请先通过 AudioProcessor 转换音频。")
    np = _dependency("numpy", "pip install numpy")
    result = np.asarray(audio, dtype=np.float32)
    if result.ndim != 1:
        raise ValueError("模型输入必须为一维单声道 PCM 数组。")
    return np.ascontiguousarray(result)


def _language(language: str | None) -> str | None:
    if not language or language.lower() in {"auto", "自动", "自动检测"}:
        return None
    return {
        "zh": "Chinese", "en": "English", "ja": "Japanese", "ko": "Korean",
        "yue": "Cantonese", "de": "German", "fr": "French", "es": "Spanish",
        "ru": "Russian", "中文": "Chinese", "英语": "English",
    }.get(language.lower(), language)


def _is_cuda_oom(error: Exception) -> bool:
    torch = sys.modules.get("torch")
    oom_type = getattr(getattr(torch, "cuda", None), "OutOfMemoryError", None)
    if isinstance(oom_type, type) and isinstance(error, oom_type):
        return True
    message = str(error).lower()
    return isinstance(error, RuntimeError) and "cuda" in message and "out of memory" in message


def _release_exception_frames(error: Exception) -> None:
    """Drop tensor references in failed decoder/loader stack frames before retry."""
    seen = set()
    pending = [error]
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        if current.__context__ is not None:
            pending.append(current.__context__)
        if current.__cause__ is not None:
            pending.append(current.__cause__)
        traceback.clear_frames(current.__traceback__)
        current.__traceback__ = None
        current.__context__ = None
        current.__cause__ = None


CUDA_OOM_MESSAGE = "CUDA 显存不足。请选择自动模式（显存不足时改用 CPU）、直接使用 CPU，或选择更小的模型。"

# Whisper's original 99 language tokens; Cantonese was added in large-v3.
# https://github.com/openai/whisper/blob/main/whisper/tokenizer.py
# Keep this local: importing openai-whisper just to validate ONNX input would
# unnecessarily import PyTorch. Sherpa exits the process on an unknown token.
_WHISPER_LANGUAGE_CODES = set("""
en zh de es ru ko fr ja pt tr pl ca nl ar sv it id hi fi vi he uk el ms cs ro da
hu ta no th ur hr bg lt la mi ml cy sk te fa lv bn sr az sl kn et mk br eu is
hy ne mn bs kk sq sw gl mr pa si km sn yo so af oc ka be tg sd gu am yi lo uz
fo ht ps tk nn mt sa lb my bo tl mg as tt haw ln ha ba jw su
""".split())


def _whisper_onnx_language(language: str | None, variant: str) -> str:
    normalized = _language(language.strip() if language else None)
    code = {"chinese": "zh", "english": "en", "japanese": "ja", "korean": "ko",
            "cantonese": "yue", "german": "de", "french": "fr", "spanish": "es", "russian": "ru"}.get(
                normalized.lower(), normalized.lower()) if normalized else ""
    variant = variant.lower()
    supported = _WHISPER_LANGUAGE_CODES | ({"yue"} if variant in {"large-v3", "large-v3-turbo", "turbo"} else set())
    if variant.endswith(".en"):
        supported = {"en"}
    if code and code not in supported:
        error = ValueError(f"此 Whisper ONNX 型号（{variant or 'unknown'}）不支持语言 {language!r}。请选择该型号支持的语言或自动识别。")
        error.message_en = f"This Whisper ONNX variant ({variant or 'unknown'}) does not support language {language!r}. Select a supported language or automatic detection."
        raise error
    return code


class BaseASRModel(ABC):
    """Uniform, synchronous API; callers run load/transcribe on worker threads."""

    model_type = "base"

    def __init__(self, model_path: str, device: str = "auto") -> None:
        self.model_path = str(Path(model_path).expanduser().resolve())
        self.device = device
        self._backend: Any = None
        self._lock = threading.RLock()

    @property
    def name(self) -> str:
        return MODEL_LABELS.get(self.model_type, self.model_type)

    @abstractmethod
    def load(self) -> None:
        """Load a model from the configured local path."""

    @abstractmethod
    def transcribe(
        self, audio: np.ndarray, sample_rate: int = 16000, language: str | None = None
    ) -> list[TranscriptSegment]:
        """Recognize audio. Timestamp quality is backend dependent."""

    def unload(self) -> None:
        with self._lock:
            self._backend = None
            gc.collect()
            torch = sys.modules.get("torch")
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _require_loaded(self) -> None:
        if self._backend is None:
            raise RuntimeError("请先在设置中选择并加载本地模型。")


class WhisperASRModel(BaseASRModel):
    """Support native openai-whisper .pt weights and HF Whisper folders."""

    model_type = "whisper"

    def __init__(self, model_path: str, device: str = "auto") -> None:
        super().__init__(model_path, device)
        self._native = False

    def load(self) -> None:
        with self._lock:
            path = resolve_model_path(self.model_path, self.model_type)
            self.model_path = str(path)
            _offline()
            torch = _dependency("torch", "pip install torch")
            self.device = _device(torch, self.device)
            self._native = path.is_file()
            if self._native:
                if path.suffix.lower() != ".pt":
                    raise ValueError("Whisper 原生模型文件应为 .pt；其他格式请选择完整模型目录。")
                whisper = _dependency("whisper", "pip install openai-whisper")
                # whisper.load_model(device='cuda') maps the checkpoint to GPU
                # and creates an FP32 model before transferring it, producing a
                # large temporary VRAM peak. Release the checkpoint on CPU,
                # then convert weights before the first GPU transfer.
                self._backend = whisper.load_model(str(path), device="cpu")
                if self.device.startswith("cuda"):
                    self._backend.half()
                    # Whisper's LayerNorm explicitly promotes activations to
                    # FP32, so keep its small normalization parameters FP32 as
                    # well; all large linear/embedding weights remain FP16.
                    for module in self._backend.modules():
                        if isinstance(module, torch.nn.LayerNorm):
                            module.float()
                    self._backend.to(self.device)
                self._backend.eval()
            else:
                transformers = _dependency("transformers", "pip install transformers==4.57.6 accelerate")
                dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
                model = transformers.AutoModelForSpeechSeq2Seq.from_pretrained(
                    str(path), torch_dtype=dtype, local_files_only=True,
                    low_cpu_mem_usage=True, trust_remote_code=False,
                ).to(self.device)
                model.eval()
                processor = transformers.AutoProcessor.from_pretrained(
                    str(path), local_files_only=True, trust_remote_code=False,
                )
                self._backend = transformers.pipeline(
                    "automatic-speech-recognition", model=model,
                    tokenizer=processor.tokenizer, feature_extractor=processor.feature_extractor,
                    torch_dtype=dtype, device=self.device, chunk_length_s=30,
                )

    def transcribe(
        self, audio: np.ndarray, sample_rate: int = 16000, language: str | None = None
    ) -> list[TranscriptSegment]:
        with self._lock:
            self._require_loaded()
            waveform = _audio(audio, sample_rate)
            duration = len(waveform) / sample_rate
            if not duration:
                return []
            language = _language(language)
            if self._native:
                result = self._backend.transcribe(
                    waveform, language=language.lower() if language else None,
                    fp16=self.device.startswith("cuda"), verbose=False,
                    condition_on_previous_text=False,
                )
                segments = [
                    TranscriptSegment(float(part["start"]), min(float(part["end"]), duration), part["text"].strip())
                    for part in result.get("segments", []) if part.get("text", "").strip()
                ]
            else:
                options = {"task": "transcribe"}
                if language:
                    options["language"] = language.lower()
                result = self._backend(
                    {"raw": waveform, "sampling_rate": sample_rate},
                    return_timestamps=True, generate_kwargs=options,
                )
                segments = []
                for part in result.get("chunks", []):
                    start, end = part.get("timestamp", (0, duration))
                    text = part.get("text", "").strip()
                    if text:
                        segments.append(TranscriptSegment(float(start or 0), min(float(end if end is not None else duration), duration), text))
            if not segments and result.get("text", "").strip():
                segments = [TranscriptSegment(0, duration, result["text"].strip())]
            return segments


class WhisperOnnxASRModel(BaseASRModel):
    """Local Sherpa-ONNX Whisper, with explicit CPU execution and real spans.

    Official API: sherpa_onnx.OfflineRecognizer.from_whisper (v1.13.8).
    That decoder clips at 2950 feature frames, so keep inputs below 29.5 s.
    Returned spans describe the actual input chunks, not word alignments.
    """

    model_type = "whisper_onnx"
    chunk_seconds = 29

    def __init__(self, model_path: str, device: str = "auto") -> None:
        super().__init__(model_path, device)
        self._variant = ""

    def load(self) -> None:
        with self._lock:
            if self.device not in {"auto", "cpu"}:
                error = RuntimeError("Sherpa-ONNX Whisper 当前适配使用 CPU。请选择 CPU 或自动模式；此安装未启用经过验证的 CUDA 后端。")
                error.message_en = "This Sherpa-ONNX Whisper adapter uses CPU. Select CPU or Auto; a verified CUDA backend is not enabled for this installation."
                raise error
            selected = resolve_model_path(self.model_path, self.model_type)
            files = resolve_whisper_onnx_files(selected)
            self._variant = files["variant"]
            self.model_path = str(selected)
            self.device = "cpu"
            sdk = _dependency("sherpa_onnx", "pip install sherpa-onnx==1.13.8")
            self._backend = sdk.OfflineRecognizer.from_whisper(
                encoder=str(files["encoder"]), decoder=str(files["decoder"]),
                tokens=str(files["tokens"]), language="", task="transcribe",
                num_threads=max(1, min(4, os.cpu_count() or 1)),
                decoding_method="greedy_search", provider="cpu", debug=False,
            )

    def transcribe(
        self, audio: np.ndarray, sample_rate: int = 16000, language: str | None = None
    ) -> list[TranscriptSegment]:
        with self._lock:
            self._require_loaded()
            waveform = _audio(audio, sample_rate)
            if not len(waveform):
                return []
            language_code = _whisper_onnx_language(language, self._variant)
            # The official Python wrapper exposes the native recognizer's
            # set_config. Whisper supports language/task changes without
            # recreating ONNX sessions or holding two models in host RAM.
            config = self._backend.config
            config.model_config.whisper.language = language_code
            self._backend.recognizer.set_config(config)
            segments = []
            chunk_samples = self.chunk_seconds * sample_rate
            for offset in range(0, len(waveform), chunk_samples):
                chunk = waveform[offset:offset + chunk_samples]
                stream = self._backend.create_stream()
                stream.accept_waveform(sample_rate, chunk)
                self._backend.decode_stream(stream)
                text = stream.result.text.strip()
                if text:
                    segments.append(TranscriptSegment(offset / sample_rate, (offset + len(chunk)) / sample_rate, text))
            return segments

    def unload(self) -> None:
        # The ONNX CPU adapter has no PyTorch/CUDA resources to inspect.
        with self._lock:
            self._backend = None
            gc.collect()


class Qwen3ASRModel(BaseASRModel):
    """Official qwen-asr Transformers inference, without a forced aligner."""

    model_type = "qwen3"

    def load(self) -> None:
        with self._lock:
            path = resolve_model_path(self.model_path, self.model_type)
            self.model_path = str(path)
            _offline()
            torch = _dependency("torch", "pip install torch")
            qwen = _dependency("qwen_asr", "pip install qwen-asr==0.0.6")
            self.device = _device(torch, self.device)
            dtype = torch.float32
            if self.device.startswith("cuda"):
                dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            self._backend = qwen.Qwen3ASRModel.from_pretrained(
                str(path), dtype=dtype, device_map=self.device,
                local_files_only=True, trust_remote_code=False,
                max_inference_batch_size=1, max_new_tokens=1024,
            )
            self._backend.model.eval()

    def transcribe(
        self, audio: np.ndarray, sample_rate: int = 16000, language: str | None = None
    ) -> list[TranscriptSegment]:
        with self._lock:
            self._require_loaded()
            waveform = _audio(audio, sample_rate)
            if not len(waveform):
                return []
            results = self._backend.transcribe(
                audio=(waveform, sample_rate), language=_language(language),
                return_time_stamps=False,
            )
            text = "\n".join(result.text.strip() for result in results if result.text.strip())
            # Without an external forced-aligner checkpoint, only the audio
            # chunk's real bounds are known. Do not invent word timestamps.
            return [TranscriptSegment(0, len(waveform) / sample_rate, text)] if text else []


class ConfuciusASRModel(Qwen3ASRModel):
    """Confucius4-R2T2's Qwen3 architecture using its official offline path.

    Upstream r2t2.R2T2ASRModel inherits from Qwen3ASRModel and delegates its
    transcribe() unchanged. Loading the Confucius weights through that shared
    backend avoids bringing the optional vLLM server stack into the desktop app.
    https://github.com/netease-youdao/Confucius4-R2T2/blob/master/r2t2/r2t2_asr.py
    """

    model_type = "confucius"


class ModelManager:
    """Keep one model resident, releasing it before a model switch."""

    adapters = {
        "whisper": WhisperASRModel,
        "whisper_onnx": WhisperOnnxASRModel,
        "qwen3": Qwen3ASRModel,
        "confucius": ConfuciusASRModel,
    }

    def __init__(self, on_status: Callable[[str], None] | None = None) -> None:
        self.model: BaseASRModel | None = None
        self._selection: tuple[str, str, str] | None = None
        self._lock = threading.RLock()
        self.on_status = on_status or (lambda _message: None)

    def load(self, model_type: str, model_path: str, device: str = "auto") -> BaseASRModel:
        with self._lock:
            if model_type not in self.adapters:
                raise ValueError(f"不支持的模型类型：{model_type}")
            resolved = str(resolve_model_path(model_path, model_type))
            selection = (model_type, resolved, device)
            if self.model is not None and selection == self._selection:
                return self.model
            self.unload()
            model = self.adapters[model_type](resolved, device)
            retry_cpu = False
            try:
                model.load()
            except Exception as error:
                is_cuda_oom = _is_cuda_oom(error)
                if is_cuda_oom:
                    log.warning("CUDA OOM while loading %s on %s", resolved, model.device, exc_info=(type(error), error, error.__traceback__))
                    _release_exception_frames(error)
                model.unload()
                if not is_cuda_oom:
                    raise
                if device != "auto":
                    raise RuntimeError(CUDA_OOM_MESSAGE) from None
                retry_cpu = True
            # Retry outside the except block: Python's active exception must
            # not retain failed GPU loader frames or tensors during CPU load.
            if retry_cpu:
                self.on_status("CUDA 显存不足，自动改用 CPU 加载模型；转写速度会变慢。")
                model = self.adapters[model_type](resolved, "cpu")
                try:
                    model.load()
                except Exception:
                    model.unload()
                    raise
            self.model = model
            self._selection = selection
            return model

    def transcribe(
        self, audio: np.ndarray, sample_rate: int = 16000, language: str | None = None
    ) -> list[TranscriptSegment]:
        """Recognize using the resident model, retrying auto-mode GPU OOM once."""
        with self._lock:
            if self.model is None or self._selection is None:
                raise RuntimeError("请先在设置中选择并加载本地模型。")
            selection = self._selection
            try:
                return self.model.transcribe(audio, sample_rate, language)
            except Exception as error:
                if not _is_cuda_oom(error):
                    raise
                log.warning("CUDA OOM while transcribing with %s on %s", selection[1], self.model.device, exc_info=(type(error), error, error.__traceback__))
                _release_exception_frames(error)
                retry_cpu = selection[2] == "auto" and self.model.device.startswith("cuda")
                self.unload()
                if not retry_cpu:
                    raise RuntimeError(CUDA_OOM_MESSAGE) from None
            self.on_status("转写时 CUDA 显存不足，自动改用 CPU 重试当前音频；后续音频继续使用 CPU。")
            # Explicit CPU load cannot trigger the automatic GPU retry path.
            self.load(selection[0], selection[1], "cpu")
            self._selection = selection
            return self.model.transcribe(audio, sample_rate, language)

    def unload(self) -> None:
        with self._lock:
            if self.model is not None:
                self.model.unload()
            self.model = None
            self._selection = None

    scan_models = staticmethod(scan_models)
