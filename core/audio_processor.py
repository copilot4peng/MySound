"""Normalize media with FFmpeg and read bounded, Silero-guided audio chunks."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
from typing import Any, Callable, Iterator
import wave

log = logging.getLogger(__name__)


class AudioProcessingCancelled(InterruptedError):
    """The user cancelled media preparation."""


@dataclass(frozen=True)
class AudioChunk:
    audio: Any
    start: float
    end: float
    sample_rate: int = 16000
    id: int | str | None = None
    reason: str = "end"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def probe_media(path: Path) -> dict:
    """Return cheap source facts; ffprobe failures never block conversion."""
    try:
        size_bytes = path.stat().st_size
    except OSError as exc:
        size_bytes = None
    info = {"name": path.name, "path": str(path), "size_bytes": size_bytes,
            "format": None, "codec": None, "duration": None, "sample_rate": None,
            "channels": None}
    executable = shutil.which("ffprobe")
    if not executable:
        info["probe_error"] = "ffprobe unavailable"
        return info
    command = [executable, "-v", "error", "-show_format", "-show_streams", "-of", "json", str(path)]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=8, check=False)
        payload = json.loads(completed.stdout or "{}")
        fmt = payload.get("format") or {}
        streams = payload.get("streams") or []
        audio = next((item for item in streams if item.get("codec_type") == "audio"), streams[0] if streams else {})
        info.update(format=fmt.get("format_name"), codec=audio.get("codec_name"),
                    duration=_number(fmt.get("duration")), sample_rate=_number(audio.get("sample_rate")),
                    channels=audio.get("channels"))
        if completed.returncode:
            info["probe_error"] = (completed.stderr or "ffprobe failed").strip()[:500]
    except Exception as exc:
        info["probe_error"] = str(exc)[:500]
        log.info("ffprobe unavailable for %s: %s", path, exc)
    return info


def _number(value):
    try:
        number = float(value)
        return number if number >= 0 else None
    except (TypeError, ValueError):
        return None


def _choose_boundary(speech: list[dict], length: int, target: int, maximum: int) -> int:
    """Choose a pause near the target, falling back to a hard maximum.

    All positions are samples. Keeping the complete interval (including pauses)
    means that chunk timestamps continue to match the original media.
    """
    if length <= target:
        return length
    pauses = []
    for index, segment in enumerate(speech):
        end = int(segment["end"])
        next_start = int(speech[index + 1]["start"]) if index + 1 < len(speech) else length
        if next_start > end:
            pauses.append((end + next_start) // 2)
    after_target = [point for point in pauses if target <= point <= maximum]
    if after_target:
        return min(after_target)
    before_target = [point for point in pauses if target // 2 <= point < target]
    if before_target and length >= maximum:
        return max(before_target)
    return min(length, maximum)


class AudioProcessor:
    """Decode on disk; keep at most ``max_seconds`` of PCM in memory.

    ``silero-vad`` ships the VAD weights, so this code never uses torch.hub or
    downloads a model. ASR weights are managed independently by ModelManager.
    """

    sample_rate = 16000

    def __init__(self, target_seconds: float = 30, max_seconds: float = 60):
        if target_seconds <= 0 or max_seconds < target_seconds:
            raise ValueError("max_seconds must be >= target_seconds > 0")
        self.target_seconds = target_seconds
        self.max_seconds = max_seconds
        self._vad = None
        self._get_timestamps = None

    def _load_vad(self):
        if self._vad is None:
            try:
                from silero_vad import get_speech_timestamps, load_silero_vad
            except ImportError as exc:
                raise RuntimeError("缺少 Silero-VAD，请先安装 requirements.txt 中的依赖。") from exc
            self._vad = load_silero_vad()
            self._get_timestamps = get_speech_timestamps

    @staticmethod
    def _cancelled(cancel_event) -> bool:
        return cancel_event is not None and cancel_event.is_set()

    def _normalize(self, source: Path, destination: Path, cancel_event, on_status):
        executable = shutil.which("ffmpeg")
        if not executable:
            raise RuntimeError("未找到 FFmpeg，请运行 sudo apt install ffmpeg。")
        if self._cancelled(cancel_event):
            raise AudioProcessingCancelled("转写已取消")
        if on_status:
            on_status("正在使用 FFmpeg 转换为 16 kHz 单声道音频…")
        log.info("normalize media %s -> 16 kHz mono", source)
        command = [executable, "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
                   "-i", str(source), "-vn", "-ac", "1", "-ar", str(self.sample_rate),
                   "-c:a", "pcm_s16le", str(destination)]
        with tempfile.TemporaryFile() as errors:
            process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=errors)
            try:
                while process.poll() is None:
                    if self._cancelled(cancel_event):
                        process.terminate()
                        try:
                            process.wait(timeout=2)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()
                        raise AudioProcessingCancelled("转写已取消")
                    time.sleep(0.05)
                if process.returncode:
                    errors.seek(0)
                    detail = errors.read(4096).decode("utf-8", errors="replace").strip()
                    raise RuntimeError(f"FFmpeg 无法读取媒体文件：{detail}")
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait()

    @contextmanager
    def process(self, path, cancel_event=None, on_status: Callable[[str], None] | None = None,
                on_event: Callable[[dict], None] | None = None, metadata: dict | None = None,
                metadata_already_emitted: bool = False
                ) -> Iterator[Iterator[AudioChunk]]:
        """Yield a chunk iterator; close WAV handles and remove temporary files.

        ``metadata`` may be supplied by the controller after an early probe,
        avoiding a second ffprobe invocation while still publishing the exact
        normalized duration once decoding has completed.

        Use ``with processor.process(path) as chunks: for chunk in chunks: ...``.
        The WAV is intentionally disk-backed so multi-hour input stays bounded.
        """
        source = Path(path).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"音频文件不存在：{source}")
        metadata = dict(metadata) if metadata is not None else probe_media(source)
        if on_event and not metadata_already_emitted:
            on_event({"kind": "media_info", "metadata": dict(metadata), "timestamp": _utc_now()})
        with tempfile.TemporaryDirectory(prefix="mysound-audio-") as directory:
            decoded = Path(directory) / "normalized.wav"
            self._normalize(source, decoded, cancel_event, on_status)
            try:
                with wave.open(str(decoded), "rb") as normalized:
                    metadata["normalized_duration"] = normalized.getnframes() / normalized.getframerate()
            except Exception:
                metadata["normalized_duration"] = None
            if on_event:
                on_event({"kind": "media_info", "metadata": dict(metadata), "timestamp": _utc_now()})
            iterator = self._chunks(decoded, cancel_event, on_status, on_event)
            try:
                yield iterator
            finally:
                iterator.close()

    def _chunks(self, path: Path, cancel_event, on_status, on_event=None) -> Iterator[AudioChunk]:
        import numpy as np
        import torch

        self._load_vad()
        if on_status:
            on_status("正在进行 Silero-VAD 语音检测与分段转写…")
        log.info("VAD chunking normalized media %s", path)
        maximum = int(self.max_seconds * self.sample_rate)
        target = int(self.target_seconds * self.sample_rate)
        with wave.open(str(path), "rb") as audio_file:
            total = audio_file.getnframes()
            position = 0
            speech_id = 0
            skipped_id = 0
            while position < total and not self._cancelled(cancel_event):
                audio_file.setpos(position)
                pcm = audio_file.readframes(min(maximum, total - position))
                samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
                if not len(samples):
                    break
                speech = self._get_timestamps(
                    torch.from_numpy(samples), self._vad,
                    sampling_rate=self.sample_rate, min_silence_duration_ms=400,
                    speech_pad_ms=30, return_seconds=False,
                )
                if self._cancelled(cancel_event):
                    break
                boundary = _choose_boundary(speech, len(samples), target, maximum)
                # Silence-only intervals advance the clock without invoking ASR.
                has_speech = any(int(segment["start"]) < boundary for segment in speech)
                reason = "end" if position + boundary >= total else ("pause" if boundary < maximum else "limit")
                if not has_speech:
                    skipped_id += 1
                    chunk = {"id": f"silence-{skipped_id}", "start": position / self.sample_rate,
                             "end": (position + boundary) / self.sample_rate, "status": "skipped", "reason": "silence"}
                    if on_event:
                        on_event({"kind": "chunk", "chunk": chunk, "timestamp": _utc_now()})
                else:
                    speech_id += 1
                    chunk = {"id": speech_id, "start": position / self.sample_rate,
                             "end": (position + boundary) / self.sample_rate, "status": "pending", "reason": reason}
                    if on_event:
                        on_event({"kind": "chunk", "chunk": chunk, "timestamp": _utc_now()})
                    yield AudioChunk(samples[:boundary], position / self.sample_rate,
                                     (position + boundary) / self.sample_rate, self.sample_rate,
                                     id=speech_id, reason=reason)
                position += boundary
