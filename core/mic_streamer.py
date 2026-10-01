"""Bounded microphone capture and utterance-based incremental recognition.

Partial results replace the current utterance. They are not native token-streaming
results: the selected local ASR model re-decodes the utterance every few seconds.
Microphone endpointing uses configurable RMS energy to avoid concurrent VAD/model
state, while imported media uses Silero-VAD in audio_processor.py.
"""

from __future__ import annotations

from collections import deque
import logging
import math
import queue
import threading
from typing import Callable


log = logging.getLogger(__name__)


def _invalid_sample_rate(error: Exception) -> bool:
    """Retry only PortAudio's invalid-rate failure, never permission/device errors."""
    return -9997 in error.args or "-9997" in str(error) or "invalid sample rate" in str(error).lower()


def _capture_error(message, message_en, attempts=()):
    error = RuntimeError(message)
    error.message_en = message_en
    error.details = "\n".join(f"{rate} Hz ({phase}): {failure!r}" for rate, phase, failure in attempts)
    if error.details and hasattr(error, "add_note"):
        error.add_note(error.details)
    return error


class MicStreamer:
    sample_rate = 16000
    capture_block_seconds = 0.02

    def __init__(self, model, on_partial: Callable[[str], None] | None = None,
                 on_segment: Callable | None = None,
                 on_status: Callable[[str], None] | None = None,
                 on_error: Callable[[Exception], None] | None = None,
                 on_done: Callable[[], None] | None = None,
                 language=None, device=None, block_seconds=2.0,
                 silence_seconds=0.7, energy_threshold=0.008,
                 max_utterance_seconds=15.0, queue_seconds=10.0,
                 on_capture_info: Callable[[dict], None] | None = None):
        if min(block_seconds, silence_seconds, max_utterance_seconds, queue_seconds) <= 0:
            raise ValueError("Audio durations must be positive")
        self.model = model
        self.on_partial = on_partial or (lambda text: None)
        self.on_segment = on_segment or (lambda segment: None)
        self.on_status = on_status or (lambda message: None)
        self.on_error = on_error or (lambda error: None)
        self.on_done = on_done or (lambda: None)
        self.on_capture_info = on_capture_info or (lambda info: None)
        self.language = language or None
        self.device = device
        self.block_seconds = block_seconds
        self.silence_seconds = silence_seconds
        self.energy_threshold = energy_threshold
        self.max_utterance_seconds = max_utterance_seconds
        self._queue = queue.Queue(maxsize=max(1, math.ceil(queue_seconds / self.capture_block_seconds)))
        self._stop_requested = threading.Event()
        self._thread = None
        self._stream = None
        self._pending_error = None
        self._sample_count = 0
        self.capture_info = None
        self.capture_sample_rate = self.sample_rate
        self._np = None
        self._sd = None

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self):
        """Start capture; callbacks run on the worker, never the Tk thread."""
        if self.is_running:
            raise RuntimeError("麦克风正在运行")
        try:
            import numpy as np
            import sounddevice as sd
        except ImportError as exc:
            raise RuntimeError("缺少麦克风依赖，请安装 numpy、sounddevice 和 PortAudio。") from exc
        self._np, self._sd = np, sd
        self._stop_requested.clear()
        self._pending_error = None
        self._sample_count = 0
        while not self._queue.empty():
            self._queue.get_nowait()
        try:
            self._open_capture()
            self._thread = threading.Thread(target=self._run, name="mysound-microphone", daemon=True)
            self._thread.start()
        except Exception:
            self._close_capture()
            raise

    def _close_capture(self):
        stream, self._stream = self._stream, None
        if stream is None:
            return
        # stop() can fail after a failed start(); close() must still run.
        for method in ("stop", "close"):
            try:
                getattr(stream, method)()
            except Exception:
                log.debug("Microphone cleanup %s failed", method, exc_info=True)

    def _open_capture(self):
        sd = self._sd
        try:
            info = sd.query_devices(self.device, "input")
            if info.get("max_input_channels", 0) < 1:
                raise ValueError("The selected device has no input channels")
        except Exception as cause:
            raise _capture_error("麦克风不可用。请检查输入设备连接、权限和系统声音设置。",
                                 "Microphone unavailable. Check its connection, permissions and system sound settings.") from cause
        rates = [self.sample_rate]
        try:
            default_rate = float(info.get("default_samplerate", 0))
            if math.isfinite(default_rate) and default_rate > 0:
                rates.append(round(default_rate))
        except (TypeError, ValueError):
            pass
        rates = list(dict.fromkeys(rates + [48000, 44100]))
        attempts = []
        for rate in rates:
            phase = "check"
            try:
                sd.check_input_settings(device=self.device, channels=1, dtype="float32", samplerate=rate)
                phase = "open"
                self.capture_sample_rate = rate
                self._stream = sd.InputStream(
                    samplerate=rate, channels=1, dtype="float32",
                    blocksize=round(self.capture_block_seconds * rate),
                    device=self.device, callback=self._capture,
                )
                # PortAudio reports the actual stream rate after opening.
                self.capture_sample_rate = round(float(self._stream.samplerate))
                self._stream.start()
            except Exception as cause:
                attempts.append((rate, phase, cause))
                self._close_capture()
                if not _invalid_sample_rate(cause):
                    raise _capture_error("无法启动麦克风。请检查设备是否被占用及录音权限。",
                                         "Cannot start the microphone. Check device availability and recording permissions.", attempts) from cause
                continue
            device_id = self.device
            if device_id is None:
                device_id = getattr(self._stream, "device", None)
            self.capture_info = {"device": device_id, "name": info.get("name", str(device_id)),
                                 "capture_sample_rate": self.capture_sample_rate,
                                 "sample_rate": self.sample_rate, "channels": 1}
            return
        attempted = ", ".join(str(rate) for rate in rates)
        raise _capture_error(f"麦克风不支持已尝试的采样率（{attempted} Hz）。请更换输入设备或检查系统音频设置。",
                             f"The microphone rejected all attempted sample rates ({attempted} Hz). Choose another input device or check system audio settings.", attempts) from attempts[-1][2]

    def stop(self):
        """Request a nonblocking stop; queued audio and the last utterance drain."""
        self._stop_requested.set()

    def _capture(self, indata, frames, time_info, status):
        if self._stop_requested.is_set():
            raise self._sd.CallbackStop()
        if status:
            self._pending_error = _capture_error(
                f"麦克风采集发生丢帧：{status}；录音已停止。",
                f"Microphone input overflowed or dropped frames ({status}); recording stopped.",
                [(self.capture_sample_rate, "callback", status)],
            )
            self._stop_requested.set()
            raise self._sd.CallbackStop()
        start = self._sample_count
        self._sample_count += frames
        try:
            self._queue.put_nowait((start, indata[:, 0].copy()))
        except queue.Full:
            self._pending_error = _capture_error(
                "模型处理速度低于录音速度，音频队列已满；录音已停止，请使用较小模型。",
                "The recognition worker fell behind and the audio queue filled; recording stopped. Try a smaller model.",
                [(self.capture_sample_rate, "callback", "queue full")],
            )
            self._stop_requested.set()
            raise self._sd.CallbackStop()

    def _decode(self, parts, start_sample, final):
        from core.model_manager import TranscriptSegment

        samples = self._np.concatenate(parts)
        duration = len(samples) / self.capture_sample_rate
        offset = start_sample / self.capture_sample_rate
        if self.capture_sample_rate != self.sample_rate:
            # Resample each complete utterance (including partial re-decodes),
            # never independent callback blocks: this avoids boundary artifacts
            # and accumulated fractional-sample drift at rates such as 44.1 kHz.
            try:
                from scipy.signal import resample_poly
            except ImportError as cause:
                error = RuntimeError("缺少重采样依赖 scipy，无法将麦克风音频转换为 16 kHz。请安装 scipy。")
                error.message_en = "The scipy dependency is required to resample microphone audio to 16 kHz. Install scipy."
                raise error from cause
            divisor = math.gcd(self.sample_rate, self.capture_sample_rate)
            samples = self._np.ascontiguousarray(
                resample_poly(samples, self.sample_rate // divisor,
                              self.capture_sample_rate // divisor).astype(self._np.float32, copy=False)
            )
        segments = self.model.transcribe(samples, self.sample_rate, self.language)
        if not final:
            self.on_partial(" ".join(segment.text.strip() for segment in segments if segment.text.strip()))
            return
        for segment in segments:
            if not segment.text.strip():
                continue
            start = max(0.0, min(float(segment.start), duration))
            end = max(start, min(float(segment.end), duration))
            self.on_segment(TranscriptSegment(start=offset + start, end=offset + end,
                                              text=segment.text.strip()))
        self.on_partial("")

    def _run(self):
        np = self._np
        pre_roll = deque(maxlen=max(1, math.ceil(0.25 / self.capture_block_seconds)))
        parts = []
        utterance_start = 0
        utterance_samples = 0
        silent_samples = 0
        last_partial_samples = 0
        capture_stopped = False
        try:
            if self.capture_info is not None:
                self.on_capture_info(dict(self.capture_info))
                self.on_status(f"麦克风 / Microphone: {self.capture_info['name']} · {self.capture_sample_rate} Hz → ASR {self.sample_rate} Hz")
            self.on_status("正在听写：停顿后确认文字，处理中间结果约每两秒更新。")
            while True:
                if self._stop_requested.is_set() and not capture_stopped:
                    # Wait for an in-flight callback before deciding the queue
                    # is drained. This runs on the worker, so stop() stays fast.
                    if self._stream is not None:
                        try:
                            self._stream.stop()
                        except Exception:
                            # A stream can already be stopped by PortAudio's
                            # callback thread; still drain queued audio.
                            log.debug("Microphone stream stop failed during drain", exc_info=True)
                    capture_stopped = True
                if capture_stopped and self._queue.empty():
                    break
                try:
                    start, samples = self._queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                is_speech = float(np.sqrt(np.mean(samples * samples))) >= self.energy_threshold
                if not parts:
                    if not is_speech:
                        pre_roll.append((start, samples))
                        continue
                    if pre_roll:
                        utterance_start = pre_roll[0][0]
                        parts = [block for _, block in pre_roll]
                        utterance_samples = sum(len(block) for block in parts)
                        pre_roll.clear()
                    else:
                        utterance_start = start
                parts.append(samples)
                utterance_samples += len(samples)
                silent_samples = 0 if is_speech else silent_samples + len(samples)
                final = (silent_samples >= self.silence_seconds * self.capture_sample_rate or
                         utterance_samples >= self.max_utterance_seconds * self.capture_sample_rate)
                if final:
                    self._decode(parts, utterance_start, final=True)
                    parts = []
                    utterance_samples = silent_samples = last_partial_samples = 0
                elif (not self._stop_requested.is_set() and
                      utterance_samples - last_partial_samples >= self.block_seconds * self.capture_sample_rate):
                    self._decode(parts, utterance_start, final=False)
                    last_partial_samples = utterance_samples
            if parts:
                self._decode(parts, utterance_start, final=True)
            if self._pending_error is not None:
                self.on_error(self._pending_error)
        except Exception as exc:
            self._stop_requested.set()
            self.on_error(exc)
        finally:
            self._stop_requested.set()
            try:
                self._close_capture()
            finally:
                self.on_done()
