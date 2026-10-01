"""Serialize inference work and deliver plain events to the Tk main thread."""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import inspect
from pathlib import Path
import logging
import threading
from typing import Callable

from core.model_manager import ModelManager

log = logging.getLogger(__name__)


class TranscriptionController:
    def __init__(self, history, emit: Callable[[dict], None]):
        self.history = history
        self.emit = emit
        self.manager = ModelManager(on_status=self._status)
        self._lock = threading.RLock()
        self._cancel = threading.Event()
        self._active = False
        self._mic = None
        self._segments: list[dict] = []
        self._source = ""
        self._model_name = ""
        self._error = False
        self._started_at = None
        self._duration = None
        self._processed_until = 0.0
        self._discovered = 0
        self._completed = 0
        self._skipped = 0
        self._total_known = None
        self._seen_chunk_ids = set()

    @staticmethod
    def _timestamp():
        return datetime.now(timezone.utc).isoformat(timespec="milliseconds")

    def _event(self, event):
        event.setdefault("timestamp", self._timestamp())
        self.emit(event)

    @property
    def busy(self):
        with self._lock:
            return self._active

    def _begin(self, source, config):
        with self._lock:
            if self._active:
                raise RuntimeError("请先停止或等待当前任务完成。")
            if not config.get("model_path"):
                raise ValueError("请先在模型设置中选择本地模型路径。")
            self._active = True
            self._cancel.clear()
            self._mic = None
            self._error = False
            self._segments = []
            self._source = source
            self._model_name = f"{config['model_type']} / {Path(config['model_path']).name}"
            self._started_at = self._timestamp()
            self._duration = None
            self._processed_until = 0.0
            self._discovered = self._completed = self._skipped = 0
            self._total_known = None
            self._seen_chunk_ids = set()
        self._event({"kind": "task_started", "mode": "mic" if source == "麦克风" else "file",
                     "source": source, "model": self._model_name,
                     "target_seconds": config.get("vad_target_seconds"),
                     "max_seconds": config.get("vad_max_seconds")})

    def _status(self, text, *, _level="INFO", **values):
        log.log(getattr(logging, _level, logging.INFO), "%s %s", text, values if values else "")
        self._event({"kind": "status", "text": str(text), "values": values, "level": _level})

    def _forward_processor_event(self, event):
        kind = event.get("kind")
        if kind == "media_info":
            metadata = dict(event.get("metadata") or {})
            self._duration = metadata.get("normalized_duration") or metadata.get("duration")
            self._event({"kind": "media_info", "metadata": metadata})
            return
        if kind != "chunk":
            self._event(event)
            return
        chunk = dict(event.get("chunk") or {})
        chunk_id = chunk.get("id")
        if chunk_id not in self._seen_chunk_ids:
            self._seen_chunk_ids.add(chunk_id)
            self._discovered += 1
        if chunk.get("status") == "skipped":
            self._skipped += 1
            self._processed_until = max(self._processed_until, float(chunk.get("end", 0)))
        self._event({"kind": "chunk", "chunk": chunk})

    def _chunk_status(self, chunk, status, **extra):
        data = dict(chunk)
        data["status"] = status
        data.update(extra)
        self._event({"kind": "chunk", "chunk": data})

    def _load(self, config):
        self._status("正在加载本地模型，首次加载可能需要一些时间…")
        self.manager.load(config["model_type"], config["model_path"], config.get("device", "auto"))
        return self.manager

    def _append(self, segment):
        data = asdict(segment) if not isinstance(segment, dict) else dict(segment)
        if data["text"].strip():
            self._segments.append(data)
            self._event({"kind": "segment", "segment": data})

    def _failed(self, error):
        self._error = True
        log.error("Transcription failed: %s", error, exc_info=(type(error), error, error.__traceback__))
        self._event({"kind": "error", "text": str(error), "text_en": getattr(error, "message_en", None),
                     "details": getattr(error, "details", "")})

    def _finish(self):
        # Mic callbacks and startup failure may both attempt finalization.
        with self._lock:
            if not self._active:
                return
            record = None
            try:
                if self._segments:
                    record = self.history.add(
                        source=self._source,
                        model=self._model_name,
                        text="\n".join(s["text"].strip() for s in self._segments),
                        segments=self._segments,
                    )
            except Exception as exc:
                self._failed(exc)
            self._mic = None
            # Queue completion before allowing the next job to start.
            cancelled = self._cancel.is_set()
            if not cancelled and not self._error:
                self._total_known = self._discovered
            self._event({"kind": "done", "record": record, "cancelled": cancelled, "failed": self._error,
                         "duration": self._duration, "processed_until": self._processed_until,
                         "discovered_chunks": self._discovered, "completed_chunks": self._completed,
                         "skipped_chunks": self._skipped, "total_chunks": self._total_known,
                         "progress": (1.0 if self._total_known is not None and self._duration else
                                      (None if not self._duration else min(1.0, self._processed_until / self._duration)))})
            self._active = False

    def start_file(self, path, config):
        config = dict(config)
        self._begin(str(path), config)
        threading.Thread(target=self._file_worker, args=(path, config), daemon=True, name="file-transcription").start()

    def _file_worker(self, path, config):
        try:
            from core.audio_processor import AudioProcessor
            from core.audio_processor import probe_media
            from core.model_manager import TranscriptSegment

            metadata = probe_media(Path(path).expanduser().resolve())
            self._duration = metadata.get("duration")
            self._event({"kind": "media_info", "metadata": dict(metadata)})
            if metadata.get("probe_error"):
                detail = str(metadata["probe_error"])
                self._status("媒体信息检测失败：{detail}", detail=detail, _level="WARNING")
            if self._cancel.is_set():
                raise InterruptedError("转写已取消")
            model = self._load(config)
            if self._cancel.is_set():
                return
            processor = AudioProcessor(
                target_seconds=float(config.get("vad_target_seconds", 30)),
                max_seconds=float(config.get("vad_max_seconds", 60)),
            )
            kwargs = {"cancel_event": self._cancel, "on_status": self._status}
            parameters = inspect.signature(processor.process).parameters
            if "on_event" in parameters:
                kwargs["on_event"] = self._forward_processor_event
            if "metadata" in parameters:
                kwargs["metadata"] = metadata
            if "metadata_already_emitted" in parameters:
                kwargs["metadata_already_emitted"] = True
            with processor.process(path, **kwargs) as chunks:
                for index, chunk in enumerate(chunks, 1):
                    if self._cancel.is_set():
                        break
                    if chunk.id is None:
                        chunk_id = index
                    else:
                        chunk_id = chunk.id
                    chunk_info = {"id": chunk_id, "start": chunk.start, "end": chunk.end,
                                  "status": "processing", "reason": getattr(chunk, "reason", "end")}
                    if chunk_id not in self._seen_chunk_ids:
                        self._seen_chunk_ids.add(chunk_id)
                        self._discovered += 1
                    self._chunk_status(chunk_info, "processing")
                    self._status("正在转写第 {index} 段 · {start:.1f}–{end:.1f} 秒", index=index, start=chunk.start, end=chunk.end)
                    try:
                        result = model.transcribe(chunk.audio, sample_rate=chunk.sample_rate, language=config.get("language"))
                    except Exception as exc:
                        self._chunk_status(chunk_info, "error", error=str(exc))
                        raise
                    for segment in result:
                        start = max(chunk.start, min(chunk.end, chunk.start + segment.start))
                        end = max(start, min(chunk.end, chunk.start + segment.end))
                        if end > start:
                            self._append(TranscriptSegment(start, end, segment.text))
                    self._completed += 1
                    self._processed_until = max(self._processed_until, chunk.end)
                    self._chunk_status(chunk_info, "done")
        except InterruptedError:
            self._cancel.set()
        except Exception as exc:
            self._failed(exc)
        finally:
            self._finish()

    def start_mic(self, config):
        config = dict(config)
        self._begin("麦克风", config)
        threading.Thread(target=self._mic_worker, args=(config,), daemon=True, name="microphone-startup").start()

    def _mic_worker(self, config):
        try:
            from core.mic_streamer import MicStreamer

            model = self._load(config)
            with self._lock:
                if self._cancel.is_set():
                    self._finish()
                    return
                self._mic = MicStreamer(
                    model=model,
                    on_partial=lambda text: self.emit({"kind": "partial", "text": text}),
                    on_segment=self._append,
                    on_status=self._status,
                    on_error=self._failed,
                    on_done=self._finish,
                    on_capture_info=lambda metadata: self._event({"kind": "mic_info", "metadata": dict(metadata)}),
                    language=config.get("language"),
                    device=config.get("mic_device"),
                    block_seconds=float(config.get("mic_block_seconds", 2.0)),
                )
                self._mic.start()
        except Exception as exc:
            self._failed(exc)
            self._finish()

    def stop(self):
        with self._lock:
            self._cancel.set()
            mic = self._mic
        if mic is not None:
            mic.stop()
        self._status("正在停止并保存已识别内容，请等待当前推理完成…")
