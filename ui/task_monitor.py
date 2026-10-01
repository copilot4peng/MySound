"""Task progress and activity views.

The monitor deliberately keeps the event model independent from the ASR
workers.  Workers only need to call :meth:`TaskMonitor.handle_event`; the
same events can be replayed in tests or by another front end.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import math
from pathlib import Path
import tkinter as tk
from typing import Any, Iterable

try:  # Keep importing the state helpers possible on headless build hosts.
    import customtkinter as ctk
    _CTkFrame = ctk.CTkFrame
    _CTkLabel = ctk.CTkLabel
    _CTkButton = ctk.CTkButton
    _CTkTextbox = ctk.CTkTextbox
    _CTkScrollableFrame = ctk.CTkScrollableFrame
except Exception:  # pragma: no cover - used only when optional UI deps absent
    ctk = None
    _CTkFrame = tk.Frame
    _CTkLabel = tk.Label
    _CTkButton = tk.Button
    _CTkTextbox = tk.Text
    _CTkScrollableFrame = tk.Frame

from ui.i18n import register_messages
from ui.layout import ellipsize
from ui.tooltip import attach_tooltip


TASK_MONITOR_EN = {
    "任务监测": "Task monitor", "活动日志": "Activity log", "文件": "File",
    "麦克风": "Microphone", "模型": "Models", "正在读取媒体信息…": "Reading media information…",
    "正在检测并转写": "Detecting and transcribing", "正在转写": "Transcribing",
    "等待中": "Waiting", "处理完成": "Completed", "已取消": "Cancelled", "失败": "Failed",
    "音频时长未知": "Audio duration unknown", "路径": "Path", "大小": "Size",
    "格式": "Format", "编码": "Codec", "时长": "Duration", "采样率": "Sample rate",
    "声道": "Channels", "标准化时长": "Normalized duration", "设备": "Device",
    "采集采样率": "Capture rate", "识别采样率": "ASR rate", "复制": "Copy", "清空": "Clear",
    "暂无活动": "No activity yet", "静音跳过": "Silence skipped", "暂停": "Pause",
    "时长上限": "Duration limit", "结束": "End", "错误": "Error", "取消": "Cancelled",
    "待处理": "Pending", "处理中": "Processing", "完成": "Done", "发现 {count} 段": "{count} segments found",
    "发现 {count} 段 · 总数检测中": "{count} segments found · total detecting",
    "当前第 {index} 段 · {start}–{end}": "Current segment {index} · {start}–{end}",
    "当前第 {index} 段 · {start}–{end} · 已发现 {count} 段": "Current segment {index} · {start}–{end} · {count} found",
    "已完成 {count} 段": "{count} segments complete", "正在等待任务": "Waiting for a task",
    "任务进行中": "Task in progress", "任务已停止": "Task stopped", "任务已完成": "Task complete",
    "任务已失败": "Task failed", "任务已取消": "Task cancelled", "总数检测中": "Total detecting",
    "VAD 切片：目标 {target:.1f} 秒 · 最长 {maximum:.1f} 秒": "VAD segments: target {target:.1f}s · maximum {maximum:.1f}s",
    "图例": "Legend", "待处理 {count} 段": "Pending {count}", "处理中 {count} 段": "Processing {count}",
    "已完成 {completed} 段 · 静音跳过 {skipped} 段 · 最终 {total} 段":
        "Done {completed} · silence skipped {skipped} · final {total}",
    "已完成 {completed} 段 · 静音跳过 {skipped} 段 · 待处理 {pending} 段 · 已发现 {discovered} 段":
        "Done {completed} · silence skipped {skipped} · pending {pending} · discovered {discovered}",
    "已完成 {completed} 段 · 错误 {errors} 段 · 静音跳过 {skipped} 段 · 待处理 {pending} 段 · 已发现 {discovered} 段":
        "Done {completed} · errors {errors} · silence skipped {skipped} · pending {pending} · discovered {discovered}",
    "边界原因：{reason}": "Boundary: {reason}",
    "目标": "Target", "最长": "Maximum", "信息": "Info",
    "开始任务": "Task started", "已读取文件信息": "Media information loaded", "麦克风已连接": "Microphone connected",
    "媒体信息检测失败：{detail}": "Media inspection failed: {detail}",
    "切片": "Segment", "任务结束": "Task finished", "处理中": "Processing", "完成": "Done",
    "错误": "Error", "取消": "Cancelled", "静音跳过": "Silence skipped",
}
register_messages(TASK_MONITOR_EN)


def _number(value: Any, default: float | None = None) -> float | None:
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def format_duration(seconds: Any) -> str:
    value = _number(seconds)
    if value is None or value < 0:
        return "—"
    value = int(round(value))
    hours, rem = divmod(value, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


def format_bytes(size: Any) -> str:
    value = _number(size)
    if value is None or value < 0:
        return "—"
    units = ("B", "KB", "MB", "GB", "TB")
    i = 0
    while value >= 1024 and i < len(units) - 1:
        value /= 1024
        i += 1
    return f"{int(value)} {units[i]}" if i == 0 else f"{value:.1f} {units[i]}"


def format_timestamp(value: Any = None) -> str:
    if value is None:
        dt = datetime.now().astimezone()
    elif isinstance(value, datetime):
        dt = value.astimezone() if value.tzinfo else value
    elif isinstance(value, (int, float)):
        dt = datetime.fromtimestamp(value).astimezone()
    else:
        text = str(value)
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
            dt = parsed.astimezone()
        except ValueError:
            return text
    return dt.strftime("%H:%M:%S")


@dataclass
class TaskState:
    """Small, serialisable state reducer for task monitor events."""

    mode: str | None = None
    source: str = ""
    model: str = ""
    target_seconds: float | None = None
    max_seconds: float | None = None
    duration: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    chunks: list[dict[str, Any]] = field(default_factory=list)
    status: str = "idle"
    started_at: Any = None
    ended_at: Any = None
    error: str = ""
    reported_count: int | None = None
    reported_discovered_count: int | None = None

    def reset(self) -> None:
        self.mode = self.source = self.model = ""
        self.target_seconds = self.max_seconds = self.duration = None
        self.metadata.clear()
        self.chunks.clear()
        self.status, self.started_at, self.ended_at, self.error = "idle", None, None, ""
        self.reported_count = self.reported_discovered_count = None

    def handle_event(self, event: dict[str, Any] | None) -> "TaskState":
        if not isinstance(event, dict):
            return self
        # Worker events use ``kind``; ``type``/``event`` are accepted for
        # replay tools and older integrations.
        kind = event.get("kind", event.get("type", event.get("event")))
        if kind == "task_started":
            self.reset()
            self.mode = str(event.get("mode") or "file")
            self.source = str(event.get("source") or "")
            self.model = str(event.get("model") or "")
            self.target_seconds = _number(event.get("target_seconds"))
            self.max_seconds = _number(event.get("max_seconds"))
            self.started_at = event.get("timestamp")
            self.status = "running"
        elif kind in ("media_info", "mic_info"):
            self.metadata.update(dict(event.get("metadata") or {}))
            normalized = self.metadata.get("normalized_duration")
            duration = normalized if _number(normalized) is not None else self.metadata.get("duration")
            self.duration = _number(duration, self.duration)
        elif kind == "chunk":
            incoming = dict(event.get("chunk") or {})
            if not incoming:
                incoming = {key: value for key, value in event.items()
                            if key not in {"kind", "type", "event"}}
            if not incoming:
                return self
            ident = incoming.get("id")
            old = next((item for item in self.chunks if item.get("id") == ident), None)
            if old is None:
                self.chunks.append(incoming)
            else:
                old.update(incoming)
            if self.status == "idle":
                self.status = "running"
        elif kind in ("done", "failed", "cancelled"):
            payload = event.get("result") if isinstance(event.get("result"), dict) else event
            result_status = str(payload.get("status") or "").lower()
            count = payload.get("total_chunks")
            if count is None:
                count = payload.get("discovered_chunks")
            discovered = payload.get("discovered_chunks")
            try:
                self.reported_discovered_count = int(discovered) if discovered is not None else None
            except (TypeError, ValueError):
                self.reported_discovered_count = None
            try:
                self.reported_count = int(count) if count is not None else None
            except (TypeError, ValueError):
                self.reported_count = None
            if kind == "failed" or payload.get("failed") or result_status == "failed":
                self.status = "failed"
                self.error = str(payload.get("error") or payload.get("message") or "")
            elif kind == "cancelled" or payload.get("cancelled") or result_status == "cancelled":
                self.status = "cancelled"
            else:
                self.status = "done"
            self.ended_at = event.get("timestamp")
        return self

    @property
    def discovered_count(self) -> int:
        return max(len(self.chunks), self.reported_discovered_count or 0)

    def _count(self, *statuses: str) -> int:
        wanted = set(statuses)
        return sum(1 for chunk in self.chunks if str(chunk.get("status") or "pending") in wanted)

    @property
    def completed_count(self) -> int:
        return self._count("done")

    @property
    def skipped_count(self) -> int:
        return self._count("skipped")

    @property
    def pending_count(self) -> int:
        return self._count("pending")

    @property
    def processing_count(self) -> int:
        return self._count("processing")

    @property
    def error_count(self) -> int:
        return self._count("error")

    @property
    def cancelled_count(self) -> int:
        return self._count("cancelled")

    @property
    def final_count(self) -> int | None:
        # A cancelled/failed task has no trustworthy final total. Its
        # discovered count is shown separately from completed_count.
        if self.status != "done":
            return None
        return self.reported_count if self.reported_count is not None else self.discovered_count

    @property
    def current(self) -> tuple[int | str, dict[str, Any]] | None:
        for index, chunk in enumerate(self.chunks, 1):
            if chunk.get("status") == "processing":
                # Chunk IDs include skipped VAD intervals.  Surface the worker
                # ID so the monitor points at the actual segment being run.
                return (chunk.get("id") if chunk.get("id") is not None else index), chunk
        return None


def timeline_bins(chunks: Iterable[dict[str, Any]], duration: Any, width: int = 600) -> list[dict[str, Any]]:
    """Aggregate chunks into at most one item per canvas pixel.

    Raw chunks remain untouched in :class:`TaskState`; this helper only limits
    drawing work for long recordings.
    """
    items = [dict(c) for c in chunks if _number(c.get("end")) is not None]
    if not items or width <= 0:
        return []
    known_duration = _number(duration)
    extent = known_duration or max(float(c.get("end") or 0) for c in items)
    extent = max(extent or 0, 0.001)
    # Build add/remove events once and sweep the canvas pixels.  The previous
    # implementation scanned every chunk for every pixel (O(width*N)); this
    # keeps rendering O(width+N), even after several thousand VAD chunks.
    pixel_count = min(max(1, int(width)), 2000)
    events: list[list[tuple[str, int, dict[str, Any] | None]]] = [[] for _ in range(pixel_count + 1)]
    for index, chunk in enumerate(items):
        start = max(0.0, min(extent, _number(chunk.get("start"), 0.0) or 0.0))
        end = max(start, min(extent, _number(chunk.get("end"), start) or start))
        x0 = min(pixel_count - 1, max(0, int(start / extent * pixel_count)))
        x1 = min(pixel_count, max(x0 + 1, int(math.ceil(end / extent * pixel_count))))
        events[x0].append(("add", index, chunk))
        events[x1].append(("remove", index, None))

    priority = {"error": 6, "processing": 5, "pending": 4, "cancelled": 3,
                "skipped": 2, "done": 1}
    active: dict[int, dict[str, Any]] = {}
    result: list[dict[str, Any]] = []
    for pixel in range(pixel_count):
        # Remove intervals ending at this boundary before adding new ones.
        for action, index, chunk in events[pixel]:
            if action == "remove":
                active.pop(index, None)
            elif chunk is not None:
                active[index] = chunk
        if not active:
            continue
        chosen = max(active.values(), key=lambda c: priority.get(str(c.get("status")), 0))
        item = {
            "start": extent * pixel / pixel_count,
            "end": extent * (pixel + 1) / pixel_count,
            "status": chosen.get("status", "pending"),
            "chunk": chosen,
            "count": len(active),
        }
        if result:
            previous = result[-1]
            previous_id = previous["chunk"].get("id")
            chosen_id = chosen.get("id")
            if (previous["status"] == item["status"] and previous_id == chosen_id
                    and abs(previous["end"] - item["start"]) < 1e-9):
                previous["end"] = item["end"]
                previous["count"] = max(previous["count"], item["count"])
                continue
        result.append(item)
    return result


def _ellipsis(value: Any, limit: int = 56) -> str:
    text = str(value or "")
    return text if len(text) <= limit else text[: max(1, limit - 1)] + "…"


class _ViewBase(_CTkFrame):
    def _style(self, widget, role="surface", **options):
        styler = getattr(self.app, "style", None)
        if callable(styler):
            try:
                styler(widget, role, **options)
                return
            except Exception:
                pass
        if options:
            try:
                widget.configure(**options)
            except Exception:
                pass

    def _tr(self, source, **kwargs):
        translator = getattr(self.app, "tr", None)
        return translator(source, **kwargs) if callable(translator) else source.format(**kwargs)

    def _font(self, role="body"):
        font = getattr(self.app, "font", None)
        return font(role) if callable(font) else None


class TaskMonitor(_ViewBase):
    """Compact task details and a time-proportional timeline."""

    def __init__(self, parent, app, **kwargs):
        self.app = app
        self._fixed_height = max(128, int(kwargs.get("height", 164)))
        self._compact = False
        super().__init__(parent, **kwargs)
        try:
            self.configure(height=self._fixed_height)
        except Exception:
            pass
        self.state = TaskState()
        self._labels: list[tuple[Any, str]] = []
        self._build()

    def _build(self):
        # This panel is explicitly bounded by the capture pane. Long metadata
        # scrolls in row 1 instead of changing the pane's requested height.
        self.grid_propagate(False)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, minsize=26, weight=0)
        self.grid_rowconfigure(1, minsize=24, weight=1)
        self.grid_rowconfigure(2, minsize=24, weight=0)
        self.grid_rowconfigure(3, minsize=18, weight=0)
        self.grid_rowconfigure(4, minsize=20, weight=0)

        header = _CTkFrame(self, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=10, pady=(3, 0))
        header.grid_columnconfigure(1, weight=1)
        self.title = _CTkLabel(header, text=self._tr("任务监测"), anchor="w", font=self._font("section"))
        self.title.grid(row=0, column=0, sticky="w", padx=(0, 12))
        self.summary = _CTkLabel(header, text=self._tr("正在等待任务"), anchor="e", justify="right",
                                 wraplength=800, font=self._font("small"))
        self.summary.grid(row=0, column=1, sticky="e")

        if ctk is None:
            self.info_frame = _CTkScrollableFrame(self)
        else:
            self.info_frame = _CTkScrollableFrame(self, fg_color="transparent", corner_radius=0)
        self.info_frame.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 1))
        try:
            self.info_frame.grid_columnconfigure(0, weight=1)
        except Exception:
            pass
        self.detail = _CTkLabel(self.info_frame, text="", anchor="w", justify="left", wraplength=800,
                                font=self._font("small"))
        self.detail.grid(row=0, column=0, sticky="ew", padx=2, pady=(0, 1))
        self.path_full = ""
        try:
            attach_tooltip(self.detail, self.app, lambda: self.path_full)
        except Exception:
            pass
        self.strategy = _CTkLabel(self.info_frame, text="", anchor="w", justify="left", wraplength=800,
                                  font=self._font("small"))
        self.strategy.grid(row=1, column=0, sticky="ew", padx=2, pady=(0, 1))
        self.stats = _CTkLabel(self.info_frame, text="", anchor="w", justify="left", wraplength=800,
                               font=self._font("small"))
        self.stats.grid(row=2, column=0, sticky="ew", padx=2, pady=(0, 1))

        self.legend_canvas = tk.Canvas(self, height=20, bd=0, highlightthickness=0, relief="flat")
        self.legend_canvas.grid(row=4, column=0, sticky="ew", padx=10, pady=(0, 1))
        self.legend_canvas.bind("<Configure>", lambda _event: self._render_legend())
        try:
            attach_tooltip(self.legend_canvas, self.app, lambda: " · ".join(self._tr(item) for item in
                                                                     ("待处理", "处理中", "完成", "错误", "取消", "静音跳过")))
        except Exception:
            pass
        self.canvas = tk.Canvas(self, height=24, bd=0, highlightthickness=1, relief="flat")
        self.canvas.grid(row=2, column=0, sticky="ew", padx=10, pady=(1, 1))
        self.canvas.bind("<Configure>", lambda _event: self._render_timeline())
        self.canvas.bind("<Motion>", self._timeline_hover)
        self.canvas.bind("<Leave>", lambda _event: self._set_hover(None))
        self.hover = _CTkLabel(self, text="", anchor="w", justify="left", wraplength=800,
                               font=self._font("small"))
        self.hover.grid(row=3, column=0, sticky="ew", padx=10, pady=(0, 1))
        self._hover_full = ""
        try:
            attach_tooltip(self.hover, self.app, lambda: self._hover_full)
        except Exception:
            pass
        self.bind("<Configure>", self._on_resize, add="+")
        self.refresh_theme()
        self._refresh_text()

    def set_compact(self, compact: bool) -> None:
        """Reduce optional decorations when a high-DPI window is short."""
        compact = bool(compact)
        if compact == self._compact:
            return
        self._compact = compact
        if compact:
            self.grid_rowconfigure(0, minsize=20)
            self.grid_rowconfigure(1, minsize=8)
            self.grid_rowconfigure(2, minsize=20)
            self.grid_rowconfigure(3, minsize=0)
            self.grid_rowconfigure(4, minsize=0)
            self.hover.grid_remove()
            self.legend_canvas.grid_remove()
        else:
            self.grid_rowconfigure(0, minsize=30)
            self.grid_rowconfigure(1, minsize=32)
            self.grid_rowconfigure(2, minsize=24)
            self.grid_rowconfigure(3, minsize=20)
            self.grid_rowconfigure(4, minsize=20)
            self.hover.grid()
            self.legend_canvas.grid()
        self._on_resize(type("Resize", (), {"width": self.winfo_width()})())
        self._render_timeline()

    def handle_event(self, event: dict[str, Any] | None):
        self.state.handle_event(event)
        self._refresh_text()
        self._render_timeline()
        return self.state

    def _refresh_text(self):
        s = self.state
        status_source = {"idle": "正在等待任务", "running": "正在检测并转写" if s.duration is None else "正在转写", "done": "任务已完成",
                         "failed": "任务已失败", "cancelled": "任务已取消"}.get(s.status, "任务进行中")
        self.summary.configure(text=self._tr(status_source))
        if s.mode:
            mode = self._tr("麦克风" if s.mode == "mic" else "文件")
            source_name = Path(s.source).name if s.mode == "file" else self._tr(s.source)
            text = f"{mode} · {source_name}" if s.source else mode
            if s.model:
                text += f" · {self._tr('模型')}: {s.model}"
            metadata = s.metadata
            self.path_full = str(metadata.get("path") or s.source or "")
            details = []
            if s.mode == "mic":
                device_name = metadata.get("name") or metadata.get("device")
                if device_name:
                    details.append(f"{self._tr('设备')}: {device_name}")
                if metadata.get("capture_sample_rate"):
                    details.append(f"{self._tr('采集采样率')}: {metadata['capture_sample_rate']} Hz")
                if metadata.get("sample_rate"):
                    details.append(f"{self._tr('识别采样率')}: {metadata['sample_rate']} Hz")
            else:
                normalized_duration = metadata.get("normalized_duration")
                duration_value = (normalized_duration if _number(normalized_duration) is not None
                                  else metadata.get("duration"))
                media_values = (("size_bytes", "大小", metadata.get("size_bytes")),
                                ("format", "格式", metadata.get("format")),
                                ("codec", "编码", metadata.get("codec")),
                                ("duration", "时长", duration_value),
                                ("sample_rate", "采样率", metadata.get("sample_rate")),
                                ("channels", "声道", metadata.get("channels")))
                for key, label, value in media_values:
                    if value is not None and value != "":
                        if key == "size_bytes":
                            value = format_bytes(value)
                        elif key in {"duration", "normalized_duration"}:
                            value = format_duration(value)
                        elif key == "sample_rate":
                            value = f"{value} Hz"
                        details.append(f"{self._tr(label)}: {value}")
            if details:
                text += "\n" + " · ".join(details)
            if self.path_full and s.mode != "mic":
                text += f"\n{self._tr('路径')}: {self.path_full}"
            self.detail.configure(text=text)
        else:
            self.path_full = ""
            self.detail.configure(text=self._tr("将音频或视频拖到这里"))
        if s.mode == "file" and (s.target_seconds is not None or s.max_seconds is not None):
            target = s.target_seconds if s.target_seconds is not None else 0.0
            maximum = s.max_seconds if s.max_seconds is not None else target
            self.strategy.configure(text=self._tr(
                "VAD 切片：目标 {target:.1f} 秒 · 最长 {maximum:.1f} 秒",
                target=target, maximum=maximum,
            ))
        else:
            self.strategy.configure(text="")
        stats = self._stats_text()
        self.stats.configure(text=stats)
        current = s.current
        if current:
            index, chunk = current
            if self._compact:
                self.summary.configure(text=f"{self._tr('切片')} {index} · {self._tr('处理中')}")
            source = "当前第 {index} 段 · {start}–{end}"
            self._show_hover(self._tr(source, index=index,
                                      start=format_duration(chunk.get("start")),
                                      end=format_duration(chunk.get("end"))) +
                             (f" · {self._tr('发现 {count} 段', count=s.discovered_count)}"
                              if s.final_count is None else ""))
        elif s.final_count is not None:
            self._show_hover(self._tr(
                "已完成 {completed} 段 · 静音跳过 {skipped} 段 · 最终 {total} 段",
                completed=s.completed_count, skipped=s.skipped_count, total=s.final_count,
            ))
        elif s.discovered_count:
            source = "发现 {count} 段 · 总数检测中" if s.status == "running" else "发现 {count} 段"
            self._show_hover(self._tr(source, count=s.discovered_count))
        else:
            self._show_hover("")

    def _stats_text(self) -> str:
        s = self.state
        discovered = s.discovered_count
        if not discovered and s.final_count is None:
            return ""
        if s.status == "done" and s.final_count is not None:
            return self._tr(
                "已完成 {completed} 段 · 静音跳过 {skipped} 段 · 最终 {total} 段",
                completed=s.completed_count, skipped=s.skipped_count, total=s.final_count,
            )
        if s.status == "failed":
            return self._tr(
                "已完成 {completed} 段 · 错误 {errors} 段 · 静音跳过 {skipped} 段 · 待处理 {pending} 段 · 已发现 {discovered} 段",
                completed=s.completed_count, errors=s.error_count, skipped=s.skipped_count,
                pending=s.pending_count + s.processing_count, discovered=discovered,
            )
        return self._tr(
            "已完成 {completed} 段 · 静音跳过 {skipped} 段 · 待处理 {pending} 段 · 已发现 {discovered} 段",
            completed=s.completed_count, skipped=s.skipped_count,
            pending=s.pending_count + s.processing_count, discovered=discovered,
        )

    def _on_resize(self, event):
        width = max(100, int(getattr(event, "width", 800) / self._get_widget_scaling()) - 42)
        for label in (getattr(self, "detail", None),
                      getattr(self, "strategy", None), getattr(self, "stats", None)):
            if label is not None:
                try:
                    label.configure(wraplength=width)
                except Exception:
                    pass
        self._show_hover(self._hover_full)

    def _render_legend(self):
        if not hasattr(self, "legend_canvas"):
            return
        palette = getattr(getattr(self.app, "theme", None), "palette", {}) or {}
        self.legend_canvas.configure(bg=palette.get("surface", palette.get("background", "#20252B")))
        self.legend_canvas.delete("all")
        colors = {"待处理": palette.get("muted", "#78818C"), "处理中": palette.get("accent", "#3A86FF"),
                  "完成": palette.get("success", "#45B97C"), "错误": palette.get("danger", "#D64545"),
                  "取消": palette.get("danger", "#D64545"), "静音跳过": palette.get("secondary", "#555E67")}
        x = 2
        width = max(1, int(self.legend_canvas.winfo_width() or 600))
        font = self._font("small")
        for source, color in colors.items():
            label = self._tr(source)
            try:
                measured = int(font.measure(label)) + 2 if font is not None else len(label) * 8
            except Exception:
                measured = len(label) * 8
            if x + 15 + measured > width - 2:
                break
            self.legend_canvas.create_rectangle(x, 6, x + 10, 16, fill=color, outline=color)
            self.legend_canvas.create_text(x + 15, 11, text=label, anchor="w",
                                           fill=palette.get("muted", "#AAB4C0"), font=font)
            x += 22 + measured

    def _render_timeline(self):
        if not hasattr(self, "canvas"):
            return
        try:
            width = max(1, int(self.canvas.winfo_width()))
        except Exception:
            width = 600
        palette = getattr(getattr(self.app, "theme", None), "palette", {}) or {}
        bg, border = palette.get("input", "#20252B"), palette.get("border", "#52606D")
        self.canvas.configure(bg=bg, highlightbackground=border, highlightcolor=border)
        self.canvas.delete("all")
        colors = {"pending": palette.get("muted", "#78818C"), "processing": palette.get("accent", "#3A86FF"),
                  "done": palette.get("success", "#45B97C"), "error": palette.get("danger", "#D64545"),
                  "cancelled": palette.get("danger", "#D64545"), "skipped": palette.get("secondary", "#555E67")}
        bins = timeline_bins(self.state.chunks, self.state.duration, width)
        extent = _number(self.state.duration) or max((float(c.get("end") or 0) for c in self.state.chunks), default=0.001)
        extent = max(extent, 0.001)
        previous = None
        for item in bins:
            x0 = item["start"] / extent * width
            x1 = item["end"] / extent * width
            self.canvas.create_rectangle(x0, 3, max(x0 + 1, x1), 29,
                                         fill=colors.get(item["status"], colors["pending"]),
                                         outline="", width=0)
            if previous is not None and abs(previous["end"] - item["start"]) < 1e-9:
                self.canvas.create_line(x0, 3, x0, 29, fill=palette.get("border", "#52606D"), width=1)
            previous = item

    def _timeline_hover(self, event):
        duration = _number(self.state.duration)
        if not duration:
            return self._set_hover(None)
        time = max(0.0, min(duration, event.x / max(1, self.canvas.winfo_width()) * duration))
        for index, chunk in enumerate(self.state.chunks, 1):
            start, end = _number(chunk.get("start"), 0), _number(chunk.get("end"), 0)
            if start <= time <= end:
                status = str(chunk.get("status") or "pending")
                label = {"pending": "待处理", "processing": "处理中", "done": "完成", "error": "错误",
                         "cancelled": "取消", "skipped": "静音跳过"}.get(status, status)
                self._set_hover(self._tr("当前第 {index} 段 · {start}–{end}",
                                         index=(chunk.get("id") if chunk.get("id") is not None else index),
                                         start=format_duration(start), end=format_duration(end)) +
                                f" · {self._tr(label)}" + self._reason_suffix(chunk))
                return
        self._set_hover(None)

    def _reason_suffix(self, chunk: dict[str, Any]) -> str:
        reason = str(chunk.get("reason") or "").lower()
        if not reason:
            return ""
        labels = {"pause": "暂停", "limit": "时长上限", "end": "结束", "silence": "静音跳过"}
        return f" · {self._tr('边界原因：{reason}', reason=self._tr(labels.get(reason, reason)))}"

    def _set_hover(self, text):
        if hasattr(self, "hover"):
            if text:
                self._show_hover(text)
            else:
                # Keep the persistent current/final count visible when the
                # pointer leaves a timeline gap; hover text is supplemental.
                self._refresh_text()

    def _show_hover(self, text: str):
        self._hover_full = str(text or "")
        width = max(30, self.winfo_width() / self._get_widget_scaling() - 24)
        self.hover.configure(text=ellipsize(self._hover_full, width, self._font("small").measure), wraplength=0)

    def refresh_theme(self):
        self._style(self, "surface")
        self._style(self.info_frame, "scroll")
        for widget, role in ((self.title, "title"), (self.summary, "label"), (self.detail, "muted"),
                             (self.strategy, "muted"), (self.stats, "muted"), (self.hover, "muted")):
            self._style(widget, role)
        self._render_legend()
        self._render_timeline()

    def retranslate(self):
        self.title.configure(text=self._tr("任务监测"))
        self._refresh_text()


class ActivityLog(_ViewBase):
    """In-memory activity feed with a readonly, bounded text view."""

    MAX_ENTRIES = 1000

    def __init__(self, parent, app, **kwargs):
        self.app = app
        self.entries: list[dict[str, str]] = []
        self._compact = False
        super().__init__(parent, **kwargs)
        self._build()

    def _build(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        bar = _CTkFrame(self, fg_color="transparent")
        bar.grid(row=0, column=0, sticky="ew", padx=8, pady=(4, 0))
        bar.grid_columnconfigure(0, weight=1)
        self.title = _CTkLabel(bar, text=self._tr("活动日志"), anchor="w", font=self._font("section"))
        self.title.grid(row=0, column=0, sticky="w")
        self.copy_button = _CTkButton(bar, text=self._tr("复制"), width=70, command=self.copy)
        self.copy_button.grid(row=0, column=1, sticky="e", padx=(4, 0))
        self.clear_button = _CTkButton(bar, text=self._tr("清空"), width=70, command=self.clear)
        self.clear_button.grid(row=0, column=2, sticky="e")
        decorate = getattr(self.app, "decorate", None)
        if callable(decorate):
            try:
                decorate(self.clear_button, "trash", tone="text")
            except Exception:
                pass
        self.textbox = _CTkTextbox(self, wrap="word", state="disabled", font=self._font("small"))
        self.textbox.grid(row=1, column=0, sticky="nsew", padx=8, pady=(3, 8))
        self._configure_tags()
        self.refresh_theme()
        self._render()

    def _configure_tags(self):
        palette = getattr(getattr(self.app, "theme", None), "palette", {}) or {}
        for level, color in (("INFO", palette.get("text", "#E8EEF5")), ("WARNING", palette.get("warning", "#D99A2B")),
                             ("ERROR", palette.get("danger", "#D64545")), ("DEBUG", palette.get("muted", "#8995A3"))):
            try:
                self.textbox.tag_config(level, foreground=color)
            except Exception:
                pass

    def append(self, message, level="INFO", timestamp=None):
        level = str(level or "INFO").upper()
        self.entries.append({"timestamp": format_timestamp(timestamp), "level": level, "message": str(message)})
        del self.entries[:-self.MAX_ENTRIES]
        self._render()

    def _render(self):
        if not hasattr(self, "textbox"):
            return
        try:
            at_bottom = float(self.textbox.yview()[1]) >= 0.995
        except Exception:
            at_bottom = True
        try:
            self.textbox.configure(state="normal")
            self.textbox.delete("1.0", "end")
            for item in self.entries:
                self.textbox.insert("end", f"[{item['timestamp']}] {item['message']}\n", item["level"])
            self.textbox.configure(state="disabled")
            if at_bottom:
                self.textbox.see("end")
        except Exception:
            pass

    def clear(self):
        self.entries.clear()
        self._render()

    def copy(self):
        text = "".join(f"[{i['timestamp']}] {i['message']}\n" for i in self.entries)
        try:
            self.clipboard_clear()
            self.clipboard_append(text)
        except Exception:
            return text
        return text

    def refresh_theme(self):
        self._style(self, "surface")
        self._style(self.title, "title")
        self._style(self.copy_button, "secondary")
        self._style(self.clear_button, "secondary")
        self._style(self.textbox, "textbox")
        self._configure_tags()

    def retranslate(self):
        self.title.configure(text=self._tr("活动日志"))
        self.copy_button.configure(text=self._tr("复制"))
        self.clear_button.configure(text=self._tr("清空"))


__all__ = ["TaskState", "TaskMonitor", "ActivityLog", "timeline_bins", "format_duration", "format_bytes", "format_timestamp"]
