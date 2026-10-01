from __future__ import annotations

import logging
from pathlib import Path
import queue
import tkinter as tk
from tkinter import filedialog, messagebox
import weakref

import customtkinter as ctk

from core.controller import TranscriptionController
from core.exporter import export_record
from storage import BootstrapStore, ConfigStore, HistoryStore, resolve_data_directory
from ui.dpi import configure_tk_dpi
from ui.i18n import Translator, register_messages
from ui.icons import IconManager, ICON_LABELS
from ui.layout import clamp_layout, clamp_sash, ellipsize
from ui.theme import ThemeManager
from ui.typography import Typography
from ui.settings import SettingsPage
from ui.task_monitor import ActivityLog, TaskMonitor

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD

    class DesktopRoot(ctk.CTk, TkinterDnD.DnDWrapper):
        def __init__(self):
            super().__init__()
            try:
                self.TkdndVersion = TkinterDnD._require(self)
                self.dnd_available = True
            except (tk.TclError, RuntimeError):
                logging.getLogger(__name__).exception("Drag and drop initialization failed")
                self.dnd_available = False
except ImportError:
    class DesktopRoot(ctk.CTk):
        dnd_available = False


LANGUAGES = {"自动识别": None, "中文": "Chinese", "英语": "English", "日语": "Japanese", "韩语": "Korean"}

register_messages({
    "转写工作台": "Transcription studio", "历史记录": "History", "文件": "File", "听写": "Mic",
    "音频与视频 · 本地处理": "Audio and video · Processed locally",
    "选择文件，或将音频拖入下方": "Choose a file or drop audio below",
    "实时预览": "Live preview", "选择一段音频，开始记录": "Capture your next idea",
    "拖入音频或视频": "Drop audio or video",
    "转写完成后，记录会保存在这里。": "Your completed transcripts will appear here.",
})
ICON_LABELS.update({
    "转写工作台": "wave", "实时预览": "wave", "本机配置": "monitor",
    "处理器": "cpu", "逻辑线程": "cpu", "显卡": "gpu", "系统内存": "memory",
    "当前可用内存": "memory", "显存总量": "gpu", "当前可用显存": "gpu",
    "操作系统": "monitor", "所选模型信息": "model", "本机模型建议": "check",
    "参数规模": "model", "本地文件体积": "folder", "CPU 推理内存预算": "cpu",
    "GPU 推理显存预算": "gpu", "GPU 推理主机内存预算": "memory",
    "速度提示": "info", "本机适配度": "check", "主题预设": "palette",
    "此模型的本地路径": "folder", "扫描根目录": "scan", "明暗模式": "palette",
    "下次启动使用的数据目录": "folder", "选择模型文件夹": "folder",
    "保存": "save", "返回": "back",
    "听写刷新间隔（秒，1–5）": "mic", "文件目标切片时长（秒，5–60）": "file",
    "文件最长切片时长（秒，目标时长–60）": "file",
    "任务监测": "monitor", "活动日志": "file", "复制": "export", "清空": "trash",
})


class MySoundApp(DesktopRoot):
    def __init__(self, data_dir=None, data_selection=None, bootstrap=None):
        self.bootstrap = bootstrap or BootstrapStore()
        self.data_selection = data_selection or resolve_data_directory(data_dir, self.bootstrap)
        self.config_store = ConfigStore(self.data_selection.path)
        self.history_store = HistoryStore(self.data_selection.path)
        config = self.config_store.load()
        ctk.set_appearance_mode(config["appearance_mode"])
        ctk.set_default_color_theme("blue")
        super().__init__()
        configure_tk_dpi(self)
        self.translator = Translator(config["ui_language"])
        self.typography = Typography(config["font_size"])
        self._sized_controls = weakref.WeakKeyDictionary()
        self.title(self.tr("MySound · 本地语音转文字"))
        self.resizable(True, True)
        self.theme = ThemeManager(config["appearance_mode"], config["color_theme"])
        self.icons = IconManager(self.theme.palette, config["font_size"])
        self._layout = clamp_layout(config["window_layout"], self._reverse_window_scaling(self.winfo_screenwidth()), self._reverse_window_scaling(self.winfo_screenheight()))
        self._normal_size = (self._layout["width"], self._layout["height"])
        self.minsize(min(900, self._layout["width"]), min(620, self._layout["height"]))
        self.geometry(f"{self._layout['width']}x{self._layout['height']}")
        self.events = queue.Queue()
        self.controller = TranscriptionController(self.history_store, self.events.put)
        self.current_record = None
        self._baseline = ""
        self._closing = False
        self._settings = None
        self._page = "transcription"
        self._search_timer = None
        self._layout_timer = None
        self._resize_timer = None
        self._poll_timer = None
        self._layout_ready = False
        self._dragging = set()
        self._history_buttons = []
        self._history_items = {}
        self._history_empty = None
        self._history_timer = None
        self._history_query = None
        self._wrap_labels = []
        self._record_title = "转写文本"
        self._job_ui_active = False
        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._close)
        self.bind("<Control-s>", lambda _event: self._settings.save_current() if self._page == "settings" else self._save())
        self.bind("<Control-o>", lambda _event: self._choose_file())
        self.bind("<Control-e>", lambda _event: self._export())
        self._update_model_label()
        self._refresh_history()
        self._poll_timer = self.after(60, self._poll_events)
        self.bind("<Configure>", self._window_configured, add="+")
        self.after(150, self._restore_layout)

    def style(self, widget, role, **options):
        if isinstance(widget, (ctk.CTkLabel, ctk.CTkButton, ctk.CTkEntry, ctk.CTkTextbox, ctk.CTkOptionMenu)):
            if widget.cget("font") not in self.typography._fonts.values():
                widget.configure(font=self.font())
            if isinstance(widget, ctk.CTkOptionMenu):
                widget.configure(dropdown_font=self.font())
            if isinstance(widget, (ctk.CTkLabel, ctk.CTkButton)) and widget.cget("text") and role not in {"history", "selected_history"}:
                source = widget.cget("text")
                self.translate_widget(widget, widget.cget("text"))
                if source in ICON_LABELS:
                    self.decorate(widget, ICON_LABELS[source], tone="on_accent" if role in {"button", "danger"} else "text")
            if isinstance(widget, ctk.CTkEntry) and widget.cget("placeholder_text"):
                self.translate_widget(widget, widget.cget("placeholder_text"), option="placeholder_text")
            if isinstance(widget, (ctk.CTkEntry, ctk.CTkButton, ctk.CTkOptionMenu)):
                lines = self._sized_controls.get(widget, 1)
                self._sized_controls[widget] = lines
                widget.configure(height=max(28, widget.cget("font").metrics("linespace") * lines + 12))
        elif isinstance(widget, ctk.CTkTabview):
            widget._segmented_button.configure(font=self.font())
        return self.theme.style(widget, role, **options)

    @property
    def ui_language(self):
        return self.translator.language

    def tr(self, source, **kwargs):
        return self.translator.tr(source, **kwargs)

    def tr_error(self, error):
        return getattr(error, "message_en", None) if self.ui_language == "en_US" and getattr(error, "message_en", None) else self.tr(str(error))

    def translate_widget(self, widget, source, option="text", **kwargs):
        if option == "text" and widget is getattr(self, "status", None):
            self._status_content = (source, dict(kwargs))
            self.after_idle(self._resize_text)
        if option == "text" and source in ICON_LABELS and isinstance(widget, (ctk.CTkLabel, ctk.CTkButton)):
            self.decorate(widget, ICON_LABELS[source])
        return self.translator.bind(widget, source, option, **kwargs)

    def font(self, role="body"):
        return self.typography.font(role)

    def decorate(self, widget, name, tone="text", **options):
        return self.icons.bind(widget, name, tone=tone, **options)

    def _status(self, source, *, _timestamp=None, _level="INFO", **kwargs):
        self.translate_widget(self.status, source, **kwargs)
        if hasattr(self, "activity_log"):
            try:
                self.activity_log.append(self.tr(source, **kwargs), _level, _timestamp)
            except Exception:
                logging.getLogger(__name__).debug("Could not append activity status", exc_info=True)

    def _partial(self, text, *, transcript=False):
        if transcript:
            self.translator.unbind(self.partial_label)
            self.partial_label.configure(text=text)
        else:
            self.translate_widget(self.partial_label, text)

    def _build_ui(self):
        self.style(self, "frame")
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self.horizontal = self.style(tk.PanedWindow(self, orient="horizontal", sashwidth=8, sashpad=0, borderwidth=0, sashrelief="flat", opaqueresize=True, sashcursor="sb_h_double_arrow"), "paned")
        self.horizontal.grid(row=0, column=0, sticky="nsew")
        sidebar = self.style(ctk.CTkFrame(self.horizontal, width=255, corner_radius=0), "surface")
        self.sidebar = sidebar
        sidebar.grid_columnconfigure(0, weight=1)
        sidebar.grid_rowconfigure(4, weight=1)
        self.brand_label = self.style(ctk.CTkLabel(sidebar, text="MySound", font=self.font("brand"), anchor="w"), "title")
        self.brand_label.grid(row=0, column=0, padx=20, pady=(22, 0), sticky="ew")
        sidebar_hint = self.style(ctk.CTkLabel(sidebar, text="本地识别 · 离线保存", font=self.font("small"), anchor="w", justify="left", wraplength=210), "muted")
        self.sidebar_hint = sidebar_hint
        sidebar_hint.grid(row=1, column=0, padx=20, pady=(2, 18), sticky="ew")
        self.search = self.style(ctk.CTkEntry(sidebar, placeholder_text="搜索历史记录…"), "entry")
        self.style(ctk.CTkLabel(sidebar, text="历史记录", font=self.font("small"), anchor="w"), "muted").grid(row=2, column=0, padx=18, pady=(0, 8), sticky="ew")
        self.search.grid(row=3, column=0, padx=16, pady=(0, 12), sticky="ew")
        self.search.bind("<KeyRelease>", self._search_changed)
        self.history_list = self.style(ctk.CTkScrollableFrame(sidebar), "scroll")
        self.history_list.grid(row=4, column=0, padx=8, sticky="nsew")
        self.history_list.bind("<Configure>", self._schedule_history_layout, add="+")
        self.settings_button = self.style(ctk.CTkButton(sidebar, text="配置", command=self._open_settings), "secondary")
        self.settings_button.grid(row=5, column=0, padx=16, pady=16, sticky="ew")

        self.content = self.style(ctk.CTkFrame(self.horizontal, corner_radius=0), "frame")
        self.content.grid_rowconfigure(0, weight=1)
        self.content.grid_columnconfigure(0, weight=1)
        self.horizontal.add(sidebar, minsize=220, width=255, stretch="never")
        self.horizontal.add(self.content, minsize=640, stretch="always")
        main = self.style(ctk.CTkFrame(self.content, corner_radius=0), "frame")
        self.transcription_page = main
        main.grid(row=0, column=0, sticky="nsew", padx=20, pady=18)
        main.grid_columnconfigure(0, weight=1)
        main.grid_rowconfigure(2, weight=1)
        self.workspace_title = self.style(ctk.CTkLabel(main, text="转写工作台", font=self.font("title"), anchor="w"), "title")
        self.workspace_title.grid(row=0, column=0, sticky="ew")
        self.model_label = self.style(ctk.CTkLabel(main, text="", anchor="w", font=self.font("small")), "muted")
        self.model_label.grid(row=1, column=0, sticky="ew", pady=(4, 14))
        self.vertical = self.style(tk.PanedWindow(main, orient="vertical", sashwidth=8, sashpad=0, borderwidth=0, sashrelief="flat", opaqueresize=True, sashcursor="sb_v_double_arrow"), "paned")
        self.vertical.grid(row=2, column=0, sticky="nsew")
        self.capture_panel = self.style(ctk.CTkFrame(self.vertical, corner_radius=0), "frame")
        self.capture_panel.grid_columnconfigure(0, weight=1)
        self.capture_panel.grid_rowconfigure(2, weight=1)
        self._capture_mode = "file"
        capture_body = self.style(ctk.CTkScrollableFrame(self.capture_panel, corner_radius=12), "scroll")
        self.capture_body = capture_body
        capture_body.grid(row=0, column=0, sticky="nsew")
        capture_body.grid_columnconfigure(0, weight=1)
        drop = self.style(ctk.CTkFrame(capture_body, border_width=1, corner_radius=14), "surface")
        self.drop_area = drop
        drop.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        drop.grid_columnconfigure(0, weight=1)
        drop_text = "将音频或视频拖到这里" if self.dnd_available else "选择音频或视频文件"
        self._drop_source = drop_text
        self.drop_label = self.style(ctk.CTkLabel(drop, text=drop_text, height=26, font=self.font()), "label")
        self.decorate(self.drop_label, "upload", tone="accent_text")
        self.drop_label.grid(row=0, column=0, padx=18, pady=(3, 0), sticky="ew")
        self.format_label = self.style(ctk.CTkLabel(drop, text="MP3 / WAV / M4A / FLAC / MP4", font=self.font("small"), wraplength=480), "muted")
        self.format_label.grid(row=1, column=0, padx=18, pady=(0, 3), sticky="ew")
        controls = self.style(ctk.CTkFrame(self.capture_panel, corner_radius=0), "frame")
        controls.grid(row=1, column=0, sticky="ew", pady=(8, 10))
        self.capture_controls = controls
        for column in range(3):
            controls.grid_columnconfigure(column, weight=1, uniform="capture-actions")
        self.file_button = self.style(ctk.CTkButton(controls, text="选择文件", width=125, command=self._choose_file), "button")
        self.file_button.grid(row=0, column=0, padx=(0, 5), sticky="ew")
        self.mic_button = self.style(ctk.CTkButton(controls, text="开始听写", width=125, command=self._start_mic), "button")
        self.mic_button.grid(row=0, column=1, padx=5, sticky="ew")
        self.stop_button = self.style(ctk.CTkButton(controls, text="停止", width=85, state="disabled", command=self._stop), "danger")
        self.stop_button.grid(row=0, column=2, padx=(5, 0), sticky="ew")
        if self.dnd_available:
            for widget in (drop, self.drop_label, self.drop_label._label):
                widget.drop_target_register(DND_FILES)
                widget.dnd_bind("<<Drop>>", self._drop)
        self.preview_title = self.style(ctk.CTkLabel(capture_body, text="实时预览", font=self.font("small"), anchor="w"), "muted")
        self.decorate(self.preview_title, "wave", tone="muted")
        self.preview_title.grid(row=1, column=0, padx=6, pady=(4, 0), sticky="ew")
        self.partial_label = self.style(ctk.CTkLabel(capture_body, text="实时听写将在这里显示…", anchor="w", justify="left", wraplength=500, height=42), "accent")
        self.partial_label.grid(row=2, column=0, sticky="ew", pady=(0, 4))
        self.preview_title.grid_remove()
        self.partial_label.grid_remove()
        capture_body.grid_remove()
        # Task details occupy the middle of the workbench; their long metadata
        # scrolls independently from the timeline and current-segment label.
        self.task_monitor = TaskMonitor(self.capture_panel, self, height=164)
        self.task_monitor.grid(row=2, column=0, sticky="nsew", padx=6, pady=(4, 4))
        self.task_monitor.grid_propagate(False)
        if self.dnd_available:
            for widget in (self.capture_panel, self.task_monitor, self.task_monitor.detail,
                           self.task_monitor.detail._label, self.task_monitor.canvas, self.file_button):
                widget.drop_target_register(DND_FILES)
                widget.dnd_bind("<<Drop>>", self._drop)
        self.editor_panel = self.style(ctk.CTkFrame(self.vertical, corner_radius=0), "frame")
        self.editor_panel.grid_columnconfigure(0, weight=1)
        self.editor_panel.grid_rowconfigure(1, weight=1)
        self.vertical.add(self.capture_panel, minsize=165, stretch="never")
        self.vertical.add(self.editor_panel, minsize=180, stretch="always")
        self.editor_header = self.style(ctk.CTkFrame(self.editor_panel), "frame")
        self.editor_header.grid(row=0, column=0, sticky="ew", pady=(6, 8))
        self.editor_header.grid_columnconfigure(0, weight=1)
        self.record_label = self.style(ctk.CTkLabel(self.editor_header, text="转写文本", anchor="w", font=self.font("section")), "title")
        self.record_label.grid(row=0, column=0, columnspan=3, sticky="ew", pady=(0, 8))
        self.save_button = self.style(ctk.CTkButton(self.editor_header, text="保存修改", width=88, command=self._save), "secondary")
        self.save_button.grid(row=1, column=0, padx=(0, 5), sticky="ew")
        self.export_button = self.style(ctk.CTkButton(self.editor_header, text="导出", width=66, command=self._export), "secondary")
        self.export_button.grid(row=1, column=1, padx=5, sticky="ew")
        self.delete_button = self.style(ctk.CTkButton(self.editor_header, text="删除", width=60, command=self._delete), "secondary")
        self.delete_button.grid(row=1, column=2, padx=(5, 0), sticky="ew")
        for column in range(3):
            self.editor_header.grid_columnconfigure(column, weight=1, uniform="editor-actions")
        from ui.tooltip import attach_tooltip
        self._toolbar_groups = ((controls, ((self.file_button, "选择文件"), (self.mic_button, "开始听写"), (self.stop_button, "停止"))), (self.editor_header, ((self.save_button, "保存修改"), (self.export_button, "导出"), (self.delete_button, "删除"))))
        for _, buttons in self._toolbar_groups:
            for button, source in buttons:
                attach_tooltip(button, self, source)
        self.editor = self.style(ctk.CTkTextbox(self.editor_panel, font=self.font("editor"), wrap="word", corner_radius=10, border_width=1), "textbox")
        self.editor.grid(row=1, column=0, sticky="nsew")
        self.activity_log = ActivityLog(self.editor_panel, self, height=112)
        self.activity_log.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        self.activity_log.grid_propagate(False)
        footer = self.style(ctk.CTkFrame(self.content, corner_radius=0), "frame")
        footer.grid(row=1, column=0, sticky="ew", padx=20, pady=(0, 12))
        footer.grid_columnconfigure(0, weight=1)
        self.progress = self.style(ctk.CTkProgressBar(footer, height=4, mode="indeterminate"), "progress")
        self.progress.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        self.progress.set(0)
        self.status = self.style(ctk.CTkLabel(footer, text="就绪 · 先配置本地模型，再导入文件或开始听写", font=self.font("small"), anchor="w", justify="left", wraplength=550), "muted")
        self.status.grid(row=1, column=0, sticky="ew")
        self._status_content = ("就绪 · 先配置本地模型，再导入文件或开始听写", {})
        attach_tooltip(self.settings_button, self, "配置")
        attach_tooltip(self.status, self, lambda: self.tr(self._status_content[0], **self._status_content[1]))
        self._wrap_labels = [(sidebar_hint, sidebar, 40), (self.partial_label, capture_body._parent_canvas, 24), (self.drop_label, capture_body._parent_canvas, 96), (self.format_label, capture_body._parent_canvas, 56)]
        self._settings = SettingsPage(self.content, self)
        self._settings.grid(row=0, column=0, sticky="nsew", padx=20, pady=18)
        self._settings.grid_remove()
        for pane, key in ((self.horizontal, "sidebar_ratio"), (self.vertical, "editor_ratio")):
            pane.bind("<Configure>", self._pane_configured, add="+")
            pane.bind("<ButtonPress-1>", lambda event, p=pane: self._sash_pressed(p, event), add="+")
            pane.bind("<ButtonRelease-1>", lambda event, p=pane, k=key: self._sash_released(p, k), add="+")

    def show_transcription(self):
        self._settings.grid_remove()
        self.transcription_page.grid()
        self._page = "transcription"
        self.translate_widget(self.settings_button, "配置")
        self.theme.style(self.settings_button, "secondary")
        self.decorate(self.settings_button, "settings")
        self.after_idle(self._place_sashes)

    def apply_appearance(self, mode, color):
        self.theme.apply(mode, color)
        self.icons.apply(self.theme.palette, self.typography.base_size)

    def apply_preferences(self, mode, color, font_size, ui_language):
        self.apply_appearance(mode, color)
        self.typography.apply(font_size)
        self.icons.apply(self.theme.palette, self.typography.base_size)
        for widget, lines in list(self._sized_controls.items()):
            if widget.winfo_exists():
                widget.configure(height=max(28, widget.cget("font").metrics("linespace") * lines + 12))
        self.translator.set_language(ui_language)
        self.title(self.tr("MySound · 本地语音转文字"))
        self._settings.retranslate()
        self.task_monitor.refresh_theme()
        self.activity_log.refresh_theme()
        self.task_monitor.retranslate()
        self.activity_log.retranslate()
        self._update_model_label()
        self.after_idle(self._resize_text)

    def _is_maximized(self):
        try:
            return self.state() == "zoomed" or bool(self.attributes("-zoomed"))
        except tk.TclError:
            return self.state() == "zoomed"

    def _restore_layout(self):
        if self._layout["maximized"]:
            try:
                self.attributes("-zoomed", True)
            except tk.TclError:
                self.state("zoomed")
        self._layout_ready = True
        self._place_sashes()

    def _place_sashes(self):
        if not self._layout_ready:
            return
        scale = self.sidebar._get_widget_scaling()
        width = self.horizontal.winfo_width()
        left_min, right_min = round(220 * scale), round(600 * scale)
        if width < left_min + right_min + 8:
            factor = max(0.1, (width - 8) / (left_min + right_min))
            left_min, right_min = int(left_min * factor), int(right_min * factor)
        if self.horizontal not in self._dragging and width > 10:
            for widget, minimum in ((self.sidebar, left_min), (self.content, right_min)):
                if int(self.horizontal.panecget(widget, "minsize")) != minimum:
                    self.horizontal.paneconfigure(widget, minsize=minimum)
            self.horizontal.sash_place(0, clamp_sash(width, self._layout["sidebar_ratio"], left_min, right_min), 0)
        if self._page == "transcription":
            height = self.vertical.winfo_height()
            # CTk heights are logical pixels. On compact/scaled displays keep
            # the metadata scrollable and reserve space for readable text.
            compact_status = self.winfo_height() / max(1, scale) < 680
            if (compact_status, scale) != getattr(self, "_workbench_geometry", None):
                self._workbench_geometry = (compact_status, scale)
                self._compact_workbench = compact_status
                self.transcription_page.grid_configure(pady=4 if compact_status else 18)
                self.record_label.grid_remove() if compact_status else self.record_label.grid()
                self.model_label.grid_remove() if compact_status else self.model_label.grid()
                if compact_status:
                    self.workspace_title.grid_remove()
                elif self.typography.base_size < 32:
                    self.workspace_title.grid()
            # CustomTkinter reapplies saved grid arguments after DPI changes.
            # Reassert compact visibility once that reflow has completed.
            if compact_status:
                if self.record_label.winfo_manager() == "grid":
                    self.record_label.grid_remove()
                if self.workspace_title.winfo_manager() == "grid":
                    self.workspace_title.grid_remove()
                if self.model_label.winfo_manager() == "grid":
                    self.model_label.grid_remove()
            self.task_monitor.set_compact(compact_status)
            monitor_height = 96 if compact_status else 164
            log_height = 72 if compact_status else 112
            if self.task_monitor.cget("height") != monitor_height:
                self.task_monitor.configure(height=monitor_height)
            if self.activity_log.cget("height") != log_height:
                self.activity_log.configure(height=log_height)
            upper_min = self.capture_controls.winfo_reqheight() + self.task_monitor.winfo_reqheight() + round(20 * scale)
            lower_min = self.editor_header.winfo_reqheight() + self.activity_log.winfo_reqheight() + 100
            if height < upper_min + lower_min + 8:
                factor = max(0.1, (height - 8) / (upper_min + lower_min))
                upper_min, lower_min = int(upper_min * factor), int(lower_min * factor)
            if self.vertical not in self._dragging and height > 10:
                for widget, minimum in ((self.capture_panel, upper_min), (self.editor_panel, lower_min)):
                    if int(self.vertical.panecget(widget, "minsize")) != minimum:
                        self.vertical.paneconfigure(widget, minsize=minimum)
                self.vertical.sash_place(0, 0, clamp_sash(height, self._layout["editor_ratio"], upper_min, lower_min))
        self._resize_text()

    def _pane_configured(self, _event):
        if self._resize_timer:
            self.after_cancel(self._resize_timer)
        self._resize_timer = self.after(40, self._finish_resize)

    def _finish_resize(self):
        self._resize_timer = None
        self._place_sashes()

    def _sash_pressed(self, pane, event):
        if pane.identify(event.x, event.y):
            self._dragging.add(pane)

    def _sash_released(self, pane, key):
        if pane not in self._dragging:
            return
        self._dragging.discard(pane)
        coordinate = pane.sash_coord(0)[0 if pane is self.horizontal else 1]
        total = pane.winfo_width() if pane is self.horizontal else pane.winfo_height()
        self._layout[key] = coordinate / max(1, total)
        self._resize_text()
        self._schedule_layout_save()

    def _window_configured(self, event):
        if event.widget is self and self._layout_ready:
            if not self._is_maximized() and self.state() == "normal":
                self._normal_size = (round(self._reverse_window_scaling(self.winfo_width())), round(self._reverse_window_scaling(self.winfo_height())))
            self._schedule_layout_save()

    def _schedule_layout_save(self):
        if self._layout_timer:
            self.after_cancel(self._layout_timer)
        self._layout_timer = self.after(500, self._save_layout)

    def _save_layout(self):
        self._layout_timer = None
        if not self._layout_ready:
            return
        self._layout.update(width=self._normal_size[0], height=self._normal_size[1], maximized=self._is_maximized())
        try:
            self.config_store.update(window_layout=dict(self._layout))
        except OSError:
            logging.getLogger(__name__).exception("无法保存窗口布局")

    def reset_layout(self):
        self._layout = clamp_layout({}, self._reverse_window_scaling(self.winfo_screenwidth()), self._reverse_window_scaling(self.winfo_screenheight()))
        self._normal_size = (self._layout["width"], self._layout["height"])
        try:
            self.attributes("-zoomed", False)
        except tk.TclError:
            self.state("normal")
        self.geometry(f"{self._layout['width']}x{self._layout['height']}")
        self.after_idle(self._place_sashes)
        self._schedule_layout_save()

    def _resize_text(self):
        for label, container, padding in self._wrap_labels:
            scale = label._get_widget_scaling()
            wrap = max(100, int(container.winfo_width() / scale) - padding)
            if label.cget("wraplength") != wrap:
                label.configure(wraplength=wrap)
        self._resize_history()
        self._fit_toolbars()
        brand_width = max(40, self.sidebar.winfo_width() / self.sidebar._get_widget_scaling() - 40)
        self.brand_label.configure(text="MySound" if self.brand_label.cget("font").measure("MySound") <= brand_width else "MS")
        compact = self.typography.base_size >= 32
        drop_source = ("拖入音频或视频" if self.dnd_available else "选择文件") if compact else self._drop_source
        if drop_source != getattr(self, "_drop_display_source", None):
            self._drop_display_source = drop_source
            self.translate_widget(self.drop_label, drop_source)
            self.decorate(self.drop_label, "upload", tone="accent_text")
        if compact != getattr(self, "_sidebar_compact", False):
            self._sidebar_compact = compact
            self.sidebar_hint.grid_remove() if compact else self.sidebar_hint.grid()
            self.workspace_title.grid_remove() if compact or getattr(self, "_compact_workbench", False) else self.workspace_title.grid()
        settings_width = self.settings_button.winfo_width() / self.settings_button._get_widget_scaling()
        self.translator.unbind(self.settings_button)
        settings_text = self.tr("配置")
        self.settings_button.configure(text="" if self.font().measure(settings_text) + self.icons.size + 40 > settings_width else settings_text)
        search_width = self.search.winfo_width() / self.search._get_widget_scaling() - 24
        self.search.configure(placeholder_text=ellipsize(self.tr("搜索历史记录…"), search_width, self.search.cget("font").measure))
        for label, text in ((self.model_label, getattr(self, "_full_model_caption", "")),
                            (self.status, self.tr(self._status_content[0], **self._status_content[1]))):
            width = max(40, label.winfo_width() / label._get_widget_scaling() - 12)
            caption = ellipsize(text, width, label.cget("font").measure)
            if label.cget("text") != caption or label.cget("wraplength"):
                label.configure(text=caption, wraplength=0)
        scale = self.record_label._get_widget_scaling()
        available = self.editor_header.winfo_width() / scale - 50
        self.record_label.configure(text=ellipsize(self._source_title(self._record_title), max(20, available), self.record_label.cget("font").measure))

    def _fit_toolbars(self):
        # At large accessibility sizes, retain all actions as icons with
        # tooltips instead of clipping labels or pushing buttons off-screen.
        for container, buttons in getattr(self, "_toolbar_groups", ()):
            width = container.winfo_width() / container._get_widget_scaling()
            available = max(1, (width - 24) / len(buttons))
            compact = any(button.cget("font").measure(self.tr(source)) + 64 > available for button, source in buttons)
            for button, source in buttons:
                self.translator.unbind(button)
                caption = "" if compact else self.tr(source)
                if button.cget("text") != caption:
                    button.configure(text=caption)

    def _source_title(self, source):
        return self.tr(source) if source in {"转写文本", "麦克风", "手动文本"} else source

    def _set_record_title(self, text):
        self._record_title = text
        self._resize_text()

    def _work_directory(self):
        path = Path(self.config_store.load().get("work_dir") or Path.home()).expanduser()
        return str(path if path.is_dir() else Path.home())

    def _show_error(self, exc):
        logging.getLogger(__name__).error("%s", exc)
        from ui.errors import concise_error
        text = concise_error(self.tr_error(exc), self.tr)
        self.translate_widget(self.status, text)
        if hasattr(self, "activity_log"):
            self.activity_log.append(text, "ERROR")
        log_path = self.data_selection.path / "mysound.log"
        messagebox.showerror("MySound", text + "\n\n" + self.tr("详细错误已记录到：{path}", path=log_path), parent=self)

    def _update_model_label(self):
        try:
            config = self.config_store.load()
            path = config.get("model_path", "")
            self._full_model_caption = self.tr("当前模型：{model} · {path} · {device}", model=config['model_type'], path=Path(path).name if path else self.tr("尚未配置"), device=config['device'])
            self.model_label.configure(text=self._full_model_caption)
            self.after_idle(self._resize_text)
        except Exception as exc:
            self.after(0, lambda error=exc: self._show_error(error))

    def _text(self):
        return self.editor.get("1.0", "end-1c")

    def _replace_text(self, text):
        self.editor.configure(state="normal")
        self.editor.delete("1.0", "end")
        self.editor.insert("1.0", text)
        self._baseline = text

    def _guard_edits(self):
        if self._text() == self._baseline:
            return True
        decision = messagebox.askyesnocancel(self.tr("保存修改"), self.tr("是否先保存当前文本的修改？"), parent=self)
        if decision is None:
            return False
        return self._save() if decision else True

    def _save(self):
        if self._job_ui_active:
            return False
        try:
            text = self._text()
            if self.current_record:
                self.current_record = self.history_store.update(self.current_record["id"], text)
            elif text.strip():
                self.current_record = self.history_store.add(source="手动文本", model="无", text=text, segments=[])
            self._baseline = text
            self._refresh_history()
            self._status("修改已保存")
            return True
        except Exception as exc:
            self._show_error(exc)
            return False

    def _refresh_history(self):
        try:
            query = self.search.get()
            records = self.history_store.list(query)
            canvas = self.history_list._parent_canvas
            self._history_scroll_target = canvas.yview()[0] if query == self._history_query else 0.0
            self._history_query = query
            visible_ids = {record["id"] for record in records}
            for record_id in list(self._history_items):
                if record_id not in visible_ids:
                    self._history_items.pop(record_id).destroy()
            self._history_buttons = []
            if not records:
                if self._history_empty is None:
                    self._history_empty = self.style(ctk.CTkLabel(self.history_list, text="暂无记录", font=self.font("small")), "muted")
                    self.decorate(self._history_empty, "history", tone="muted", compound="top")
                    self._history_empty.pack(pady=28)
            elif self._history_empty is not None:
                self._history_empty.destroy()
                self._history_empty = None
            for record in records:
                selected = self.current_record and record["id"] == self.current_record["id"]
                title = Path(record["source"]).name
                date = record["created_at"][:16].replace("T", " ")
                role = "selected_history" if selected else "history"
                button = self._history_items.get(record["id"])
                if button is None:
                    # Give rows real text before geometry has settled. A new
                    # row can initially report width=1 without another parent
                    # Configure event when it replaces an equal-sized row.
                    width = max(80, self.history_list.winfo_width() / self.sidebar._get_widget_scaling() - 60)
                    caption = self._history_caption(title, date, width, self.font())
                    button = self.style(ctk.CTkButton(self.history_list, text=caption, width=1, anchor="w", corner_radius=10), role)
                    self._sized_controls[button] = 2
                    button.configure(height=button.cget("font").metrics("linespace") * 2 + 20)
                    self._history_items[record["id"]] = button
                    button.bind("<Configure>", self._schedule_history_layout, add="+")
                    button.bind("<Map>", self._schedule_history_layout, add="+")
                else:
                    self.theme.style(button, role)
                self.decorate(button, "mic" if record["source"] == "麦克风" else "file", tone="accent_text" if selected else "muted")
                button.configure(command=lambda r=record: self._select_record(r), state="disabled" if self._job_ui_active else "normal")
                # Reorder reused rows without destroying them or their canvas.
                button.pack_forget()
                button.pack(fill="x", padx=2, pady=4)
                self._history_buttons.append((button, title, date))
            self._schedule_history_layout()
        except Exception as exc:
            self._show_error(exc)

    def _history_caption(self, title, date, width, font):
        return ellipsize(self._source_title(title), width, font.measure) + "\n" + ellipsize(date, width, font.measure)

    def _resize_history(self):
        if self._history_empty is not None:
            width = max(80, self.history_list.winfo_width() / self.sidebar._get_widget_scaling() - 24)
            if self._history_empty.cget("wraplength") != width:
                self._history_empty.configure(wraplength=width)
        for button, title, date in self._history_buttons:
            scale = button._get_widget_scaling()
            width = max(30, (button.winfo_width() if button.winfo_width() > 1 else self.history_list.winfo_width()) / scale - 60)
            caption = self._history_caption(title, date, width, button.cget("font"))
            if button.cget("text") != caption:
                button.configure(text=caption)

    def _schedule_history_layout(self, _event=None):
        if self._history_timer is None:
            self._history_timer = self.after_idle(self._finish_history_layout)

    def _finish_history_layout(self):
        self._history_timer = None
        self._resize_history()
        canvas = self.history_list._parent_canvas
        canvas.configure(scrollregion=canvas.bbox("all"))
        if getattr(self, "_history_scroll_target", None) is not None:
            canvas.yview_moveto(self._history_scroll_target)
            self._history_scroll_target = None

    def _search_changed(self, _event):
        if self._search_timer:
            self.after_cancel(self._search_timer)
        self._search_timer = self.after(200, self._refresh_history)

    def _select_record(self, record):
        if self._job_ui_active or not self._guard_edits():
            return
        self.current_record = self.history_store.get(record["id"])
        if self.current_record:
            self._replace_text(self.current_record["text"])
            self._set_record_title(Path(self.current_record["source"]).name)
            self.show_transcription()
            self._refresh_history()

    def _set_busy(self, busy):
        self._job_ui_active = busy
        for widget in (self.file_button, self.mic_button, self.save_button, self.export_button, self.delete_button):
            widget.configure(state="disabled" if busy else "normal")
        self.stop_button.configure(state="normal" if busy else "disabled")
        self.editor.configure(state="disabled" if busy else "normal")
        self._settings.set_busy(busy)
        if busy:
            self.progress.start()
        else:
            self.progress.stop()
            self.progress.set(0)
        self._refresh_history()

    def _prepare_job(self, source):
        if self._job_ui_active or not self._guard_edits():
            return None
        config = self.config_store.load()
        if not config.get("model_path"):
            self._open_settings()
            self._status("请先在设置中选择模型路径并保存。")
            return None
        self.current_record = None
        self._replace_text("")
        self._set_record_title(Path(source).name)
        self.show_transcription()
        self._partial("准备中…")
        self._set_busy(True)
        return config

    def _choose_file(self):
        if self._job_ui_active:
            return
        path = filedialog.askopenfilename(parent=self, title=self.tr("选择音频或视频"), initialdir=self._work_directory(), filetypes=[(self.tr("音频与视频"), "*.wav *.mp3 *.m4a *.flac *.ogg *.opus *.aac *.mp4 *.mkv *.webm *.wma"), (self.tr("所有文件"), "*")])
        if path:
            self._start_file(path)

    def _drop(self, event):
        paths = self.tk.splitlist(event.data)
        if paths and not self._job_ui_active:
            if len(paths) > 1:
                self._status("一次处理一个文件，已选择第一个文件。")
            self._start_file(paths[0])
        return "copy"

    def _start_file(self, path):
        try:
            if not Path(path).is_file():
                raise ValueError("请拖入一个音频或视频文件。")
            config = self._prepare_job(path)
            if config is not None:
                self.controller.start_file(path, config)
        except Exception as exc:
            self._set_busy(False)
            self._show_error(exc)

    def _start_mic(self):
        try:
            config = self._prepare_job("麦克风")
            if config is not None:
                self.controller.start_mic(config)
        except Exception as exc:
            self._set_busy(False)
            self._show_error(exc)

    def _stop(self):
        self.stop_button.configure(state="disabled")
        self.controller.stop()

    def _poll_events(self):
        try:
            for _ in range(100):
                event = self.events.get_nowait()
                kind = event["kind"]
                if kind in {"task_started", "media_info", "mic_info", "chunk", "done", "failed", "cancelled"}:
                    self.task_monitor.handle_event(event)
                    self._log_task_event(event)
                if kind == "task_started":
                    self._capture_mode = event.get("mode", "file")
                    if event.get("mode") == "mic":
                        self.capture_body.grid()
                        self.capture_panel.grid_rowconfigure(0, weight=1)
                        self.drop_area.grid_remove()
                        self.preview_title.grid()
                        self.partial_label.grid()
                    else:
                        self.capture_body.grid_remove()
                        self.capture_panel.grid_rowconfigure(0, weight=0)
                        self.preview_title.grid_remove()
                        self.partial_label.grid_remove()
                if kind == "status":
                    self._status(event["text"], _timestamp=event.get("timestamp"), _level=event.get("level", "INFO"), **event.get("values", {}))
                elif kind == "partial":
                    self._partial(event["text"][-220:], transcript=True) if event["text"] else self._partial("正在聆听…")
                elif kind == "segment":
                    self.editor.configure(state="normal")
                    if self._text():
                        self.editor.insert("end", "\n")
                    self.editor.insert("end", event["segment"]["text"].strip())
                    self.editor.see("end")
                    self.editor.configure(state="disabled")
                elif kind == "error":
                    error_text = event.get("text_en") if self.ui_language == "en_US" and event.get("text_en") else event["text"]
                    self._show_error(error_text)
                    details = str(event.get("details") or "").strip()
                    if details and hasattr(self, "activity_log"):
                        self.activity_log.append(details, "ERROR", event.get("timestamp"))
                elif kind == "done":
                    self.current_record = event["record"]
                    if self.current_record is not None:
                        self._baseline = self._text()
                    self._set_busy(False)
                    self._partial("实时听写将在这里显示…")
                    if not event["failed"]:
                        text = "已停止" if event["cancelled"] else "转写完成"
                        self._status(text + (" · 已保存到历史记录" if self.current_record else " · 未检测到语音"))
                    if self._closing:
                        self._closing = False
                        self._close()
                        return
                elif kind == "settings_result":
                    dialog = event["dialog"]
                    if dialog.winfo_exists():
                        dialog.accept_result(event)
        except queue.Empty:
            pass
        self._poll_timer = self.after(60, self._poll_events)

    def _log_task_event(self, event):
        """Turn worker lifecycle events into short, readable activity entries."""
        if not hasattr(self, "activity_log"):
            return
        kind = event.get("kind")
        if kind == "task_started":
            mode = self.tr("麦克风" if event.get("mode") == "mic" else "文件")
            self.activity_log.append(f"{self.tr('开始任务')} · {mode} · {event.get('source', '')}", "INFO", event.get("timestamp"))
        elif kind == "media_info":
            metadata = event.get("metadata") or {}
            self.activity_log.append(f"{self.tr('已读取文件信息')} · {metadata.get('name') or metadata.get('path', '')}", "INFO", event.get("timestamp"))
        elif kind == "mic_info":
            metadata = event.get("metadata") or {}
            self.activity_log.append(f"{self.tr('麦克风已连接')} · {metadata.get('name', '')} · {metadata.get('capture_sample_rate', '—')} Hz", "INFO", event.get("timestamp"))
        elif kind == "chunk":
            chunk = event.get("chunk") or {}
            status = chunk.get("status")
            if status in {"processing", "done", "error", "cancelled", "skipped"}:
                status_label = {
                    "processing": "处理中",
                    "done": "完成",
                    "error": "错误",
                    "cancelled": "取消",
                    "skipped": "静音跳过",
                }.get(status, "处理中")
                self.activity_log.append(
                    f"{self.tr('切片')} {chunk.get('id', '—')} · {self.tr(status_label)} · "
                    f"{float(chunk.get('start', 0)):.1f}–{float(chunk.get('end', 0)):.1f}s",
                    "ERROR" if status == "error" else ("WARNING" if status == "skipped" else "INFO"),
                    event.get("timestamp"),
                )
        elif kind == "done":
            level = "ERROR" if event.get("failed") else ("WARNING" if event.get("cancelled") else "INFO")
            self.activity_log.append(self.tr("任务结束"), level, event.get("timestamp"))

    def _export(self):
        if self._job_ui_active or not self._text().strip():
            return
        path = filedialog.asksaveasfilename(parent=self, title=self.tr("导出文本或字幕"), initialdir=self._work_directory(), defaultextension=".txt", filetypes=[(self.tr("纯文本"), "*.txt"), ("Markdown", "*.md"), (self.tr("SRT 字幕"), "*.srt")])
        if not path:
            return
        try:
            record = dict(self.current_record or {"source": "文本", "model": "", "segments": [], "created_at": ""})
            record["text"] = self._text()
            export_record(record, path)
            self._status("已导出：{path}", path=path)
        except Exception as exc:
            self._show_error(exc)

    def _delete(self):
        if self._job_ui_active or not self.current_record:
            return
        if messagebox.askyesno(self.tr("删除记录"), self.tr("确定删除这条历史记录及其文本？"), parent=self):
            try:
                self.history_store.delete(self.current_record["id"])
                self.current_record = None
                self._replace_text("")
                self._set_record_title("转写文本")
                self._refresh_history()
            except Exception as exc:
                self._show_error(exc)

    def _open_settings(self):
        self.transcription_page.grid_remove()
        self._settings.grid()
        self._page = "settings"
        self.translate_widget(self.settings_button, "配置")
        self.theme.style(self.settings_button, "button")
        self.decorate(self.settings_button, "settings", tone="on_accent")
        self._settings.set_busy(self._job_ui_active)

    def _close(self):
        if self._job_ui_active:
            if messagebox.askyesno(self.tr("停止并退出"), self.tr("正在转写，是否停止并保存已有结果后退出？"), parent=self):
                self._closing = True
                self._stop()
            return
        if self._guard_edits():
            self._save_layout()
            self.destroy()

    def destroy(self):
        # Cancel our callbacks before Tcl widgets are destroyed (also useful for
        # GUI tests creating more than one application in a single process).
        for timer in (self._poll_timer, self._layout_timer, self._resize_timer, self._search_timer, self._history_timer):
            if timer:
                self.after_cancel(timer)
        super().destroy()
