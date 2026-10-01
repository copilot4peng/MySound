"""In-window settings with separate validation and save actions per section."""

from __future__ import annotations

import math
from pathlib import Path
import threading
import tkinter as tk
from tkinter import filedialog

import customtkinter as ctk

from core.model_manager import ModelManager
from core.model_paths import resolve_model_path
from ui.i18n import register_messages
from ui.layout import ellipsize
from ui.typography import FONT_SIZES
from ui.section_tabs import SectionTabs
from ui.system_panel import ModelInfoPanel, SystemPanel
from ui.theme_cards import ThemeCards


MODEL_LABELS = {"Qwen3-ASR": "qwen3", "OpenAI Whisper": "whisper", "Sherpa-ONNX Whisper": "whisper_onnx", "Confucius4-R2T2": "confucius"}
LANGUAGES = {"自动识别": None, "中文": "Chinese", "英语": "English", "日语": "Japanese", "韩语": "Korean"}
APPEARANCES = {"深色": "dark", "浅色": "light"}
COLORS = {"蓝色": "blue", "绿色": "green", "紫色": "purple", "橙色": "orange"}
UI_LANGUAGES = {"简体中文": "zh_CN", "English": "en_US"}
SECTIONS = ("模型", "目录", "音频", "外观", "本机配置")
SECTION_ICONS = {"模型": "model", "目录": "folder", "音频": "microphone", "外观": "palette", "本机配置": "hardware"}

# Exported for the application's shared translator; source strings remain
# stable even when an option's displayed label changes between languages.
SETTINGS_EN = {
    "设置": "Settings", "返回转写": "Back to transcription", "模型": "Models",
    "目录": "Folders", "音频": "Audio", "外观": "Appearance", "浏览": "Browse",
    "保存模型设置": "Save model settings", "保存目录设置": "Save folder settings",
    "保存音频设置": "Save audio settings", "保存外观设置": "Save appearance",
    "自动识别": "Auto-detect", "中文": "Chinese", "英语": "English", "日语": "Japanese",
    "韩语": "Korean", "深色": "Dark", "浅色": "Light", "蓝色": "Blue", "绿色": "Green",
    "紫色": "Purple", "橙色": "Orange", "简体中文": "Simplified Chinese",
    "当前使用的模型": "Active model", "此模型的本地路径": "Local path for this model",
    "选择模型文件夹": "Choose model folder", "选择模型文件": "Choose model file",
    "模型配置或权重": "Model configuration or weights", "所有文件": "All files",
    "选择文件夹": "Choose a folder", "运行设备": "Compute device", "识别语言": "Speech language",
    "自动选择": "Automatic", "扫描根目录": "Root folder to scan", "扫描本地模型": "Scan local models",
    "每种模型分别记住自己的路径。切换类型后可选择或修改对应路径；保存后下次转写生效。":
        "Each model keeps its own path. Switch models to edit their paths; saved changes apply to the next transcription.",
    "扫描后将列出完整路径。请点击需要使用的模型，扫描不会改变当前选择。":
        "Scanning lists full model paths. Click a model to select it; scanning does not change your current selection.",
    "Whisper 支持完整 .pt 文件或 Hugging Face 目录；Qwen3 / Confucius 需包含 config.json、分词器和完整权重。压缩包请先解压，程序不会下载模型。":
        "Whisper accepts complete .pt files or Hugging Face folders. Qwen3 / Confucius need config.json, tokenizer files, and complete weights. Extract archives first; models are never downloaded automatically.",
    "可选择模型文件夹，或其中的 config.json、权重或分片索引文件。父目录只含一个可用的同类型模型时会自动定位。":
        "Choose a model folder or its config.json, weight, or weight-index file. A parent folder containing one ready model of the selected type is resolved automatically.",
    "默认工作目录": "Default working folder", "下次启动使用的数据目录": "Data folder for the next launch",
    "仅作为导入音频和导出文本时的默认文件夹。模型、配置和历史记录分别使用各自的目录。":
        "Used as the default folder when importing audio or exporting text. Models, settings, and history use their own folders.",
    "启动参数 --data-dir": "--data-dir launch argument", "环境变量 MYSOUND_DATA_DIR": "MYSOUND_DATA_DIR environment variable",
    "已保存的目录选择": "Saved folder preference", "系统默认目录": "System default folder",
    "当前数据目录：\n{path}\n来源：{origin}": "Current data folder:\n{path}\nSource: {origin}",
    "配置文件：{config}\n历史文件：{history}": "Settings file: {config}\nHistory file: {history}",
    "本次启动通过命令行或环境变量指定了数据目录，页面内暂不能更改。移除启动覆盖后，再使用这里保存的选择。工作目录仍可单独保存。":
        "The data folder was set by a launch argument or environment variable. Remove that override to use the saved folder preference. The working folder can still be changed separately.",
    "保存数据目录选择后需重启应用。旧目录中的配置、历史和日志会保留；程序不会迁移或覆盖它们。新目录已有的数据将在重启后读取。":
        "Restart after saving a new data folder. Existing settings, history, and logs stay in their original folder. Existing data in the new folder will be read after restart.",
    "系统默认": "System default", "设备 {index}": "Device {index}",
    "设备 {index}（当前未发现）": "Device {index} (currently unavailable)",
    "麦克风输入设备": "Microphone input", "刷新录音设备": "Refresh microphones",
    "听写刷新间隔（秒，1–5）": "Dictation update interval (seconds, 1–5)",
    "文件目标切片时长（秒，5–60）": "Target file segment length (seconds, 5–60)",
    "文件最长切片时长（秒，目标时长–60）": "Maximum segment length (seconds, target–60)",
    "文件在自然停顿处切片。实时听写按短句更新，实际速度取决于模型、设备和语句长度；短刷新间隔会增加推理频率。":
        "Files are split at natural pauses. Dictation updates by utterance; speed depends on the model, device, and utterance length. Shorter intervals run inference more often.",
    "明暗模式": "Appearance mode", "主题颜色": "Accent color", "界面字号": "Interface font size",
    "界面语言": "Interface language", "恢复默认布局": "Restore default layout",
    "保存后立即应用到整个界面，下次启动也会保留。转写进行中仍可调整外观。":
        "Saved changes apply immediately and persist across restarts. Appearance can be changed during transcription.",
    "恢复默认窗口和分栏尺寸，不改变模型、目录、文本和主题设置。":
        "Restore the default window and pane sizes. Model paths, folders, text, and appearance preferences are preserved.",
    "正在读取…": "Reading…", "请选择存在的模型根目录后扫描。": "Choose an existing model root folder before scanning.",
    "找到 {count} 项。请点击选择，再保存；当前选择未改变。": "Found {count} items. Select one, then save; your current selection is unchanged.",
    "找到 {count} 个录音设备。": "Found {count} microphone devices.",
    "未找到模型，可以在上方直接输入或浏览路径。": "No models found. Enter or browse a path above.",
    "本地模型": "Local model", "无法判断此模型的类型，请在上方手动选择类型和路径。":
        "The model type could not be detected. Select its type and path above.",
    "已定位模型：{path}。保存后生效。": "Resolved model: {path}. Save to apply.",
    "已选择 {name}。{status}；保存后生效。": "Selected {name}. {status}; save to apply.",
    "正在转写：模型、目录和音频设置暂不能保存。可返回转写页面停止任务，外观设置仍可使用。":
        "Transcription is running. Model, folder, and audio changes cannot be saved until it stops. Appearance settings remain available.",
    "各类别独立保存；可随时返回转写页面。": "Save each section separately. You can return to transcription at any time.",
    "请先返回转写页面停止任务或等待任务完成。": "Stop the task on the transcription page or wait for it to finish.",
    "模型设置已保存，下次转写时使用。": "Model settings saved for the next transcription.",
    "请选择默认工作目录。": "Choose a default working folder.",
    "默认工作目录不存在或不是文件夹。": "The working folder does not exist or is not a directory.",
    "工作目录已保存。": "Working folder saved.",
    "工作目录已保存。数据目录选择已保存，重启后生效；旧数据保留在原目录。":
        "Working and data folders saved. Restart to use the data folder; existing data stays in its original folder.",
    "音频设置已保存，下次转写时使用。": "Audio settings saved for the next transcription.",
    "外观设置已保存并应用。": "Appearance settings saved and applied.", "已恢复默认布局。": "Default layout restored.",
    "音频时间请填写有效数字。": "Enter valid numbers for audio timing.",
    "音频时间请填写有限的有效数字。": "Audio timing values must be finite numbers.",
    "切片时间需满足：5 ≤ 目标 ≤ 最长 ≤ 60 秒。": "Segment lengths must satisfy: 5 ≤ target ≤ maximum ≤ 60 seconds.",
    "听写刷新间隔请设为 1–5 秒。": "Set the dictation update interval to 1–5 seconds.",
    "请选择有效的界面字号。": "Choose a valid interface font size.",
    "返回": "Back", "保存": "Save", "主题预设": "Theme presets", "本机配置": "This PC",
    "每个预设同时调整背景、面板、边框与强调色。选择后保存外观设置生效。":
        "Each preset coordinates backgrounds, panels, borders, and accents. Choose a preset, then save appearance settings.",
    "本机配置为只读信息，无需保存。": "This PC is read-only; no save is needed.",
    "查看本机建议": "View hardware suggestions",
    "支持 Whisper tiny、base、small、medium、large、turbo，Sherpa-ONNX Whisper，Qwen3-ASR 0.6B / 1.7B 和 Confucius4-R2T2。":
        "Supports Whisper tiny, base, small, medium, large, turbo; Sherpa-ONNX Whisper; Qwen3-ASR 0.6B / 1.7B; and Confucius4-R2T2.",
    "Sherpa-ONNX Whisper 选择包含 encoder、decoder 和 tokens 的目录，或具体 .onnx 文件。目录默认优先 INT8 以节省内存；选择文件则沿用对应精度。当前后端仅使用 CPU，auto 也使用 CPU。":
        "For Sherpa-ONNX Whisper, choose a folder containing encoder, decoder, and tokens, or a specific .onnx file. Folders prefer INT8 to reduce memory use; choosing a file preserves its precision. This backend uses CPU only, including auto mode.",
    "ONNX 模型文件": "ONNX model files",
    "转写中，模型、目录和音频设置暂停保存；外观及本机配置仍可使用。":
        "During transcription, model, folder, and audio changes cannot be saved. Appearance and This PC remain available.",
}
register_messages(SETTINGS_EN)


def validate_model_path(raw_path, model_type):
    return str(resolve_model_path(raw_path, model_type))


def validate_audio_values(target, maximum, block):
    try:
        target, maximum, block = float(target), float(maximum), float(block)
    except (TypeError, ValueError) as exc:
        raise ValueError("音频时间请填写有效数字。") from exc
    if not all(math.isfinite(value) for value in (target, maximum, block)):
        raise ValueError("音频时间请填写有限的有效数字。")
    if not 5 <= target <= maximum <= 60:
        raise ValueError("切片时间需满足：5 ≤ 目标 ≤ 最长 ≤ 60 秒。")
    if not 1 <= block <= 5:
        raise ValueError("听写刷新间隔请设为 1–5 秒。")
    return target, maximum, block


class SettingsPage(ctk.CTkFrame):
    """Settings stay in the main window; scrolling never hides save actions."""

    def __init__(self, parent, app):
        super().__init__(parent, corner_radius=0)
        self.app = app
        self.app.style(self, "frame")
        self.config = app.config_store.load()
        self._busy = bool(app._job_ui_active)
        self._notes = {}
        self._note_sources = {}
        self._choice_specs = []
        self._choice_maps = {}
        self._tab_names = {name: app.tr(name) for name in SECTIONS}
        self._model_row_labels = []
        self._buttons = []
        self._inputs = []
        self._model_detail_source = ("", {})
        self._save_buttons = {}
        self._request_ids = {"scan": 0, "devices": 0}
        self._model_info_timer = None
        self._model_paths = dict(self.config.get("model_paths", {}))
        current_type = self.config.get("model_type", "qwen3")
        self._model_paths.setdefault(current_type, self.config.get("model_path", ""))
        self._active_type = current_type
        self._models = []
        self._mic_devices = {app.tr("系统默认"): None}
        self._device_records = []
        self._devices_loaded = False
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        header = self._widget(ctk.CTkFrame(self), "frame")
        header.grid(row=0, column=0, sticky="ew", padx=20, pady=(16, 0))
        header.grid_columnconfigure(0, weight=1)
        self._widget(ctk.CTkLabel(header, text="设置", anchor="w", font=app.font("title")), "title").grid(row=0, column=0, sticky="ew")
        self.back_button = self._widget(ctk.CTkButton(header, text="返回", width=96, command=app.show_transcription), "secondary")
        self.app.decorate(self.back_button, "back")
        self._fit_button(self.back_button)
        self.back_button.grid(row=0, column=1, sticky="e", padx=(12, 0))

        self.tabs = SectionTabs(self, app)
        self.tabs.grid(row=1, column=0, sticky="nsew", padx=12, pady=(12, 12))
        self._sections = {}
        for name in SECTIONS:
            tab = self.tabs.add(self._tab_names[name], source=name, icon=SECTION_ICONS[name])
            self.app.style(tab, "surface")
            tab.grid_columnconfigure(0, weight=1)
            tab.grid_rowconfigure(0, weight=1)
            body = self._widget(ctk.CTkScrollableFrame(tab), "scroll")
            body.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
            body.grid_columnconfigure(0, weight=1)
            self._sections[name] = body
            if name != "本机配置":
                self._footer(tab, name)
        self.busy_note = self._label(self._sections["模型"], "", role="muted", font=app.font("small"))
        self.busy_note.grid(row=20, column=0, sticky="ew", padx=7, pady=10)
        self._build_model()
        self._build_directories()
        self._build_audio()
        self._build_appearance()
        self.system_panel = SystemPanel(self._sections["本机配置"], self, self.model_info.set_hardware)
        self.system_panel.grid(row=0, column=0, sticky="ew", padx=7, pady=7)
        self._description(self._sections["本机配置"], 1, "本机配置为只读信息，无需保存。")
        self.path_var.trace_add("write", self._schedule_model_info)
        self.type_var.trace_add("write", self._schedule_model_info)
        self._schedule_model_info()
        self.set_busy(self._busy)

    def _widget(self, widget, role):
        widget = self.app.style(widget, role)
        if isinstance(widget, ctk.CTkButton):
            self._buttons.append(widget)
            self._fit_button(widget)
        elif isinstance(widget, (ctk.CTkEntry, ctk.CTkOptionMenu)):
            self._inputs.append(widget)
            widget.configure(height=max(28, widget.cget("font").metrics("linespace") + 12))
        return widget

    @staticmethod
    def _fit_button(button):
        font = button.cget("font")
        image = button.cget("image")
        image_width, image_height = image.cget("size") if image is not None else (0, 0)
        button.configure(width=max(76, font.measure(button.cget("text")) + image_width + (12 if image else 0) + 28),
                         height=max(28, font.metrics("linespace") + 12, image_height + 12))

    def _label(self, parent, text, role="label", text_values=None, **kwargs):
        values = text_values or {}
        label = self._widget(ctk.CTkLabel(parent, text=self.app.tr(text, **values), anchor="w", justify="left", wraplength=300, **kwargs), role)
        self.app.translate_widget(label, text, **values)
        pending = None

        def resize():
            nonlocal pending
            pending = None
            if not label.winfo_exists():
                return
            # winfo_width is the allocated widget width in physical pixels.
            # CTkLabel's wraplength is expressed in logical (unscaled) pixels.
            width = label.winfo_width()
            if width <= 1:
                return
            image = label.cget("image")
            image_space = image.cget("size")[0] + 12 if image is not None else 0
            wrap = max(40, int(width / label._get_widget_scaling()) - image_space - 6)
            if abs(float(label.cget("wraplength")) - wrap) > 1:
                label.configure(wraplength=wrap)

        def schedule(_event=None):
            nonlocal pending
            if pending is None and label.winfo_exists():
                pending = label.after_idle(resize)

        # CTkLabel.bind targets its inner canvas/text label; configuring the
        # label from its own Configure callback recursively re-enters Tk.
        # Observe the containing frame and let grid finish before measuring.
        parent.bind("<Configure>", schedule, add="+")
        def cleanup(_event=None):
            nonlocal pending
            if pending is not None:
                label.after_cancel(pending)
                pending = None

        label.bind("<Destroy>", cleanup, add="+")
        schedule()
        return label

    def _set_note(self, section, text, **values):
        # A long backend traceback/path should not enlarge the fixed footer
        # until it consumes the scrollable settings area.
        self._note_sources[section] = (text, values)
        translated = self.app.tr_error(text) if isinstance(text, BaseException) else self.app.tr(str(text), **values)
        message = " ".join(translated.split())
        note = self._notes[section]
        if isinstance(note, ctk.CTkLabel) and note.winfo_width() > 1:
            width = note.winfo_width() / note._get_widget_scaling() - 8
            message = ellipsize(message, max(40, width), note.cget("font").measure)
        else:
            message = message if len(message) <= 72 else message[:71] + "…"
        if note.cget("text") != message:
            note.configure(text=message)

    def _set_model_note(self, text, **values):
        self._model_detail_source = (text, values)
        self._set_note("模型", text, **values)
        translated = self.app.tr_error(text) if isinstance(text, BaseException) else self.app.tr(str(text), **values)
        self.model_detail.configure(text=translated)

    def _footer(self, tab, name):
        footer = self._widget(ctk.CTkFrame(tab), "surface")
        footer.grid(row=1, column=0, sticky="ew", padx=12, pady=(6, 12))
        footer.grid_columnconfigure(0, weight=1)
        note = self._label(footer, "", role="muted", font=self.app.font("small"))
        note.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        self._notes[name] = note
        actions = {"模型": self._save_model, "目录": self._save_directories, "音频": self._save_audio, "外观": self._save_appearance}
        button = self._widget(ctk.CTkButton(footer, text="保存", width=118, command=actions[name]), "button")
        button.grid(row=1, column=0, sticky="e")
        self._save_buttons[name] = button

    def _field(self, parent, row, title, variable, browse=None, button_text="浏览"):
        frame = self._widget(ctk.CTkFrame(parent), "surface")
        frame.grid(row=row, column=0, sticky="ew", padx=7, pady=(7, 10))
        frame.grid_columnconfigure(0, weight=1)
        self._label(frame, title).grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 5))
        entry = self._widget(ctk.CTkEntry(frame, textvariable=variable, width=1), "entry")
        entry.grid(row=1, column=0, sticky="ew")
        button = None
        if browse:
            button = self._widget(ctk.CTkButton(frame, text=button_text, width=76, command=browse), "secondary")
            button.grid(row=1, column=1, padx=(8, 0))
        return entry, button

    def _option(self, parent, row, title, variable, values, command=None, localize=True):
        frame = self._widget(ctk.CTkFrame(parent), "surface")
        frame.grid(row=row, column=0, sticky="ew", padx=7, pady=(7, 10))
        frame.grid_columnconfigure(0, weight=1)
        self._label(frame, title).grid(row=0, column=0, sticky="ew", pady=(0, 5))
        choices = dict(values) if isinstance(values, dict) else {value: value for value in values}
        current = choices.get(variable.get(), variable.get())
        labels = {self.app.tr(label) if localize else label: value for label, value in choices.items()}
        variable.set(next((label for label, value in labels.items() if value == current), variable.get()))
        menu = self._widget(ctk.CTkOptionMenu(frame, variable=variable, values=list(labels), width=1, dynamic_resizing=False, command=command), "option")
        menu.grid(row=1, column=0, sticky="ew")
        self._choice_maps[id(variable)] = labels
        if localize:
            self._choice_specs.append((menu, variable, choices))
        return menu

    def _choice_value(self, variable, choices):
        return self._choice_maps.get(id(variable), choices).get(variable.get(), variable.get())

    def _description(self, body, row, text, **values):
        label = self._label(body, text, role="muted", text_values=values, font=self.app.font("small"))
        label.grid(row=row, column=0, sticky="ew", padx=7, pady=(5, 10))
        return label

    def _build_model(self):
        body = self._sections["模型"]
        config = self.config
        self.root_var = tk.StringVar(value=config.get("model_root", "~/asr_models"))
        self.type_var = tk.StringVar(value=next((label for label, value in MODEL_LABELS.items() if value == self._active_type), "Qwen3-ASR"))
        self.path_var = tk.StringVar(value=self._model_paths.get(self._active_type, ""))
        self.device_var = tk.StringVar(value=config.get("device", "auto"))
        self.language_var = tk.StringVar(value=next((label for label, value in LANGUAGES.items() if value == config.get("language")), config.get("language") or "自动识别"))
        self._option(body, 0, "当前使用的模型", self.type_var, list(MODEL_LABELS), self._change_type)
        self._field(body, 1, "此模型的本地路径", self.path_var)
        choices = self._widget(ctk.CTkFrame(body), "surface")
        choices.grid(row=2, column=0, sticky="ew", padx=7, pady=(0, 7))
        choices.grid_columnconfigure(0, weight=1)
        self._widget(ctk.CTkButton(choices, text="选择模型文件夹", command=self._browse_model_folder), "secondary").grid(row=0, column=0, sticky="ew", pady=(0, 6))
        self._widget(ctk.CTkButton(choices, text="选择模型文件", command=self._browse_model_file), "secondary").grid(row=1, column=0, sticky="ew")
        self._option(body, 3, "运行设备", self.device_var, {"自动选择": "auto", "CPU": "cpu", "NVIDIA CUDA": "cuda"})
        self._option(body, 4, "识别语言", self.language_var, LANGUAGES)
        self.model_detail = self._description(body, 5, "可选择模型文件夹，或其中的 config.json、权重或分片索引文件。父目录只含一个可用的同类型模型时会自动定位。")
        self._model_detail_source = ("可选择模型文件夹，或其中的 config.json、权重或分片索引文件。父目录只含一个可用的同类型模型时会自动定位。", {})
        self.model_info = ModelInfoPanel(body, self)
        self.model_info.grid(row=6, column=0, sticky="ew", padx=7, pady=(5, 10))
        self._field(body, 7, "扫描根目录", self.root_var, lambda: self._browse(self.root_var))
        self.scan_button = self._widget(ctk.CTkButton(body, text="扫描本地模型", command=self._scan), "secondary")
        self.scan_button.grid(row=8, column=0, sticky="ew", padx=7, pady=(0, 8))
        self.model_results = self._widget(ctk.CTkFrame(body), "surface")
        self.model_results.grid(row=9, column=0, sticky="ew", padx=7, pady=(0, 8))
        self.model_results.grid_columnconfigure(0, weight=1)
        self._label(self.model_results, "扫描后将列出完整路径。请点击需要使用的模型，扫描不会改变当前选择。", role="muted").grid(row=0, column=0, sticky="ew", padx=10, pady=10)
        self._description(body, 10, "每种模型分别记住自己的路径。切换类型后可选择或修改对应路径；保存后下次转写生效。")
        self._description(body, 11, "Whisper 支持完整 .pt 文件或 Hugging Face 目录；Qwen3 / Confucius 需包含 config.json、分词器和完整权重。压缩包请先解压，程序不会下载模型。")
        self._description(body, 12, "支持 Whisper tiny、base、small、medium、large、turbo，Sherpa-ONNX Whisper，Qwen3-ASR 0.6B / 1.7B 和 Confucius4-R2T2。")
        hardware_link = self._widget(ctk.CTkButton(body, text="查看本机建议", command=lambda: self.tabs.set(self._tab_names["本机配置"])), "secondary")
        self.app.decorate(hardware_link, "hardware")
        hardware_link.grid(row=13, column=0, sticky="ew", padx=7, pady=10)
        self._description(body, 14, "Sherpa-ONNX Whisper 选择包含 encoder、decoder 和 tokens 的目录，或具体 .onnx 文件。目录默认优先 INT8 以节省内存；选择文件则沿用对应精度。当前后端仅使用 CPU，auto 也使用 CPU。")

    def _schedule_model_info(self, *_args):
        if self._model_info_timer is not None:
            self.after_cancel(self._model_info_timer)
        self._model_info_timer = self.after(350, self._refresh_model_info)

    def _refresh_model_info(self):
        self._model_info_timer = None
        self.model_info.refresh(MODEL_LABELS[self.type_var.get()], self.path_var.get().strip())

    def _build_directories(self):
        body = self._sections["目录"]
        selection = self.app.data_selection
        self._data_overridden = bool(selection.overridden)
        self.work_var = tk.StringVar(value=self.config.get("work_dir") or str(Path.home()))
        self._field(body, 0, "默认工作目录", self.work_var, lambda: self._browse(self.work_var))
        self._description(body, 1, "仅作为导入音频和导出文本时的默认文件夹。模型、配置和历史记录分别使用各自的目录。")
        source = {"cli": "启动参数 --data-dir", "env": "环境变量 MYSOUND_DATA_DIR", "bootstrap": "已保存的目录选择", "default": "系统默认目录"}.get(selection.source, selection.source)
        self._data_source = source
        self.data_info = self._description(body, 2, "当前数据目录：\n{path}\n来源：{origin}", path=selection.path, origin=self.app.tr(source))
        pending = self.app.bootstrap.load().get("data_dir") or str(selection.path)
        self.data_var = tk.StringVar(value=pending)
        self.data_entry, self.data_browse = self._field(body, 3, "下次启动使用的数据目录", self.data_var, lambda: self._browse(self.data_var))
        if self._data_overridden:
            self.data_entry.configure(state="disabled")
            self.data_browse.configure(state="disabled")
            self._description(body, 4, "本次启动通过命令行或环境变量指定了数据目录，页面内暂不能更改。移除启动覆盖后，再使用这里保存的选择。工作目录仍可单独保存。")
        else:
            self._description(body, 4, "保存数据目录选择后需重启应用。旧目录中的配置、历史和日志会保留；程序不会迁移或覆盖它们。新目录已有的数据将在重启后读取。")
        self._description(body, 5, "配置文件：{config}\n历史文件：{history}", config=self.app.config_store.path, history=self.app.history_store.path)

    def _build_audio(self):
        body = self._sections["音频"]
        config = self.config
        current_mic = config.get("mic_device")
        initial_label = self.app.tr("系统默认") if current_mic is None else self.app.tr("设备 {index}", index=current_mic)
        self._mic_devices[initial_label] = current_mic
        self.mic_var = tk.StringVar(value=initial_label)
        self.target_var = tk.StringVar(value=str(config.get("vad_target_seconds", 30)))
        self.max_var = tk.StringVar(value=str(config.get("vad_max_seconds", 60)))
        self.block_var = tk.StringVar(value=str(config.get("mic_block_seconds", 2)))
        self.mic_menu = self._option(body, 0, "麦克风输入设备", self.mic_var, self._mic_devices, localize=False)
        self.devices_button = self._widget(ctk.CTkButton(body, text="刷新录音设备", command=self._devices), "secondary")
        self.devices_button.grid(row=1, column=0, sticky="ew", padx=7, pady=(0, 8))
        self._field(body, 2, "听写刷新间隔（秒，1–5）", self.block_var)
        self._field(body, 3, "文件目标切片时长（秒，5–60）", self.target_var)
        self._field(body, 4, "文件最长切片时长（秒，目标时长–60）", self.max_var)
        self._description(body, 5, "文件在自然停顿处切片。实时听写按短句更新，实际速度取决于模型、设备和语句长度；短刷新间隔会增加推理频率。")

    def _build_appearance(self):
        body = self._sections["外观"]
        mode, color = self.config.get("appearance_mode", "dark"), self.config.get("color_theme", "blue")
        self.appearance_var = tk.StringVar(value=next((label for label, value in APPEARANCES.items() if value == mode), "深色"))
        self.color_var = tk.StringVar(value=next((label for label, value in COLORS.items() if value == color), "蓝色"))
        self.font_size_var = tk.StringVar(value=str(self.config.get("font_size", 14)))
        self.ui_language_var = tk.StringVar(value=next((label for label, value in UI_LANGUAGES.items() if value == self.config.get("ui_language", "zh_CN")), "简体中文"))
        self._option(body, 0, "明暗模式", self.appearance_var, APPEARANCES)
        self._option(body, 1, "界面字号", self.font_size_var, [str(size) for size in FONT_SIZES])
        self._option(body, 2, "界面语言", self.ui_language_var, UI_LANGUAGES)
        self._description(body, 3, "保存后立即应用到整个界面，下次启动也会保留。转写进行中仍可调整外观。")
        self._label(body, "主题预设", role="accent").grid(row=4, column=0, sticky="ew", padx=7, pady=(14, 3))
        self.theme_cards = ThemeCards(body, self, lambda: self._choice_value(self.color_var, COLORS), self._select_theme)
        self.theme_cards.grid(row=5, column=0, sticky="ew", padx=7, pady=5)
        self._description(body, 6, "每个预设同时调整背景、面板、边框与强调色。选择后保存外观设置生效。")
        self._widget(ctk.CTkButton(body, text="恢复默认布局", command=self._reset_layout), "secondary").grid(row=7, column=0, sticky="ew", padx=7, pady=(12, 7))
        self._description(body, 8, "恢复默认窗口和分栏尺寸，不改变模型、目录、文本和主题设置。")

    def _select_theme(self, key):
        self.color_var.set(next(label for label, value in COLORS.items() if value == key))

    def _change_type(self, label):
        self._model_paths[self._active_type] = self.path_var.get().strip()
        model_type = MODEL_LABELS[label]
        if model_type == "whisper_onnx" and self._active_type != model_type:
            self.device_var.set("CPU")
        self._active_type = model_type
        self.path_var.set(self._model_paths.get(self._active_type, ""))

    def _browse(self, variable):
        value = Path(variable.get()).expanduser() if variable.get() else Path.home()
        if value.is_file():
            value = value.parent
        path = filedialog.askdirectory(parent=self.winfo_toplevel(), title=self.app.tr("选择文件夹"), mustexist=True, initialdir=str(value if value.is_dir() else Path.home()))
        if path:
            variable.set(path)

    def _model_initial_directory(self):
        current = Path(self.path_var.get() or self.root_var.get() or Path.home()).expanduser()
        if current.is_file():
            current = current.parent
        return str(current if current.is_dir() else Path.home())

    def _browse_model_folder(self):
        path = filedialog.askdirectory(parent=self.winfo_toplevel(), title=self.app.tr("选择模型文件夹"), mustexist=True, initialdir=self._model_initial_directory())
        if path:
            self._accept_model_path(path)

    def _browse_model_file(self):
        model_type = MODEL_LABELS[self.type_var.get()]
        pattern = "*.onnx" if model_type == "whisper_onnx" else "*.json *.safetensors *.bin" + (" *.pt" if model_type == "whisper" else "")
        kind = "ONNX 模型文件" if model_type == "whisper_onnx" else "模型配置或权重"
        path = filedialog.askopenfilename(parent=self.winfo_toplevel(), title=self.app.tr("选择模型文件"), initialdir=self._model_initial_directory(), filetypes=[(self.app.tr(kind), pattern), (self.app.tr("所有文件"), "*")])
        if path:
            self._accept_model_path(path)

    def _browse_checkpoint(self):
        """Compatibility for callers of the previous file-picker action."""
        self._browse_model_file()

    def _accept_model_path(self, raw_path):
        self.path_var.set(str(raw_path))
        try:
            path = validate_model_path(raw_path, MODEL_LABELS[self.type_var.get()])
            self.path_var.set(path)
            self._set_model_note("已定位模型：{path}。保存后生效。", path=path)
            return True
        except (OSError, ValueError) as exc:
            self._set_model_note(exc)
            # Listing candidates also explains incomplete model downloads and
            # ambiguous parent folders without silently changing model type.
            if Path(raw_path).expanduser().is_dir():
                self.root_var.set(str(raw_path))
                self._background("scan", lambda: ModelManager.scan_models(raw_path))
            return False

    def _background(self, operation, action):
        self._request_ids[operation] += 1
        request_id = self._request_ids[operation]
        section = "模型" if operation == "scan" else "音频"
        self._set_note(section, "正在读取…")
        def work():
            try:
                result = {"result": action()}
            except Exception as exc:
                result = {"error": exc}
            self.app.events.put({"kind": "settings_result", "dialog": self, "operation": operation, "request_id": request_id, **result})
        threading.Thread(target=work, daemon=True, name=f"settings-{operation}").start()

    def _scan(self):
        raw_path = self.root_var.get().strip()
        if not raw_path or not Path(raw_path).expanduser().is_dir():
            self._set_note("模型", "请选择存在的模型根目录后扫描。")
            return
        self._background("scan", lambda: ModelManager.scan_models(raw_path))

    def _devices(self):
        def read():
            import sounddevice as sd
            return [(index, item["name"]) for index, item in enumerate(sd.query_devices()) if item["max_input_channels"] > 0]
        self._background("devices", read)

    def accept_result(self, event):
        operation = event["operation"]
        if event.get("request_id", self._request_ids.get(operation)) != self._request_ids.get(operation):
            return
        section = "模型" if operation == "scan" else "音频"
        if "error" in event:
            self._set_note(section, event["error"])
            return
        if operation == "scan":
            self._models = event["result"]
            self._render_models()
            self._set_note(section, "找到 {count} 项。请点击选择，再保存；当前选择未改变。", count=len(self._models))
        elif operation == "devices":
            current = self._mic_devices.get(self.mic_var.get(), self.config.get("mic_device"))
            self._device_records = event["result"]
            self._devices_loaded = True
            self._refresh_mic_choices(current)
            self._set_note(section, "找到 {count} 个录音设备。", count=len(event["result"]))

    def _refresh_mic_choices(self, current):
        self._mic_devices = {self.app.tr("系统默认"): None, **{f"{index}: {name}": index for index, name in self._device_records}}
        if current is not None and current not in self._mic_devices.values():
            source = "设备 {index}（当前未发现）" if self._devices_loaded else "设备 {index}"
            self._mic_devices[self.app.tr(source, index=current)] = current
        self.mic_menu.configure(values=list(self._mic_devices))
        self.mic_var.set(next(label for label, value in self._mic_devices.items() if value == current))

    def _model_status(self, model):
        if self.app.ui_language == "en_US" and model.get("status_en"):
            return model["status_en"]
        return self.app.tr(model.get("status", "本地模型"))

    def _update_model_row(self, label, model):
        self.app.translate_widget(label, "{name} · {type}\n{path}\n{status}",
                                  name=model["name"], type=model.get("type", "unknown"),
                                  path=model["path"], status=self._model_status(model))

    def _render_models(self):
        self._model_row_labels = []
        for widget in self.model_results.winfo_children():
            widget.destroy()
        if not self._models:
            self._label(self.model_results, "未找到模型，可以在上方直接输入或浏览路径。", role="muted").grid(row=0, column=0, sticky="ew", padx=10, pady=10)
        for index, model in enumerate(self._models):
            row = self._widget(ctk.CTkFrame(self.model_results, border_width=1), "surface")
            row.grid(row=index, column=0, sticky="ew", pady=4)
            row.grid_columnconfigure(0, weight=1)
            label = self._label(row, "")
            self._update_model_row(label, model)
            self._model_row_labels.append((label, model))
            label.grid(row=0, column=0, sticky="ew", padx=12, pady=9)
            row.configure(cursor="hand2")
            label.configure(cursor="hand2")
            for item in (row, label):
                item.bind("<Button-1>", lambda _event, chosen=model: self._select_model(chosen), add="+")

    def _select_model(self, model):
        model_type = model.get("type")
        if model_type not in MODEL_LABELS.values():
            self._set_note("模型", "无法判断此模型的类型，请在上方手动选择类型和路径。")
            return
        label = next(label for label, value in MODEL_LABELS.items() if value == model_type)
        self._change_type(label)
        self.type_var.set(label)
        self._accept_model_path(model["path"])

    def set_busy(self, busy):
        self._busy = bool(busy)
        for section, button in self._save_buttons.items():
            button.configure(state="disabled" if busy and section != "外观" else "normal")
        self.app.translate_widget(self.busy_note, "转写中，模型、目录和音频设置暂停保存；外观及本机配置仍可使用。" if busy else "各类别独立保存；可随时返回转写页面。")

    def _can_save(self, section):
        if section != "外观" and (self._busy or self.app._job_ui_active):
            self._set_note(section, "请先返回转写页面停止任务或等待任务完成。")
            return False
        return True

    def _saved(self, section, message):
        self._set_note(section, message)
        self.app.translate_widget(self.app.status, message)

    def _save_model(self):
        if not self._can_save("模型"):
            return False
        try:
            model_type = MODEL_LABELS[self.type_var.get()]
            path = validate_model_path(self.path_var.get(), model_type)
            paths = {**self._model_paths, model_type: path}
            self.config = self.app.config_store.update(
                model_type=model_type, model_paths=paths,
                model_root=str(Path(self.root_var.get().strip() or "~/asr_models").expanduser()),
                device=self._choice_value(self.device_var, {"自动选择": "auto", "CPU": "cpu", "NVIDIA CUDA": "cuda"}),
                language=self._choice_value(self.language_var, LANGUAGES),
            )
            self._model_paths = paths
            self.path_var.set(path)
            self.app._update_model_label()
            self._saved("模型", "模型设置已保存，下次转写时使用。")
            return True
        except Exception as exc:
            self._set_model_note(exc)
            return False

    def _save_directories(self):
        if not self._can_save("目录"):
            return False
        try:
            raw_work = self.work_var.get().strip()
            if not raw_work:
                raise ValueError("请选择默认工作目录。")
            work = Path(raw_work).expanduser().resolve()
            if not work.is_dir():
                raise ValueError("默认工作目录不存在或不是文件夹。")
            data = None
            if not self._data_overridden:
                from storage.bootstrap import validate_data_directory
                # Validate both fields before writing any configuration. Invalid
                # data-directory input must not partially save the work folder.
                data = validate_data_directory(self.data_var.get().strip())
            self.config = self.app.config_store.update(work_dir=str(work))
            if data is not None:
                self.app.bootstrap.select_data_dir(data)
                self.data_var.set(str(data))
            self.work_var.set(str(work))
            message = "工作目录已保存。"
            if data is not None and data != Path(self.app.data_selection.path).resolve():
                message += "数据目录选择已保存，重启后生效；旧数据保留在原目录。"
            self._saved("目录", message)
            return True
        except Exception as exc:
            self._set_note("目录", exc)
            return False

    def _save_audio(self):
        if not self._can_save("音频"):
            return False
        try:
            target, maximum, block = validate_audio_values(self.target_var.get(), self.max_var.get(), self.block_var.get())
            mic = self._mic_devices.get(self.mic_var.get(), self.config.get("mic_device"))
            self.config = self.app.config_store.update(mic_device=mic, vad_target_seconds=target, vad_max_seconds=maximum, mic_block_seconds=block)
            self._saved("音频", "音频设置已保存，下次转写时使用。")
            return True
        except Exception as exc:
            self._set_note("音频", exc)
            return False

    def _save_appearance(self):
        try:
            mode = self._choice_value(self.appearance_var, APPEARANCES)
            color = self._choice_value(self.color_var, COLORS)
            ui_language = self._choice_value(self.ui_language_var, UI_LANGUAGES)
            try:
                font_size = int(self.font_size_var.get())
            except (TypeError, ValueError) as exc:
                raise ValueError("请选择有效的界面字号。") from exc
            if font_size not in FONT_SIZES:
                raise ValueError("请选择有效的界面字号。")
            self.config = self.app.config_store.update(appearance_mode=mode, color_theme=color,
                                                      font_size=font_size, ui_language=ui_language)
            self.app.apply_preferences(mode, color, font_size, ui_language)
            self._saved("外观", "外观设置已保存并应用。")
            return True
        except Exception as exc:
            self._set_note("外观", exc)
            return False

    def _reset_layout(self):
        try:
            self.app.reset_layout()
            self._saved("外观", "已恢复默认布局。")
        except Exception as exc:
            self._set_note("外观", exc)

    def retranslate(self):
        """Update labels and choices in place while keeping every draft value."""
        current_name = self.tabs.get()
        selected = next((key for key, label in self._tab_names.items() if label == current_name), "模型")
        for key, old_name in list(self._tab_names.items()):
            new_name = self.app.tr(key)
            if new_name != old_name:
                self.tabs.rename(old_name, new_name)
                self._tab_names[key] = new_name
        # CTkTabview.rename does not update its current_name bookkeeping.
        self.tabs.set(self._tab_names[selected])
        for menu, variable, choices in self._choice_specs:
            current = self._choice_value(variable, choices)
            labels = {self.app.tr(label): value for label, value in choices.items()}
            menu.configure(values=list(labels))
            variable.set(next((label for label, value in labels.items() if value == current), variable.get()))
            self._choice_maps[id(variable)] = labels
        current_mic = self._mic_devices.get(self.mic_var.get(), self.config.get("mic_device"))
        self._refresh_mic_choices(current_mic)
        for section, (source, values) in list(self._note_sources.items()):
            self._set_note(section, source, **values)
        source, values = self._model_detail_source
        translated = self.app.tr_error(source) if isinstance(source, BaseException) else self.app.tr(source, **values)
        self.model_detail.configure(text=translated)
        self.app.translate_widget(self.data_info, "当前数据目录：\n{path}\n来源：{origin}",
                                  path=self.app.data_selection.path, origin=self.app.tr(self._data_source))
        for label, model in self._model_row_labels:
            if label.winfo_exists():
                self._update_model_row(label, model)
        for button in self._buttons:
            if button.winfo_exists():
                self._fit_button(button)
        for widget in self._inputs:
            if widget.winfo_exists():
                widget.configure(height=max(28, widget.cget("font").metrics("linespace") + 12))
        if isinstance(self.tabs, SectionTabs):
            self.tabs.reflow()
        for name in ("theme_cards", "system_panel", "model_info"):
            component = self.__dict__.get(name)
            if component is not None:
                component.retranslate()
        self.set_busy(self._busy)

    def save_current(self):
        section = next(key for key, label in self._tab_names.items() if label == self.tabs.get())
        if section == "本机配置":
            return True
        return {"模型": self._save_model, "目录": self._save_directories, "音频": self._save_audio, "外观": self._save_appearance}[section]()

    def destroy(self):
        if self._model_info_timer is not None:
            self.after_cancel(self._model_info_timer)
        super().destroy()
