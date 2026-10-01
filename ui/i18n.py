"""Chinese source-string translations for application chrome and messages.

Only call this module for interface strings. Model names, user paths, and
transcribed text remain data and must not be translated. Runtime messages from
the ASR worker can match the templates below without changing that worker's API.
"""

from __future__ import annotations

import re
from string import Formatter
import tkinter as tk
import weakref


LANGUAGES = {"zh_CN": "简体中文", "en_US": "English"}

EN_US = {
    "MySound · 本地语音转文字": "MySound · Local speech to text",
    "本地识别 · 离线保存": "Local recognition · Offline storage",
    "把声音，变成文字": "Turn speech into text",
    "搜索历史记录…": "Search history…",
    "配置": "Settings",
    "配置 · 当前页面": "Settings · Current page",
    "设置": "Settings",
    "返回转写": "Back to transcription",
    "模型": "Models",
    "目录": "Folders",
    "音频": "Audio",
    "外观": "Appearance",
    "保存": "Save",
    "浏览": "Browse",
    "转写文本": "Transcript",
    "将音频或视频拖到这里": "Drop an audio or video file here",
    "选择音频或视频文件": "Choose an audio or video file",
    "\nMP3 / WAV / M4A / FLAC / MP4 等 · 自动切片": "\nMP3 / WAV / M4A / FLAC / MP4 and more · Automatic segmentation",
    "将音频或视频拖到这里\nMP3 / WAV / M4A / FLAC / MP4 等 · 自动切片": "Drop an audio or video file here\nMP3 / WAV / M4A / FLAC / MP4 and more · Automatic segmentation",
    "选择音频或视频文件\nMP3 / WAV / M4A / FLAC / MP4 等 · 自动切片": "Choose an audio or video file\nMP3 / WAV / M4A / FLAC / MP4 and more · Automatic segmentation",
    "选择文件": "Choose file",
    "开始听写": "Start dictation",
    "停止": "Stop",
    "保存修改": "Save changes",
    "导出": "Export",
    "删除": "Delete",
    "实时听写将在这里显示…": "Live dictation will appear here…",
    "就绪 · 先配置本地模型，再导入文件或开始听写": "Ready · Configure a local model, then open a file or start dictation",
    "当前模型：{model} · {path} · {device}": "Current model: {model} · {path} · {device}",
    "尚未配置": "Not configured",
    "暂无记录": "No history yet",
    "是否先保存当前文本的修改？": "Save changes to the current transcript first?",
    "修改已保存": "Changes saved",
    "手动文本": "Manual text",
    "无": "None",
    "文本": "Text",
    "麦克风": "Microphone",
    "准备中…": "Preparing…",
    "请先在设置中选择模型路径并保存。": "Choose a local model path in Settings and save it first.",
    "选择音频或视频": "Choose audio or video",
    "音频与视频": "Audio and video",
    "所有文件": "All files",
    "一次处理一个文件，已选择第一个文件。": "One file can be processed at a time. The first file was selected.",
    "请拖入一个音频或视频文件。": "Drop an audio or video file.",
    "正在聆听…": "Listening…",
    "已停止": "Stopped",
    "转写完成": "Transcription complete",
    " · 已保存到历史记录": " · Saved to history",
    " · 未检测到语音": " · No speech detected",
    "已停止 · 已保存到历史记录": "Stopped · Saved to history",
    "已停止 · 未检测到语音": "Stopped · No speech detected",
    "转写完成 · 已保存到历史记录": "Transcription complete · Saved to history",
    "转写完成 · 未检测到语音": "Transcription complete · No speech detected",
    "导出文本或字幕": "Export transcript or subtitles",
    "纯文本": "Plain text",
    "SRT 字幕": "SRT subtitles",
    "已导出：{path}": "Exported: {path}",
    "删除记录": "Delete recording",
    "确定删除这条历史记录及其文本？": "Delete this history record and its transcript?",
    "停止并退出": "Stop and quit",
    "正在转写，是否停止并保存已有结果后退出？": "Transcription is running. Stop, save existing results, and quit?",
    "无法保存窗口布局": "Could not save window layout",
    "自动识别": "Automatic detection",
    "中文": "Chinese",
    "英语": "English",
    "日语": "Japanese",
    "韩语": "Korean",
    "深色": "Dark",
    "浅色": "Light",
    "蓝色": "Blue",
    "绿色": "Green",
    "紫色": "Purple",
    "橙色": "Orange",
    "系统默认": "System default",
    "当前使用的模型": "Selected model",
    "此模型的本地路径": "Local path for this model",
    "选择 Whisper .pt 权重": "Choose Whisper .pt weights",
    "选择 Whisper 权重": "Choose Whisper weights",
    "Whisper 权重": "Whisper weights",
    "每种模型分别记住自己的路径。切换类型后可选择或修改对应路径；保存后下次转写生效。": "Each model type remembers its own path. Switch types to select or edit its path; saved changes apply to the next transcription.",
    "运行设备": "Compute device",
    "识别语言": "Recognition language",
    "扫描根目录": "Model search folder",
    "扫描本地模型": "Scan local models",
    "扫描后将列出完整路径。请点击需要使用的模型，扫描不会改变当前选择。": "Scanning lists full paths. Click a model to select it; scanning does not change your selection.",
    "Whisper 支持完整 .pt 文件或 Hugging Face 目录；Qwen3 / Confucius 需包含 config.json、分词器和完整权重。压缩包请先解压，程序不会下载模型。": "Whisper accepts a complete .pt file or Hugging Face folder. Qwen3 / Confucius require config.json, tokenizer files, and complete weights. Extract archives first; the app does not download models.",
    "默认工作目录": "Default working folder",
    "仅作为导入音频和导出文本时的默认文件夹。模型、配置和历史记录分别使用各自的目录。": "The initial folder for opening audio and exporting text. Models, configuration, and history use their own folders.",
    "当前数据目录：\n{path}\n来源：{source}": "Current data folder:\n{path}\nSource: {source}",
    "下次启动使用的数据目录": "Data folder for the next launch",
    "本次启动通过命令行或环境变量指定了数据目录，页面内暂不能更改。移除启动覆盖后，再使用这里保存的选择。工作目录仍可单独保存。": "A command-line argument or environment variable overrides the data folder for this launch. Remove that override to use the saved selection. You can still save the working folder.",
    "保存数据目录选择后需重启应用。旧目录中的配置、历史和日志会保留；程序不会迁移或覆盖它们。新目录已有的数据将在重启后读取。": "Restart to use the selected data folder. Configuration, history, and logs remain in the old folder; nothing is migrated or overwritten. Existing data in the new folder loads after restart.",
    "配置文件：{config}\n历史文件：{history}": "Configuration: {config}\nHistory: {history}",
    "启动参数 --data-dir": "Command-line argument --data-dir",
    "环境变量 MYSOUND_DATA_DIR": "Environment variable MYSOUND_DATA_DIR",
    "已保存的目录选择": "Saved folder selection",
    "系统默认目录": "System default folder",
    "麦克风输入设备": "Microphone input device",
    "刷新录音设备": "Refresh input devices",
    "听写刷新间隔（秒，1–5）": "Dictation refresh interval (seconds, 1–5)",
    "文件目标切片时长（秒，5–60）": "Target file segment length (seconds, 5–60)",
    "文件最长切片时长（秒，目标时长–60）": "Maximum segment length (seconds, target–60)",
    "文件在自然停顿处切片。实时听写按短句更新，实际速度取决于模型、设备和语句长度；短刷新间隔会增加推理频率。": "Files are segmented at natural pauses. Dictation updates by phrase; speed depends on the model, device, and phrase length. Shorter intervals require more inference.",
    "明暗模式": "Appearance mode",
    "主题颜色": "Accent color",
    "界面语言": "Interface language",
    "字体大小": "Font size",
    "界面字体大小": "Interface font size",
    "简体中文": "简体中文",
    "保存后立即应用到整个界面，下次启动也会保留。转写进行中仍可调整外观。": "Changes apply throughout the interface when saved and are restored next launch. Appearance can be changed during transcription.",
    "界面语言只影响按钮和提示，不改变模型识别语言或转写内容。": "Interface language changes labels and messages, not the recognition language or transcript.",
    "字体大小应用到整个界面，并在下次启动时保留。": "Font size applies throughout the interface and is restored on the next launch.",
    "恢复默认布局": "Reset layout",
    "恢复默认窗口和分栏尺寸，不改变模型、目录、文本和主题设置。": "Reset the window and panel sizes. Model, folder, transcript, and theme settings are preserved.",
    "正在读取…": "Loading…",
    "请选择存在的模型根目录后扫描。": "Choose an existing model folder before scanning.",
    "找到 {count} 项。请点击选择，再保存；当前选择未改变。": "Found {count} items. Click a model and save; your current selection is unchanged.",
    "设备 {device}": "Device {device}",
    "设备 {device}（当前未发现）": "Device {device} (not currently detected)",
    "找到 {count} 个录音设备。": "Found {count} audio input devices.",
    "未找到模型，可以在上方直接输入或浏览路径。": "No models found. Enter or browse to a path above.",
    "本地模型": "Local model",
    "本地目录": "Local folder",
    "本地权重": "Local weights",
    "请先解压模型": "Extract the model first",
    "模型下载未完成": "Model download is incomplete",
    "（需解压）": " (extract first)",
    "（下载未完成）": " (download incomplete)",
    "无法判断此模型的类型，请在上方手动选择类型和路径。": "Could not determine the model type. Select its type and path above.",
    "已选择 {name}。{status}；保存后生效。": "Selected {name}. {status}; save to apply.",
    "正在转写：模型、目录和音频设置暂不能保存。可返回转写页面停止任务，外观设置仍可使用。": "Transcription is running. Model, folder, and audio settings cannot be saved yet. Return to transcription to stop the task. Appearance settings remain available.",
    "各类别独立保存；可随时返回转写页面。": "Save each section separately. You can return to transcription at any time.",
    "请先返回转写页面停止任务或等待任务完成。": "Return to transcription and stop the task, or wait for it to finish.",
    "模型设置已保存，下次转写时使用。": "Model settings saved for the next transcription.",
    "工作目录已保存。": "Working folder saved.",
    "数据目录选择已保存，重启后生效；旧数据保留在原目录。": "Data folder selection saved. Restart to apply; old data remains in its original folder.",
    "音频设置已保存，下次转写时使用。": "Audio settings saved for the next transcription.",
    "外观设置已保存并应用。": "Appearance settings saved and applied.",
    "已恢复默认布局。": "Default layout restored.",
    "保存模型设置": "Save model settings",
    "保存目录设置": "Save folder settings",
    "保存音频设置": "Save audio settings",
    "保存外观设置": "Save appearance",
    "保存{section}设置": "Save {section} settings",
    "请选择本地模型目录或 Whisper .pt 文件。": "Choose a local model folder or Whisper .pt file.",
    "模型路径不存在，请检查路径。": "The model path does not exist. Check the path.",
    "请先解压模型压缩包或完成模型下载。": "Extract the model archive or complete the download first.",
    "Whisper 需要完整 .pt 文件，或包含 .pt / config.json 的模型目录。": "Whisper needs a complete .pt file or a model folder containing .pt or config.json.",
    "Qwen3 / Confucius 需要包含 config.json、分词器和权重的模型目录。": "Qwen3 / Confucius need a model folder containing config.json, tokenizer files, and weights.",
    "音频时间请填写有效数字。": "Enter valid numbers for audio durations.",
    "音频时间请填写有限的有效数字。": "Enter finite, valid numbers for audio durations.",
    "切片时间需满足：5 ≤ 目标 ≤ 最长 ≤ 60 秒。": "Segment lengths must satisfy: 5 ≤ target ≤ maximum ≤ 60 seconds.",
    "听写刷新间隔请设为 1–5 秒。": "Set the dictation refresh interval to 1–5 seconds.",
    "请选择默认工作目录。": "Choose a default working folder.",
    "默认工作目录不存在或不是文件夹。": "The working folder does not exist or is not a directory.",
    "请选择数据目录。": "Choose a data folder.",
    "数据目录必须是文件夹。": "The data path must be a directory.",
    "数据目录不可读写：{path}": "The data folder is not readable and writable: {path}",
    "请先停止或等待当前任务完成。": "Stop the current task or wait for it to finish first.",
    "请先在模型设置中选择本地模型路径。": "Choose a local model path in model settings first.",
    "正在加载本地模型，首次加载可能需要一些时间…": "Loading the local model. The first load may take some time…",
    "正在转写第 {index} 段 · {start}–{end} 秒": "Transcribing segment {index} · {start}–{end} seconds",
    "正在转写第 {index} 段 · {start:.1f}–{end:.1f} 秒": "Transcribing segment {index} · {start:.1f}–{end:.1f} seconds",
    "正在停止并保存已识别内容，请等待当前推理完成…": "Stopping and saving recognized text. Waiting for the current inference to finish…",
    "未找到 FFmpeg，请运行 sudo apt install ffmpeg。": "FFmpeg was not found. Run sudo apt install ffmpeg.",
    "转写已取消": "Transcription cancelled",
    "正在使用 FFmpeg 转换为 16 kHz 单声道音频…": "Converting to 16 kHz mono audio with FFmpeg…",
    "正在进行 Silero-VAD 语音检测与分段转写…": "Detecting speech with Silero-VAD and transcribing segments…",
    "音频文件不存在：{path}": "Audio file does not exist: {path}",
    "缺少 Silero-VAD，请先安装 requirements.txt 中的依赖。": "Silero-VAD is missing. Install the dependencies in requirements.txt.",
    "FFmpeg 无法读取媒体文件：{detail}": "FFmpeg could not read the media file: {detail}",
    "麦克风正在运行": "The microphone is already running",
    "正在听写：停顿后确认文字，处理中间结果约每两秒更新。": "Listening: phrases are confirmed after a pause; interim results update about every two seconds.",
    "缺少麦克风依赖，请安装 numpy、sounddevice 和 PortAudio。": "Microphone dependencies are missing. Install numpy, sounddevice, and PortAudio.",
    "麦克风采集发生丢帧：{status}；录音已停止。": "Microphone frames were lost: {status}; recording has stopped.",
    "模型处理速度低于录音速度，音频队列已满；录音已停止，请使用较小模型。": "The model is slower than the recording and the audio queue is full. Recording stopped; use a smaller model.",
    "已选择 CUDA，但 PyTorch 未检测到可用的 NVIDIA GPU。请选择 CPU 或安装对应的 CUDA 版 PyTorch。": "CUDA was selected, but PyTorch detected no available NVIDIA GPU. Select CPU or install a compatible CUDA build of PyTorch.",
    "Sherpa-ONNX Whisper 当前适配使用 CPU。请选择 CPU 或自动模式；此安装未启用经过验证的 CUDA 后端。": "Sherpa-ONNX Whisper currently uses CPU. Select CPU or auto mode; this installation has no verified CUDA backend enabled.",
    "CUDA 显存不足，自动改用 CPU 加载模型；转写速度会变慢。": "CUDA memory is insufficient. Loading the model on CPU instead; transcription will be slower.",
    "转写时 CUDA 显存不足，自动改用 CPU 重试当前音频；后续音频继续使用 CPU。": "CUDA memory ran out during transcription. Retrying this audio on CPU; subsequent audio will also use CPU.",
    "CUDA 显存不足。请选择自动模式（显存不足时改用 CPU）、直接使用 CPU，或选择更小的模型。": "CUDA memory is insufficient. Select auto mode to fall back to CPU, choose CPU directly, or use a smaller model.",
    "显存不足。请在「配置 → 模型」将设备改为 auto 或 cpu，然后重试。": "GPU memory is insufficient. In Settings → Models, set the device to auto or cpu, then retry.",
    "详细错误已记录到：{path}": "Full error details were logged to: {path}",
    "当前错误信息过长，请查看日志中的完整内容。": "This error message is too long to display. See the log for complete details.",
    "模型输入必须为 16000 Hz；请先通过 AudioProcessor 转换音频。": "Model input must be 16000 Hz. Convert the audio with AudioProcessor first.",
    "模型输入必须为一维单声道 PCM 数组。": "Model input must be a one-dimensional mono PCM array.",
    "请先将模型压缩包解压，再选择包含 config.json 的模型目录。": "Extract the model archive first, then choose the folder containing config.json.",
    "模型文件尚未下载完整，请完成下载后再加载。": "The model download is incomplete. Finish downloading before loading it.",
    "本地模型不存在：{path}": "Local model does not exist: {path}",
    "此目录只有模型压缩包。请先解压，再选择包含 config.json 的模型目录。": "This folder only contains a model archive. Extract it and select the folder containing config.json.",
    "请选择包含 config.json、分词器和完整权重的模型目录：{path}": "Choose a model folder containing config.json, tokenizer files, and complete weights: {path}",
    "请先在设置中选择并加载本地模型。": "Select and load a local model in Settings first.",
    "无法导入 {module}：{error}。请在运行程序的 Python 环境执行：{install}": "Could not import {module}: {error}. Run this in the application's Python environment: {install}",
    "请在设置中选择本地模型目录或 Whisper .pt 文件。": "Choose a local model folder or Whisper .pt file in Settings.",
    "Whisper 目录中需要一个 .pt 权重文件，或完整的 Hugging Face 模型文件。": "The Whisper folder must contain one .pt weights file or complete Hugging Face model files.",
    "目录中有多个 .pt 权重，请直接选择要加载的 .pt 文件。": "This folder contains multiple .pt files. Select the specific .pt file to load.",
    "Whisper 原生模型文件应为 .pt；其他格式请选择完整模型目录。": "Native Whisper weights must use .pt. For other formats, select the complete model folder.",
    "不支持的模型类型：{model}": "Unsupported model type: {model}",
    "Whisper 权重尚未下载完整（.incomplete）；请完成下载后再加载。": "Whisper weights are incomplete (.incomplete). Finish downloading before loading.",
    "该记录没有时间戳，请导出为 TXT 或 Markdown。": "This record has no timestamps. Export it as TXT or Markdown.",
    "字幕片段时间无效，无法导出 SRT。": "Subtitle segment timings are invalid; SRT cannot be exported.",
    "仅支持 .txt、.md 和 .srt 格式。": "Only .txt, .md, and .srt formats are supported.",
    "转写记录": "Transcription",
}


def _templates():
    patterns = []
    for source, translated in EN_US.items():
        if "{" not in source:
            continue
        pieces = []
        for literal, field, _spec, _conversion in Formatter().parse(source):
            pieces.append(re.escape(literal))
            if field is not None:
                pieces.append(f"(?P<{field}>.*?)")
        # Captured worker values are already formatted text. Keep that text
        # rather than applying numeric format specs a second time to strings.
        rendered_template = "".join(
            literal + ("{" + field + "}" if field is not None else "")
            for literal, field, _spec, _conversion in Formatter().parse(translated)
        )
        patterns.append((re.compile("".join(pieces), re.DOTALL), rendered_template))
    # Match more specific messages before a general template such as Device {id}.
    return sorted(patterns, key=lambda item: len(item[0].pattern), reverse=True)


_TEMPLATES = _templates()


def register_messages(messages: dict[str, str]) -> None:
    """Register a feature's catalogue without importing that feature here.

    Feature modules call this once when loaded. Rebuild the runtime-message
    templates so their formatted status strings translate as well as literals.
    """
    global _TEMPLATES
    EN_US.update(messages)
    _TEMPLATES = _templates()


class Translator:
    def __init__(self, language="zh_CN"):
        self.language = language if language in LANGUAGES else "zh_CN"
        self._bindings = weakref.WeakKeyDictionary()

    def tr(self, source: str, **kwargs) -> str:
        source = str(source)
        translated = source
        if self.language == "en_US":
            translated = EN_US.get(source, source)
            if translated == source and not kwargs:
                for pattern, english in _TEMPLATES:
                    match = pattern.fullmatch(source)
                    if match:
                        return english.format(**match.groupdict())
        return translated.format(**kwargs) if kwargs else translated

    def bind(self, widget, source: str, option="text", **kwargs):
        """Register a UI string and update it immediately; return the widget."""
        options = self._bindings.setdefault(widget, {})
        options[option] = (source, dict(kwargs))
        widget.configure(**{option: self.tr(source, **kwargs)})
        return widget

    def unbind(self, widget, option="text"):
        """Stop translating a widget option before showing user-owned text."""
        options = self._bindings.get(widget)
        if options is not None:
            options.pop(option, None)
            if not options:
                self._bindings.pop(widget, None)
        return widget

    def set_language(self, language):
        self.language = language if language in LANGUAGES else "zh_CN"
        for widget, options in list(self._bindings.items()):
            try:
                if not widget.winfo_exists():
                    self._bindings.pop(widget, None)
                    continue
                widget.configure(**{
                    option: self.tr(source, **kwargs)
                    for option, (source, kwargs) in options.items()
                })
            except tk.TclError:
                try:
                    exists = widget.winfo_exists()
                except tk.TclError:
                    exists = False
                if exists:
                    raise
                self._bindings.pop(widget, None)
        return self.language
