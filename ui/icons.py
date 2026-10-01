"""Antialiased line icons drawn locally; no symbol font or asset download."""

from __future__ import annotations

from functools import lru_cache
import math
import tkinter as tk
import weakref

from PIL import Image, ImageDraw

from storage.config_store import normalize_font_size


ICON_NAMES = (
    "settings", "model", "folder", "mic", "palette", "monitor", "cpu", "gpu",
    "memory", "upload", "file", "play", "stop", "save", "export", "trash",
    "back", "search", "history", "refresh", "scan", "language", "type",
    "check", "wave", "info", "chevron_left", "chevron_right",
)
_ALIASES = {"microphone": "mic", "hardware": "monitor"}

# Exact source labels only. Never infer an icon from transcript or path text.
ICON_LABELS = {
    "配置": "settings", "配置 · 当前页面": "settings", "设置": "settings",
    "模型": "model", "目录": "folder", "音频": "mic", "外观": "palette", "硬件": "monitor",
    "硬件信息": "monitor", "返回转写": "back", "选择文件": "upload", "开始听写": "mic",
    "停止": "stop", "保存修改": "save", "导出": "export", "删除": "trash", "浏览": "folder",
    "扫描本地模型": "scan", "刷新录音设备": "refresh", "刷新硬件信息": "refresh",
    "刷新": "refresh", "界面语言": "language", "识别语言": "language",
    "字体大小": "type", "界面字号": "type", "默认工作目录": "folder",
    "麦克风输入设备": "mic", "运行设备": "cpu", "主题颜色": "palette",
    "当前使用的模型": "model", "当前模型": "model", "转写文本": "file",
    "保存模型设置": "save", "保存目录设置": "save", "保存音频设置": "save",
    "保存外观设置": "save", "选择 Whisper .pt 权重": "file", "选择模型文件": "file",
    "恢复默认布局": "refresh", "历史记录": "history", "无历史记录": "history",
}


@lru_cache(maxsize=512)
def _render_icon(name: str, color: str, size: int) -> Image.Image:
    scale = 4
    canvas = Image.new("RGBA", (size * scale, size * scale), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    unit = size * scale / 24
    stroke = max(1, round(1.7 * unit))

    def points(values):
        return [(round(x * unit), round(y * unit)) for x, y in values]

    def line(values, width=stroke):
        locations = points(values)
        draw.line(locations, fill=color, width=width, joint="curve")
        radius = width / 2
        for x, y in (locations[0], locations[-1]):
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)

    def rect(box, radius=0, fill=None):
        bounds = tuple(round(value * unit) for value in box)
        if radius:
            draw.rounded_rectangle(bounds, radius=round(radius * unit), outline=color, width=stroke, fill=fill)
        else:
            draw.rectangle(bounds, outline=color, width=stroke, fill=fill)

    def ellipse(box, fill=None):
        draw.ellipse(tuple(round(value * unit) for value in box), outline=color, width=stroke, fill=fill)

    def arc(box, start, end):
        draw.arc(tuple(round(value * unit) for value in box), start=start, end=end, fill=color, width=stroke)

    if name == "settings":
        ellipse((6.5, 6.5, 17.5, 17.5))
        ellipse((10, 10, 14, 14))
        for angle in range(0, 360, 45):
            theta = math.radians(angle)
            line([(12 + radius * math.cos(theta), 12 + radius * math.sin(theta)) for radius in (6, 9)])
    elif name == "model":
        line([(12, 2.8), (20, 7.4), (20, 16.6), (12, 21.2), (4, 16.6), (4, 7.4), (12, 2.8)])
        line([(4, 7.4), (12, 12), (20, 7.4)])
        line([(12, 12), (12, 21)])
    elif name == "folder":
        line([(3, 7), (3, 5), (9, 5), (11, 7), (20, 7), (20, 19), (3, 19), (3, 7), (20, 7)])
    elif name == "mic":
        rect((9, 3, 15, 14), radius=3)
        arc((6, 6, 18, 18), 0, 180)
        line([(6, 9), (6, 12)])
        line([(18, 9), (18, 12)])
        line([(12, 18), (12, 21)])
        line([(8, 21), (16, 21)])
    elif name == "palette":
        # A round palette with a thumb hole and three paint wells.
        ellipse((3, 3, 21, 21))
        ellipse((14, 13, 17.5, 16.5))
        for x, y in ((7, 8), (11, 6.5), (16, 8)):
            ellipse((x - 0.7, y - 0.7, x + 0.7, y + 0.7), fill=color)
    elif name == "monitor":
        rect((3, 4, 21, 17), radius=1.5)
        line([(12, 17), (12, 21)])
        line([(8, 21), (16, 21)])
    elif name == "cpu":
        rect((6, 6, 18, 18), radius=1)
        rect((9, 9, 15, 15), radius=0.5)
        for point in (8, 12, 16):
            line([(point, 3), (point, 6)])
            line([(point, 18), (point, 21)])
            line([(3, point), (6, point)])
            line([(18, point), (21, point)])
    elif name == "gpu":
        rect((3, 6, 20, 17), radius=1)
        ellipse((6, 8.5, 12, 14.5))
        line([(15, 9), (17, 9)])
        line([(15, 12), (17, 12)])
        line([(6, 17), (6, 20), (16, 20), (16, 17)])
        line([(21, 4), (21, 19)])
    elif name == "memory":
        rect((3, 6, 21, 17), radius=1)
        for x in (6, 10, 14):
            rect((x, 9, x + 2, 13), radius=0.3)
        for x in (6, 9, 12, 15, 18):
            line([(x, 17), (x, 20)])
    elif name == "upload":
        line([(4, 15), (4, 20), (20, 20), (20, 15)])
        line([(12, 4), (12, 15)])
        line([(7, 9), (12, 4), (17, 9)])
    elif name == "file":
        line([(5, 3), (14, 3), (19, 8), (19, 21), (5, 21), (5, 3)])
        line([(14, 3), (14, 8), (19, 8)])
        line([(8, 12), (16, 12)])
        line([(8, 16), (14, 16)])
    elif name == "play":
        line([(7, 4), (20, 12), (7, 20), (7, 4)])
    elif name == "stop":
        rect((5, 5, 19, 19), radius=2)
    elif name == "save":
        line([(4, 3), (17, 3), (21, 7), (21, 21), (3, 21), (3, 3), (4, 3)])
        rect((7, 3, 16, 9))
        rect((7, 14, 17, 21), radius=0.7)
    elif name == "export":
        line([(10, 5), (4, 5), (4, 20), (19, 20), (19, 14)])
        line([(11, 13), (21, 3)])
        line([(14, 3), (21, 3), (21, 10)])
    elif name == "trash":
        line([(4, 6), (20, 6)])
        line([(9, 6), (9, 3), (15, 3), (15, 6)])
        line([(6, 7), (7, 21), (17, 21), (18, 7)])
        line([(10, 10), (10.5, 17)])
        line([(14, 10), (13.5, 17)])
    elif name == "back":
        line([(10, 5), (3, 12), (10, 19)])
        line([(3, 12), (21, 12)])
    elif name == "search":
        ellipse((3, 3, 16, 16))
        line([(14, 14), (21, 21)])
    elif name == "history":
        arc((3, 3, 21, 21), 215, 535)
        line([(3, 3), (3, 9), (9, 9)])
        line([(12, 7), (12, 12), (16, 14)])
    elif name == "refresh":
        arc((4, 4, 20, 20), 205, 355)
        arc((4, 4, 20, 20), 25, 175)
        line([(16, 8), (20, 12), (21, 6)])
        line([(8, 16), (4, 12), (3, 18)])
    elif name == "scan":
        for values in (
            [(8, 3), (3, 3), (3, 8)], [(16, 3), (21, 3), (21, 8)],
            [(3, 16), (3, 21), (8, 21)], [(16, 21), (21, 21), (21, 16)],
        ):
            line(values)
        line([(3, 12), (21, 12)])
    elif name == "language":
        ellipse((3, 3, 21, 21))
        ellipse((8, 3, 16, 21))
        line([(3, 12), (21, 12)])
    elif name == "type":
        line([(3, 5), (17, 5)])
        line([(3, 5), (3, 8)])
        line([(17, 5), (17, 8)])
        line([(10, 5), (10, 20)])
        line([(7, 20), (13, 20)])
        line([(15, 13), (21, 13)])
        line([(18, 13), (18, 20)])
    elif name == "check":
        line([(4, 12), (9, 17), (20, 6)])
    elif name == "wave":
        for x, top, bottom in ((3, 10, 14), (7.5, 6, 18), (12, 3, 21), (16.5, 7, 17), (21, 10, 14)):
            line([(x, top), (x, bottom)])
    elif name == "info":
        ellipse((3, 3, 21, 21))
        ellipse((11.2, 6.4, 12.8, 8), fill=color)
        line([(11, 11), (12, 11), (12, 17)])
        line([(10, 17), (14, 17)])
    elif name == "chevron_left":
        line([(15, 5), (8, 12), (15, 19)])
    elif name == "chevron_right":
        line([(9, 5), (16, 12), (9, 19)])
    return canvas.resize((size, size), Image.Resampling.LANCZOS)


def draw_icon(name: str, color: str = "#FFFFFF", size: int = 24) -> Image.Image:
    """Return an independent transparent Pillow image at the requested size."""
    name = _ALIASES.get(name, name)
    if name not in ICON_NAMES:
        raise ValueError(f"Unknown icon: {name}")
    size = max(12, min(192, int(size)))
    return _render_icon(name, color, size).copy()


class IconManager:
    """Theme-aware icon registry; bind/apply run on the Tk main thread."""

    def __init__(self, palette: dict, base_size=14, *, image_factory=None):
        if image_factory is None:
            from customtkinter import CTkImage
            image_factory = CTkImage
        self.palette = dict(palette)
        self.base_size = normalize_font_size(base_size)
        self._image_factory = image_factory
        self._widgets = weakref.WeakKeyDictionary()
        self._images = {}

    @property
    def size(self):
        return self.base_size + 4

    def _image(self, name, tone):
        name = _ALIASES.get(name, name)
        color = tone if str(tone).startswith("#") else self.palette.get(tone, self.palette["text"])
        key = (name, color, self.size)
        if key not in self._images:
            image = draw_icon(name, color, self.size * 2)
            # Both variants use the current palette. apply() regenerates them
            # when the app changes mode; 2x pixels stay sharp under display DPI.
            self._images[key] = self._image_factory(
                light_image=image, dark_image=image, size=(self.size, self.size),
            )
        return self._images[key]

    def bind(self, widget, name, tone="text", compound="left"):
        icon = self._image(name, tone)
        widget.configure(image=icon, compound=compound)
        self._widgets[widget] = (name, tone, compound)
        return widget

    def unbind(self, widget):
        self._widgets.pop(widget, None)
        widget.configure(image=None)
        return widget

    def apply(self, palette: dict, base_size):
        self.palette = dict(palette)
        self.base_size = normalize_font_size(base_size)
        self._images.clear()
        for widget, (name, tone, compound) in list(self._widgets.items()):
            try:
                if not widget.winfo_exists():
                    self._widgets.pop(widget, None)
                    continue
                widget.configure(image=self._image(name, tone), compound=compound)
            except tk.TclError:
                try:
                    exists = widget.winfo_exists()
                except tk.TclError:
                    exists = False
                if exists:
                    raise
                self._widgets.pop(widget, None)
