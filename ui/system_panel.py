"""Read-only hardware and model guidance panels; detection runs off the UI thread."""

from __future__ import annotations

import math
import threading
import webbrowser

import customtkinter as ctk

from ui.i18n import register_messages


register_messages({
    "本机配置": "This PC", "刷新检测": "Refresh hardware", "正在检测本机硬件…": "Detecting local hardware…",
    "硬件信息已更新。": "Hardware information updated.", "处理器": "Processor",
    "逻辑线程": "Logical threads", "系统内存": "System memory", "当前可用内存": "Currently available memory",
    "操作系统": "Operating system", "显卡": "Graphics", "显存总量": "Total video memory",
    "当前可用显存": "Currently available video memory", "未检测到可用的 CUDA 显卡": "No available CUDA GPU detected",
    "未知": "Unknown", "尚未检测": "Not checked yet", "未识别规格": "Unrecognized variant",
    "本机模型建议": "Models for this computer", "所选模型信息": "Selected model information",
    "参数规模": "Parameters", "本地文件体积": "Local files on disk", "CPU 推理内存预算": "RAM budget for CPU inference",
    "GPU 推理显存预算": "VRAM budget for GPU inference", "GPU 推理主机内存预算": "Host RAM budget for GPU inference",
    "速度提示": "Speed guidance", "本机适配度": "Fit for this computer", "建议运行设备": "Suggested device",
    "建议使用": "Recommended", "资源较紧": "Limited headroom", "资源不足": "Insufficient resources",
    "需要确认": "Needs checking", "CPU": "CPU", "CUDA 显卡": "CUDA GPU",
    "没有识别到具体型号，选择完整模型路径后可显示对应预算。": "The exact variant is not identified. Choose a complete local model path to see its resource budget.",
    "尚未选择模型。选择本地模型后显示型号与资源预算。": "No model selected. Choose a local model to see its variant and resource budget.",
    "以下是运行预算的估算范围，不是精确最低要求或速度保证。音频长度、精度、后台应用及推理版本都会影响占用。":
        "These are estimated resource budgets, not exact minimum requirements or speed guarantees. Audio length, precision, background applications, and inference versions affect usage.",
    "建议基于当前可用内存和显存。选择更小的模型通常响应更快；较大的模型需要更多资源。":
        "Suggestions use currently available RAM and VRAM. Smaller models generally respond faster; larger models need more resources.",
    "检测不会加载模型，可在转写期间刷新；可用资源会随其他程序变化。":
        "Detection does not load models and can run during transcription. Available resources change as other applications run.",
    "正在读取模型信息…": "Reading model information…", "刷新本机配置后可显示适配建议。": "Refresh This PC to see fit guidance.",
    "检测未能完整读取所有设备信息，未知项目请结合系统设置确认。": "Some hardware information could not be read. Check unknown values in your system settings.",
    "未发现独立显卡，可使用 CPU 识别。": "No discrete GPU was found. CPU transcription remains available.",
    "{low}–{high} GiB": "{low}–{high} GiB", "{value} GiB": "{value} GiB",
    "本机首选": "First choice", "系列建议": "Family pick", "全部型号": "All variants",
    "资源有余量": "Estimated headroom", "官方参考": "Official reference",
    "官方约需显存（GB）": "Official approximate VRAM (GB)",
    "官方相对速度（A100，large = 1）": "Official relative speed (A100, large = 1)",
    "官方模型文档 ↗": "Official model documentation ↗",
    "当前后端不支持此显卡加速，将按 CPU 方案评估。": "The current backend cannot accelerate on this GPU; suggestions use CPU execution.",
    "本机尚未测速，资源适配不代表可实时听写。较小模型通常响应更快。": "This machine has not been benchmarked. Resource fit does not guarantee real-time dictation; smaller models generally respond faster.",
    "CPU 内存 {ram} · GPU 显存 {vram}": "CPU RAM {ram} · GPU VRAM {vram}",
    "CPU 内存 {ram} · 仅 CPU": "CPU RAM {ram} · CPU only",
    "不适用（仅 CPU）": "Not applicable (CPU only)",
    "选择型号查看资源建议，也可使用左右按钮逐项浏览。": "Select a variant to view resource guidance, or use the arrows to browse all variants.",
})


def format_gib(value, tr=lambda text, **kwargs: text.format(**kwargs)):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return tr("未知")
    if not math.isfinite(number) or number < 0:
        return tr("未知")
    return tr("{value} GiB", value=f"{number:.1f}")


def format_budget(value, tr=lambda text, **kwargs: text.format(**kwargs)):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return format_gib(value, tr)
    try:
        low, high = float(value[0]), float(value[1])
    except (TypeError, ValueError):
        return tr("未知")
    if not all(math.isfinite(number) and number >= 0 for number in (low, high)):
        return tr("未知")
    return tr("{low}–{high} GiB", low=f"{low:g}", high=f"{high:g}")


def model_budget(model, field, tr):
    if field.startswith("gpu_") and "cuda" not in model.get("supported_devices", ["cpu", "cuda"]):
        return tr("不适用（仅 CPU）")
    return format_budget(model.get(field), tr)


FIT_LABELS = {"recommended": "资源有余量", "tight": "资源较紧", "insufficient": "资源不足", "unknown": "需要确认"}


def localized_field(data, name, app):
    if app.ui_language == "en_US" and data.get(name + "_en"):
        return str(data[name + "_en"])
    return app.tr(str(data.get(name) or "未知"))


class _RecommendationMenu(ctk.CTkOptionMenu):
    def destroy(self):
        # CTk 5.2.2 DropdownMenu.destroy omits its scaling callback removal.
        # These menus are replaced after detection/language changes, so remove
        # that callback before Tk disposes of the underlying native menu.
        ctk.ScalingTracker.remove_widget(self._dropdown_menu._set_scaling, self._dropdown_menu)
        super().destroy()


class _InformationPanel(ctk.CTkFrame):
    def __init__(self, parent, page):
        super().__init__(parent, corner_radius=12)
        self.page = page
        self.app = page.app
        self.app.style(self, "surface")
        self.grid_columnconfigure(0, weight=1)
        self.body = self.app.style(ctk.CTkFrame(self), "surface")
        self.body.grid(row=1, column=0, sticky="ew", padx=14, pady=10)
        self.body.grid_columnconfigure(0, weight=1)

    def _clear(self):
        for child in self.body.winfo_children():
            child.destroy()

    def _text(self, parent, source, row, role="label", *, data=False, **kwargs):
        label = self.page._label(parent, source, role=role,
                                 font=self.app.font("small" if role == "muted" else "body"), **kwargs)
        if data:
            # Hardware names/model identifiers are data, never dictionary keys.
            self.app.translator.unbind(label)
            label.configure(text=source)
        label.grid(row=row, column=0, sticky="ew", pady=(3, 6))
        return label

    def _metric(self, parent, row, title, value):
        card = self.app.style(ctk.CTkFrame(parent, border_width=1, corner_radius=10), "surface")
        card.grid(row=row, column=0, sticky="ew", pady=5)
        card.grid_columnconfigure(0, weight=1)
        inner = self.app.style(ctk.CTkFrame(card, corner_radius=0), "surface")
        inner.grid(row=0, column=0, sticky="ew", padx=12, pady=7)
        inner.grid_columnconfigure(0, weight=1)
        self._text(inner, title, 0, "muted")
        self._text(inner, str(value), 1, data=True)
        return card

    def _metrics(self, parent, row, values):
        """Use two columns only when the font and available width permit it."""
        group = self.app.style(ctk.CTkFrame(parent), "surface")
        group.grid(row=row, column=0, sticky="ew")
        cards = [self._metric(group, index, title, value) for index, (title, value) in enumerate(values)]
        previous = None
        pending = None

        def arrange():
            nonlocal pending, previous
            pending = None
            if not group.winfo_exists():
                return
            # A stable outer viewport avoids changing column count in response
            # to this group's own requested width after re-gridding cards.
            viewport = self.master._parent_canvas if isinstance(self.master, ctk.CTkScrollableFrame) else parent
            width = viewport.winfo_width() / group._get_widget_scaling() - 60
            columns = 2 if width >= max(480, self.app.font().cget("size") * 22) else 1
            if columns == previous:
                return
            previous = columns
            group.grid_columnconfigure(0, weight=1, uniform="metrics")
            group.grid_columnconfigure(1, weight=1 if columns == 2 else 0, uniform="metrics" if columns == 2 else "")
            for index, card in enumerate(cards):
                card.grid(row=index // columns, column=index % columns, sticky="nsew",
                          padx=(0, 6) if columns == 2 and index % 2 == 0 else (0, 0))

        def schedule(_event=None):
            nonlocal pending
            if pending is None:
                pending = group.after_idle(arrange)

        group.bind("<Configure>", schedule, add="+")
        schedule()
        return group

    def _sources(self, parent, row, sources):
        if not sources:
            return
        self._text(parent, "官方参考", row, "accent")
        for index, source in enumerate(sources, row + 1):
            url = source.get("url", "")
            if not url.startswith("https://"):
                continue
            link = self._text(parent, "官方模型文档 ↗", index, "accent", cursor="hand2")
            link.bind("<Button-1>", lambda _event, target=url: webbrowser.open(target))


class SystemPanel(_InformationPanel):
    def __init__(self, parent, page, on_detect=None):
        super().__init__(parent, page)
        self.hardware = None
        self.recommendations = []
        self._selected_model_id = None
        self.on_detect = on_detect or (lambda hardware: None)
        self._request = 0
        self._loading = False
        self._error = None
        self.refresh_button = page._widget(ctk.CTkButton(self, text="刷新检测", command=self.refresh), "secondary")
        self.app.decorate(self.refresh_button, "refresh")
        page._fit_button(self.refresh_button)
        self.refresh_button.grid(row=0, column=0, sticky="w", padx=14, pady=(14, 0))
        self.retranslate()
        self.after(100, self.refresh)

    def refresh(self):
        self._request += 1
        request = self._request
        self._loading = True
        self._error = None
        self.refresh_button.configure(state="disabled")
        self.retranslate()

        def work():
            try:
                from core.hardware_info import detect_hardware
                from core.model_catalog import recommend_models
                hardware = detect_hardware()
                result = {"hardware": hardware, "recommendations": recommend_models(hardware)}
            except Exception as exc:
                result = {"error": exc}
            self.app.events.put({"kind": "settings_result", "dialog": self, "request": request, **result})

        threading.Thread(target=work, daemon=True, name="hardware-detection").start()

    def accept_result(self, event):
        if event.get("request") != self._request:
            return
        self._loading = False
        self._error = event.get("error")
        if not self._error:
            self.hardware = event["hardware"]
            self.recommendations = event["recommendations"]
            self.on_detect(self.hardware)
        self.refresh_button.configure(state="normal")
        self.retranslate()

    def retranslate(self):
        state = (self.app.ui_language, self._loading, self._error, id(self.hardware), id(self.recommendations))
        if self.__dict__.get("_render_state") == state:
            return
        self._render_state = state
        self._clear()
        row = 0
        self._text(self.body, "检测不会加载模型，可在转写期间刷新；可用资源会随其他程序变化。", row, "muted")
        row += 1
        if self._loading:
            self._text(self.body, "正在检测本机硬件…", row, "accent")
            row += 1
        if self._error:
            self._text(self.body, self.app.tr_error(self._error), row, "muted", data=True)
            row += 1
        if not self.hardware:
            if not self._loading and not self._error:
                self._text(self.body, "尚未检测", row, "muted")
            return
        hardware = self.hardware
        metrics = [
            ("处理器", hardware.get("cpu_model") or self.app.tr("未知")),
            ("逻辑线程", hardware.get("logical_cores") or self.app.tr("未知")),
            ("系统内存", format_gib(hardware.get("ram_total_gib"), self.app.tr)),
            ("当前可用内存", format_gib(hardware.get("ram_available_gib"), self.app.tr)),
            ("操作系统", hardware.get("system") or self.app.tr("未知")),
        ]
        for gpu in hardware.get("gpus", []):
            metrics.extend([("显卡", gpu.get("name") or self.app.tr("未知")),
                            ("显存总量", format_gib(gpu.get("total_vram_gib"), self.app.tr)),
                            ("当前可用显存", format_gib(gpu.get("free_vram_gib"), self.app.tr))])
        self._metrics(self.body, row, metrics)
        row += 1
        if not hardware.get("gpus"):
            self._text(self.body, "未发现独立显卡，可使用 CPU 识别。", row, "muted")
            row += 1
        if hardware.get("gpus") and not any(gpu.get("cuda") for gpu in hardware["gpus"]):
            self._text(self.body, "当前后端不支持此显卡加速，将按 CPU 方案评估。", row, "muted")
            row += 1
        if any("unknown" in code or "unavailable" in code for code in hardware.get("warnings", [])):
            self._text(self.body, "检测未能完整读取所有设备信息，未知项目请结合系统设置确认。", row, "muted")
            row += 1
        self._text(self.body, "本机模型建议", row, "accent")
        self._text(self.body, "建议基于当前可用内存和显存。选择更小的模型通常响应更快；较大的模型需要更多资源。", row + 1, "muted")
        row += 2
        first = [model for model in self.recommendations if model.get("top_recommendation")]
        if first:
            self._recommendation(first[0], row, prominent=True)
            row += 1
        self._text(self.body, "全部型号", row, "accent")
        row += 1
        if self.recommendations:
            self._text(self.body, "选择型号查看资源建议，也可使用左右按钮逐项浏览。", row, "muted")
            self._model_menu = self.page._widget(_RecommendationMenu(
                self.body, values=[model["name"] for model in self.recommendations],
                command=self._choose_recommendation, width=1, dynamic_resizing=False), "option")
            self._model_menu.grid(row=row + 1, column=0, sticky="ew", pady=(0, 8))
            navigation = self.app.style(ctk.CTkFrame(self.body), "surface")
            navigation.grid(row=row + 2, column=0, sticky="ew")
            navigation.grid_columnconfigure(1, weight=1)
            previous = self.page._widget(ctk.CTkButton(navigation, text="", command=lambda: self._step_recommendation(-1)), "secondary")
            self.app.decorate(previous, "chevron_left")
            self.page._fit_button(previous)
            previous.grid(row=0, column=0, padx=(0, 8))
            self._model_position = self._text(navigation, "", 0, "muted", data=True)
            self._model_position.grid_configure(column=1)
            following = self.page._widget(ctk.CTkButton(navigation, text="", command=lambda: self._step_recommendation(1)), "secondary")
            self.app.decorate(following, "chevron_right")
            self.page._fit_button(following)
            following.grid(row=0, column=2, padx=(8, 0))
            self._recommendation_body = self.app.style(ctk.CTkFrame(self.body), "surface")
            self._recommendation_body.grid(row=row + 3, column=0, sticky="ew")
            self._recommendation_body.grid_columnconfigure(0, weight=1)
            self._render_selected_recommendation()
            row += 4
        self._text(self.body, "本机尚未测速，资源适配不代表可实时听写。较小模型通常响应更快。", row, "muted")
        self._text(self.body, "以下是运行预算的估算范围，不是精确最低要求或速度保证。音频长度、精度、后台应用及推理版本都会影响占用。", row + 1, "muted")

    def _selected_recommendation_index(self):
        selected = self.__dict__.get("_selected_model_id")
        return next((index for index, model in enumerate(self.recommendations) if model["id"] == selected), 0)

    def _choose_recommendation(self, label):
        self._selected_model_id = next(model["id"] for model in self.recommendations if model["name"] == label)
        self._render_selected_recommendation()

    def _step_recommendation(self, direction):
        index = (self._selected_recommendation_index() + direction) % len(self.recommendations)
        self._selected_model_id = self.recommendations[index]["id"]
        self._render_selected_recommendation()

    def _render_selected_recommendation(self):
        # One card at a time bounds the underlying Tk canvas height even with
        # 48 px fonts. All variants remain available in the menu and arrows.
        index = self._selected_recommendation_index()
        model = self.recommendations[index]
        self._selected_model_id = model["id"]
        self._model_menu.set(model["name"])
        self._model_position.configure(text=f"{index + 1} / {len(self.recommendations)}")
        for child in self._recommendation_body.winfo_children():
            child.destroy()
        self._recommendation(model, 0, parent=self._recommendation_body)

    def _recommendation(self, model, row, prominent=False, parent=None):
        card = self.app.style(ctk.CTkFrame(parent or self.body, border_width=1), "surface")
        card.grid(row=row, column=0, sticky="ew", pady=6)
        card.grid_columnconfigure(0, weight=1)
        content = self.app.style(ctk.CTkFrame(card), "surface")
        content.grid(row=0, column=0, sticky="ew", padx=12, pady=8)
        content.grid_columnconfigure(0, weight=1)
        title = model["name"]
        if prominent:
            title = self.app.tr("本机首选") + " · " + title
        elif model.get("preferred"):
            title = self.app.tr("系列建议") + " · " + title
        self._text(content, title, 0, data=True)
        device = "CUDA 显卡" if model.get("device") == "cuda" else "CPU"
        fit = self.app.tr(FIT_LABELS.get(model.get("fit"), "需要确认"))
        self._text(content, fit + " · " + self.app.tr(device), 1, "accent", data=True)
        if "cuda" in model.get("supported_devices", ["cpu", "cuda"]):
            self._text(content, "CPU 内存 {ram} · GPU 显存 {vram}", 2, "muted", text_values={
                "ram": format_budget(model.get("cpu_ram_gib"), self.app.tr),
                "vram": format_budget(model.get("gpu_vram_gib"), self.app.tr)})
        else:
            self._text(content, "CPU 内存 {ram} · 仅 CPU", 2, "muted", text_values={
                "ram": format_budget(model.get("cpu_ram_gib"), self.app.tr)})
        self._text(content, localized_field(model, "reason", self.app), 3, "muted", data=True)
        if prominent and model.get("preference_reason"):
            self._text(content, localized_field(model, "preference_reason", self.app), 4, "muted", data=True)

class ModelInfoPanel(_InformationPanel):
    def __init__(self, parent, page):
        super().__init__(parent, page)
        self.model = None
        self.hardware = None
        self.fit = None
        self._request = 0
        self._loading = False
        self._error = None
        self.retranslate()

    def refresh(self, model_type, path):
        path = path.strip()
        selection = (model_type, path)
        if self.__dict__.get("_selection") == selection and not self._error:
            return
        self._selection = selection
        self._request += 1
        request = self._request
        if not path:
            self._loading = False
            self._error = None
            self.model = None
            self.fit = None
            self.retranslate()
            return
        self._loading = True
        self._error = None
        self.retranslate()

        def work():
            try:
                from core.model_catalog import describe_model
                result = {"model": describe_model(model_type, path)}
            except Exception as exc:
                result = {"error": exc}
            self.app.events.put({"kind": "settings_result", "dialog": self, "request": request, **result})

        threading.Thread(target=work, daemon=True, name="model-information").start()

    def accept_result(self, event):
        if event.get("request") != self._request:
            return
        self._loading = False
        self._error = event.get("error")
        self.model = event.get("model")
        self.set_hardware(self.hardware)

    def set_hardware(self, hardware):
        self.hardware = hardware
        self.fit = None
        if hardware and self.model:
            from core.model_catalog import assess_model
            self.fit = assess_model(self.model, hardware)
        self.retranslate()

    def retranslate(self):
        state = (self.app.ui_language, self._loading, self._error, id(self.model), id(self.fit))
        if self.__dict__.get("_render_state") == state:
            return
        self._render_state = state
        self._clear()
        self._text(self.body, "所选模型信息", 0, "accent")
        if self._loading:
            self._text(self.body, "正在读取模型信息…", 1, "muted")
            return
        if self._error:
            self._text(self.body, self.app.tr_error(self._error), 1, "muted", data=True)
            return
        if not self.model:
            self._text(self.body, "尚未选择模型。选择本地模型后显示型号与资源预算。", 1, "muted")
            return
        model = self.model
        self._text(self.body, str(model.get("name") or ""), 1, data=True)
        metrics = (
            ("参数规模", str(model.get("parameters") or self.app.tr("未知"))),
            ("本地文件体积", format_gib(model.get("disk_gib"), self.app.tr)),
            ("CPU 推理内存预算", model_budget(model, "cpu_ram_gib", self.app.tr)),
            ("GPU 推理显存预算", model_budget(model, "gpu_vram_gib", self.app.tr)),
            ("GPU 推理主机内存预算", model_budget(model, "gpu_host_ram_gib", self.app.tr)),
            ("速度提示", localized_field(model, "speed_note", self.app)),
        )
        self._metrics(self.body, 2, metrics)
        row = 3
        if self.fit:
            device = self.app.tr("CUDA 显卡" if self.fit.get("device") == "cuda" else "CPU")
            if self.fit.get("gpu_name") and self.fit.get("device") == "cuda":
                device += " · " + self.fit["gpu_name"]
            self._metrics(self.body, row, [
                ("本机适配度", self.app.tr(FIT_LABELS.get(self.fit.get("fit"), "需要确认"))),
                ("建议运行设备", device),
            ])
            self._text(self.body, localized_field(self.fit, "reason", self.app), row + 1, "muted", data=True)
        else:
            self._text(self.body, "刷新本机配置后可显示适配建议。" if not self.hardware else "没有识别到具体型号，选择完整模型路径后可显示对应预算。", row, "muted")
        self._text(self.body, localized_field(model, "budget_note", self.app), row + 2, "muted", data=True)
        row += 3
        official = []
        if model.get("official_vram_gb") is not None:
            official.append(("官方约需显存（GB）", str(model["official_vram_gb"])))
        if model.get("official_speed_relative") is not None:
            official.append(("官方相对速度（A100，large = 1）", f"{model['official_speed_relative']:g}×"))
        if official:
            self._metrics(self.body, row, official)
            row += 1
        self._sources(self.body, row, model.get("sources", []))
