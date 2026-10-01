"""Theme preset cards keep color, surface, and text choices together."""

from __future__ import annotations

import customtkinter as ctk

from ui.i18n import register_messages
from ui.theme import THEMES


class ThemeCards(ctk.CTkFrame):
    def __init__(self, parent, page, current, select):
        super().__init__(parent, corner_radius=0)
        self.page = page
        self.app = page.app
        self.current = current
        self.select = select
        self.app.style(self, "surface")
        self.grid_columnconfigure(0, weight=1)
        self._cards = {}
        self._layout_pending = None
        self._columns = None
        self._viewport = parent._parent_canvas if isinstance(parent, ctk.CTkScrollableFrame) else parent
        for row, (key, theme) in enumerate(THEMES.items()):
            register_messages({theme["name"]: theme["name_en"], theme["description"]: theme["description_en"]})
            card = self.app.style(ctk.CTkFrame(self, border_width=2, corner_radius=12), "surface")
            card.grid(row=row, column=0, sticky="ew", pady=6)
            card.grid_columnconfigure(0, weight=1)
            content = self.app.style(ctk.CTkFrame(card, corner_radius=0), "surface")
            content.grid(row=0, column=0, sticky="ew", padx=14, pady=12)
            content.grid_columnconfigure(0, weight=1)
            choose = page._widget(ctk.CTkButton(content, text=theme["name"], anchor="w", command=lambda item=key: self._select(item)), "secondary")
            self.app.decorate(choose, "palette")
            choose.grid(row=0, column=0, sticky="ew")
            samples = self.app.style(ctk.CTkFrame(content, corner_radius=0), "surface")
            samples.grid(row=1, column=0, sticky="ew", pady=(12, 8))
            for column, color in enumerate(theme["preview"]):
                samples.grid_columnconfigure(column, weight=1)
                swatch = ctk.CTkFrame(samples, fg_color=color, height=20, corner_radius=5)
                swatch.grid(row=0, column=column, sticky="ew", padx=(0, 5))
                swatch.bind("<Button-1>", lambda _event, item=key: self._select(item), add="+")
            note = page._label(content, theme["description"], role="muted", font=self.app.font("small"))
            note.grid(row=2, column=0, sticky="ew")
            self._cards[key] = (card, choose)
        self._layout_binding = self._viewport.bind("<Configure>", self._schedule_layout, add="+")
        self.retranslate()

    def _select(self, key):
        self.select(key)
        self.retranslate()

    def retranslate(self):
        selected = self.current()
        for key, (card, button) in self._cards.items():
            card.configure(border_color=self.app.theme.palette["accent" if key == selected else "border"])
            self.app.theme.style(button, "selected_history" if key == selected else "secondary")
            self.page._fit_button(button)
        self._schedule_layout()

    def _schedule_layout(self, _event=None):
        if self._layout_pending is None and self.winfo_exists():
            self._layout_pending = self.after_idle(self._arrange)

    def _arrange(self):
        self._layout_pending = None
        if not self.winfo_exists():
            return
        width = self._viewport.winfo_width() / self._get_widget_scaling() - 24
        font_size = self.app.font().cget("size")
        required = 2 * (max(button.cget("width") for _, button in self._cards.values()) + 32) + 12
        columns = 2 if font_size < 40 and width >= max(480, font_size * 22, required) else 1
        if columns == self._columns:
            return
        self._columns = columns
        self.grid_columnconfigure(0, weight=1, uniform="presets")
        self.grid_columnconfigure(1, weight=1 if columns == 2 else 0, uniform="presets" if columns == 2 else "")
        for index, (card, _) in enumerate(self._cards.values()):
            card.grid(row=index // columns, column=index % columns, sticky="nsew",
                      padx=(0, 10) if columns == 2 and index % 2 == 0 else (0, 0))

    def destroy(self):
        if self._layout_pending is not None:
            self.after_cancel(self._layout_pending)
        if self._layout_binding:
            self._viewport.unbind("<Configure>", self._layout_binding)
        super().destroy()
