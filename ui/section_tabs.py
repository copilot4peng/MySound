"""Scrollable section navigation with a small CTkTabview-compatible API."""

from __future__ import annotations

import customtkinter as ctk


class SectionTabs(ctk.CTkFrame):
    """Keep large navigation labels reachable without squeezing page content."""

    def __init__(self, parent, app):
        super().__init__(parent, corner_radius=0)
        self.app = app
        app.style(self, "frame")
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        self._tabs = {}
        self._buttons = {}
        self._current = ""
        self.navigation = app.style(ctk.CTkScrollableFrame(self, orientation="horizontal", height=52, corner_radius=10), "scroll")
        self.navigation.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        self.navigation.bind("<Button-4>", lambda _event: self._scroll(-2), add="+")
        self.navigation.bind("<Button-5>", lambda _event: self._scroll(2), add="+")

    def _scroll(self, direction):
        self.navigation._parent_canvas.xview_scroll(direction, "units")
        return "break"

    def add(self, name, *, icon=None, source=None):
        if name in self._tabs:
            raise ValueError(f"Section already exists: {name}")
        frame = self.app.style(ctk.CTkFrame(self, corner_radius=0), "surface")
        button = self.app.style(ctk.CTkButton(self.navigation, text=source or name,
                                             command=lambda item=frame: self._show_frame(item)), "secondary")
        if source:
            self.app.translate_widget(button, source)
        if icon:
            self.app.decorate(button, icon)
        button.grid(row=0, column=len(self._tabs), padx=4, pady=4)
        button.bind("<Button-4>", lambda _event: self._scroll(-2), add="+")
        button.bind("<Button-5>", lambda _event: self._scroll(2), add="+")
        self._tabs[name] = frame
        self._buttons[name] = button
        self.reflow()
        if not self._current:
            self.set(name)
        return frame

    def _show_frame(self, frame):
        self.set(next(name for name, candidate in self._tabs.items() if candidate is frame))

    def tab(self, name):
        return self._tabs[name]

    def set(self, name):
        if name not in self._tabs:
            raise ValueError(f"Unknown section: {name}")
        self._current = name
        for key, frame in self._tabs.items():
            if key == name:
                frame.grid(row=1, column=0, sticky="nsew")
            else:
                frame.grid_remove()
            self.app.style(self._buttons[key], "selected_history" if key == name else "secondary")
        self.after_idle(self._reveal_current)

    def get(self):
        return self._current

    def rename(self, old_name, new_name):
        if old_name == new_name:
            return
        if new_name in self._tabs:
            raise ValueError(f"Section already exists: {new_name}")
        self._tabs = {new_name if key == old_name else key: value for key, value in self._tabs.items()}
        self._buttons = {new_name if key == old_name else key: value for key, value in self._buttons.items()}
        self._buttons[new_name].configure(text=new_name)
        if self._current == old_name:
            self._current = new_name

    def reflow(self):
        font = self.app.font()
        line = font.metrics("linespace")
        self.navigation.configure(height=line + 28)
        for button in self._buttons.values():
            icon_space = max(24, int(font.cget("size") * 1.25)) + 12
            button.configure(height=line + 18, width=font.measure(button.cget("text")) + icon_space + 32)
        self.after_idle(self._reveal_current)

    def _reveal_current(self):
        if not self.winfo_exists() or self._current not in self._buttons:
            return
        canvas = self.navigation._parent_canvas
        button = self._buttons[self._current]
        total = max(1, self.navigation.winfo_reqwidth())
        viewport = canvas.winfo_width()
        left, right = canvas.xview()
        start = button.winfo_x()
        end = start + button.winfo_width()
        if start < left * total:
            canvas.xview_moveto(start / total)
        elif end > right * total:
            canvas.xview_moveto(max(0, end - viewport) / total)

    def cget(self, name):
        if name == "text_color":
            return self.app.theme.palette["text"]
        return super().cget(name)
