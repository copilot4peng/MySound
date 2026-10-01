"""Apply complete light/dark palettes to registered CustomTkinter widgets."""

from __future__ import annotations

import tkinter as tk
import weakref


MODES = ("dark", "light")

# Each preset coordinates every surface and control, rather than applying an
# accent to a shared gray background. Ordered values keep the eight palettes
# easy to compare while get_palette exposes ordinary named dictionaries.
_PALETTE_KEYS = (
    "background", "surface", "input", "text", "muted", "border", "secondary",
    "secondary_hover", "disabled", "scrollbar", "scrollbar_hover", "accent",
    "accent_hover", "accent_text", "selected",
)


def _palette(mode, values):
    palette = dict(zip(_PALETTE_KEYS, values))
    palette.update(
        on_accent="#FFFFFF",
        danger="#B83E4A" if mode == "dark" else "#B72F41",
        danger_hover="#96313C" if mode == "dark" else "#952333",
        success="#81D8AD" if mode == "dark" else "#19754E",
        warning="#F4C57D" if mode == "dark" else "#91500A",
    )
    return palette


THEMES = {
    "blue": {
        "name": "深海蓝", "name_en": "Deep Ocean",
        "description": "沉静蓝调，清晰专注", "description_en": "Calm blue, clear focus",
        "dark": _palette("dark", (
            "#0E192A", "#17263B", "#0D2035", "#EEF5FF", "#AABDD6", "#405D80",
            "#263D59", "#345271", "#91A6C0", "#507091", "#7194B8", "#2461BA",
            "#194C99", "#91BEFF", "#244A77",
        )),
        "light": _palette("light", (
            "#EAF1FA", "#F8FBFF", "#FFFFFF", "#172B45", "#4F6684", "#B7CBE3",
            "#DDE9F8", "#C9DCF3", "#657F9E", "#8DAACB", "#6585AC", "#2461BA",
            "#194C99", "#174F9E", "#C9DFFF",
        )),
    },
    "green": {
        "name": "森林绿", "name_en": "Forest Green",
        "description": "自然绿意，舒适柔和", "description_en": "Natural greens, gentle contrast",
        "dark": _palette("dark", (
            "#101E19", "#192D25", "#0E241B", "#EFF8F1", "#ABC7B6", "#456957",
            "#294637", "#385B46", "#91AF9D", "#547D66", "#76A28A", "#18754D",
            "#105C3B", "#8BDCB2", "#25563E",
        )),
        "light": _palette("light", (
            "#EAF3EC", "#F8FCF8", "#FFFFFF", "#1B3427", "#536F5F", "#B6CFBF",
            "#DBEBDF", "#C7DFCD", "#668773", "#8FB09B", "#658B75", "#18754D",
            "#105C3B", "#0D613E", "#C4E5CF",
        )),
    },
    "purple": {
        "name": "鸢尾紫", "name_en": "Iris Violet",
        "description": "柔雅紫调，安静灵感", "description_en": "Soft violet, quiet inspiration",
        "dark": _palette("dark", (
            "#1A1628", "#292039", "#20172F", "#F6F0FF", "#C4B4D8", "#63517E",
            "#40314F", "#524063", "#AA96BF", "#79658F", "#9E85B6", "#7746B7",
            "#5E3494", "#D1B0FF", "#533A76",
        )),
        "light": _palette("light", (
            "#F0ECF7", "#FCF9FF", "#FFFFFF", "#342442", "#6C5A7E", "#CEBFDF",
            "#E9E0F3", "#DCCEEB", "#8A749D", "#B29BC7", "#8D71A6", "#7746B7",
            "#5E3494", "#6636A1", "#E4D3F8",
        )),
    },
    "orange": {
        "name": "暖沙橙", "name_en": "Warm Sand",
        "description": "温暖沙色，轻松阅读", "description_en": "Warm sand, easy reading",
        "dark": _palette("dark", (
            "#241B16", "#35271E", "#2B1E16", "#FFF4EA", "#D3BBA6", "#795D48",
            "#503927", "#644C36", "#BCA088", "#927055", "#B89271", "#AA4D0D",
            "#873B09", "#F6BA80", "#704722",
        )),
        "light": _palette("light", (
            "#F8EFE5", "#FFFBF5", "#FFFFFF", "#432D1F", "#80664F", "#DEC6AE",
            "#F1E2D0", "#E7D2B9", "#9C8068", "#C5A788", "#A48361", "#AA4D0D",
            "#873B09", "#914007", "#F5D9B6",
        )),
    },
}

for _theme in THEMES.values():
    _theme["preview"] = tuple(_theme["dark"][key] for key in ("background", "surface", "accent"))

COLORS = tuple(THEMES)


def get_palette(mode="dark", color="blue") -> dict:
    """Return a copy so preview cards can use any preset without applying it."""
    mode = mode if mode in MODES else "dark"
    color = color if color in THEMES else "blue"
    return dict(THEMES[color][mode])


class ThemeManager:
    """Keep registered widgets up to date without rebuilding their state.

    Calling ``style`` again changes the widget's role and replaces its explicit
    overrides. Overrides always take precedence over palette values. Run these
    methods on the Tk main thread, like other widget configuration operations.
    """

    def __init__(self, mode="dark", color="blue"):
        self._widgets = weakref.WeakKeyDictionary()
        self.mode = mode if mode in MODES else "dark"
        self.color = color if color in COLORS else "blue"
        self.apply(self.mode, self.color)

    @property
    def palette(self) -> dict:
        return get_palette(self.mode, self.color)

    def _options(self, role):
        p = self.palette
        button = {"fg_color": p["accent"], "hover_color": p["accent_hover"],
                  "text_color": p["on_accent"], "text_color_disabled": "#D4DDE9"}
        secondary = {"fg_color": p["secondary"], "hover_color": p["secondary_hover"],
                     "text_color": p["text"], "text_color_disabled": p["disabled"]}
        roles = {
            "frame": {"fg_color": p["background"]},
            "surface": {"fg_color": p["surface"], "border_color": p["border"]},
            "card": {"fg_color": p["surface"], "border_color": p["border"], "border_width": 1},
            "selected_card": {"fg_color": p["selected"], "border_color": p["accent_text"], "border_width": 2},
            "label": {"text_color": p["text"]},
            "title": {"text_color": p["text"]},
            "muted": {"text_color": p["muted"]},
            "accent": {"text_color": p["accent_text"]},
            "button": button,
            "secondary": secondary,
            "danger": {**button, "fg_color": p["danger"], "hover_color": p["danger_hover"]},
            "entry": {"fg_color": p["input"], "border_color": p["border"],
                      "text_color": p["text"], "placeholder_text_color": p["muted"]},
            "option": {"fg_color": p["secondary"], "button_color": p["secondary_hover"],
                       "button_hover_color": p["selected"], "text_color": p["text"],
                       "text_color_disabled": p["disabled"], "dropdown_fg_color": p["surface"],
                       "dropdown_hover_color": p["selected"], "dropdown_text_color": p["text"]},
            "textbox": {"fg_color": p["input"], "border_color": p["border"],
                        "text_color": p["text"], "scrollbar_button_color": p["scrollbar"],
                        "scrollbar_button_hover_color": p["scrollbar_hover"]},
            "scroll": {"fg_color": p["surface"], "border_color": p["border"],
                       "scrollbar_fg_color": p["surface"], "scrollbar_button_color": p["scrollbar"],
                       "scrollbar_button_hover_color": p["scrollbar_hover"],
                       "label_fg_color": p["surface"], "label_text_color": p["text"]},
            "tabview": {"fg_color": p["surface"], "border_color": p["border"],
                        "segmented_button_fg_color": p["secondary"],
                        "segmented_button_selected_color": p["selected"],
                        "segmented_button_selected_hover_color": p["selected"],
                        "segmented_button_unselected_color": p["secondary"],
                        "segmented_button_unselected_hover_color": p["secondary_hover"],
                        "text_color": p["text"], "text_color_disabled": p["disabled"]},
            "history": secondary,
            "selected_history": {**secondary, "fg_color": p["selected"], "hover_color": p["selected"]},
            "paned": {"bg": p["border"]},
            "progress": {"fg_color": p["secondary"], "progress_color": p["accent"]},
        }
        if role not in roles:
            raise ValueError(f"Unknown theme role: {role}")
        return roles[role]

    def style(self, widget, role, **overrides):
        options = {**self._options(role), **overrides}
        widget.configure(**options)
        self._widgets[widget] = (role, dict(overrides))
        return widget

    def apply(self, mode, color):
        import customtkinter as ctk

        self.mode = mode if mode in MODES else "dark"
        self.color = color if color in COLORS else "blue"
        ctk.set_appearance_mode(self.mode)
        for widget, (role, overrides) in list(self._widgets.items()):
            try:
                if not widget.winfo_exists():
                    self._widgets.pop(widget, None)
                    continue
                widget.configure(**{**self._options(role), **overrides})
            except tk.TclError:
                # A widget may have been removed while Tk processed a theme
                # callback; invalid options still surface for living widgets.
                try:
                    exists = widget.winfo_exists()
                except tk.TclError:
                    exists = False
                if exists:
                    raise
                self._widgets.pop(widget, None)
