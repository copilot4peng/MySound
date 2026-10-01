"""Shared CTkFont objects make interface text sizing update in place."""

from __future__ import annotations

from storage.config_store import normalize_font_size


FONT_SIZES = (12, 14, 16, 18, 20, 22, 24, 28, 32, 36, 40, 48)
_OFFSETS = {"body": 0, "title": 12, "brand": 13, "section": 3, "editor": 4, "small": -1}
_BOLD = {"title", "brand", "section"}


class Typography:
    """Create each semantic font once; existing widgets share later changes.

    Construct after the Tk root is initialized. ``font_factory`` is injectable
    for headless tests and otherwise uses CustomTkinter's CTkFont.
    """

    def __init__(self, base_size=14, *, font_factory=None):
        if font_factory is None:
            from customtkinter import CTkFont
            font_factory = CTkFont
        self.base_size = normalize_font_size(base_size)
        self._font_factory = font_factory
        self._fonts = {}

    def _size(self, role):
        if role == "small":
            return max(12, round(self.base_size * 0.8))
        return max(12, self.base_size + _OFFSETS[role])

    def font(self, role="body"):
        if role not in _OFFSETS:
            raise ValueError(f"Unknown font role: {role}")
        if role not in self._fonts:
            self._fonts[role] = self._font_factory(
                size=self._size(role), weight="bold" if role in _BOLD else "normal",
            )
        return self._fonts[role]

    def apply(self, size) -> int:
        self.base_size = normalize_font_size(size)
        for role, font in self._fonts.items():
            font.configure(size=self._size(role))
        return self.base_size


FontManager = Typography
