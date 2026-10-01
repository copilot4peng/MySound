"""Keep Tk's pixel-font conversion aligned with Xft on Linux/Xwayland.

Some desktops advertise 96 DPI through X11 screen dimensions while setting
Xft.dpi to 192. Tk then converts a requested 16-pixel font to 12 points and Xft
renders those points at 32 pixels. Matching Tk's conversion to Xft fixes text
and CustomTkinter's shape glyphs without changing widget or font sizes.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import logging
import math
import re
import sys


log = logging.getLogger(__name__)
_XFT_DPI = re.compile(r"^\s*Xft\.dpi\s*:\s*([^\r\n]+)\s*$", re.MULTILINE)


def parse_xft_dpi(resources: str | bytes | None) -> float | None:
    """Read only the exact Xft.dpi resource; ignore malformed/extreme values."""
    if not resources:
        return None
    if isinstance(resources, bytes):
        resources = resources.decode("utf-8", errors="replace")
    if not isinstance(resources, str):
        return None
    for match in _XFT_DPI.finditer(resources):
        try:
            dpi = float(match.group(1).strip())
        except ValueError:
            continue
        if math.isfinite(dpi) and 48 <= dpi <= 768:
            return dpi
    return None


def _read_x_resources() -> bytes | None:
    """Read the server's resource string without modifying the X11 display."""
    library = ctypes.util.find_library("X11")
    if not library:
        return None
    x11 = ctypes.CDLL(library)
    x11.XOpenDisplay.argtypes = [ctypes.c_char_p]
    x11.XOpenDisplay.restype = ctypes.c_void_p
    x11.XResourceManagerString.argtypes = [ctypes.c_void_p]
    x11.XResourceManagerString.restype = ctypes.c_char_p
    x11.XCloseDisplay.argtypes = [ctypes.c_void_p]
    x11.XCloseDisplay.restype = ctypes.c_int
    display = x11.XOpenDisplay(None)
    if not display:
        return None
    try:
        # c_char_p copies the resource into Python bytes while display is open.
        return x11.XResourceManagerString(display)
    finally:
        x11.XCloseDisplay(display)


def configure_tk_dpi(root) -> float | None:
    """Set this Tk interpreter's scaling before creating fonts or widgets.

    Applies only to Linux Tk using X11. Missing resources or unavailable X11
    libraries leave Tk's own scaling intact. No external executable is needed.
    Returns the applied Xft DPI, or None when the original scaling is retained.
    """
    if not sys.platform.startswith("linux"):
        return None
    try:
        if root.tk.call("tk", "windowingsystem") != "x11":
            return None
        dpi = parse_xft_dpi(_read_x_resources())
        if dpi is None:
            return None
        root.tk.call("tk", "scaling", dpi / 72.0)
        log.debug("Aligned Tk font scaling with Xft.dpi=%s", dpi)
        return dpi
    except Exception:
        log.debug("Could not read Xft DPI; retaining Tk's native scaling", exc_info=True)
        return None
