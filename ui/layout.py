"""Small geometry helpers with no Tk or display dependency.

All dimensions passed to ``clamp_sash`` and ``ellipsize`` must use the same
units. Tk sash positions are physical pixels; CustomTkinter dimensions may be
logical pixels when widget scaling is enabled.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import math


DEFAULT_LAYOUT = {
    "width": 1180,
    "height": 780,
    "maximized": False,
    "sidebar_ratio": 0.23,
    "editor_ratio": 0.40,
}


def _number(value, fallback: float) -> float:
    if isinstance(value, bool):
        return fallback
    try:
        result = float(value)
        return result if math.isfinite(result) else fallback
    except (TypeError, ValueError, OverflowError):
        return fallback


def clamp_layout(layout, screen_width, screen_height) -> dict:
    """Restore a usable window size after a monitor or scaling change.

    Leave 80 pixels for desktop decorations. Normally the minimum is 900×620;
    on a smaller display the available screen space wins over that minimum.
    Ratios retain the complete 0..1 range; ``clamp_sash`` enforces actual pane
    minimum sizes once the widgets have their final dimensions. Narrow panes
    on a large monitor must restore to the position the user actually chose.
    """
    data = layout if isinstance(layout, Mapping) else {}
    available_width = max(1, round(_number(screen_width, 1920)) - 80)
    available_height = max(1, round(_number(screen_height, 1080)) - 80)
    width = round(_number(data.get("width"), DEFAULT_LAYOUT["width"]))
    height = round(_number(data.get("height"), DEFAULT_LAYOUT["height"]))
    return {
        "width": max(min(900, available_width), min(width, available_width)),
        "height": max(min(620, available_height), min(height, available_height)),
        "maximized": data.get("maximized") is True,
        "sidebar_ratio": max(0.0, min(1.0, _number(data.get("sidebar_ratio"), 0.23))),
        "editor_ratio": max(0.0, min(1.0, _number(data.get("editor_ratio"), 0.40))),
    }


def clamp_sash(total, ratio, min_first, min_second, sash_width=8) -> int:
    """Return the first pane's extent, measured from the container's origin.

    ``ratio`` is relative to the full container extent, including its sash. If
    a container cannot fit both minima, divide its usable space in proportion
    to those minima; never return an out-of-bounds or negative sash position.
    """
    total = max(0, round(_number(total, 0)))
    available = max(0, total - max(0, round(_number(sash_width, 8))))
    first = max(0, round(_number(min_first, 0)))
    second = max(0, round(_number(min_second, 0)))
    ratio = max(0.0, min(1.0, _number(ratio, 0.5)))
    if first + second > available:
        return round(available * first / (first + second))
    return max(first, min(round(total * ratio), available - second))


def ellipsize(text: str, max_width, measure_callable: Callable[[str], float]) -> str:
    """Fit text to a measured width, including a visible trailing ellipsis."""
    text = str(text)
    width = _number(max_width, 0)
    if width <= 0:
        return ""
    if measure_callable(text) <= width:
        return text
    ellipsis = "…"
    if measure_callable(ellipsis) > width:
        return ""
    low, high = 0, len(text)
    while low < high:
        middle = (low + high + 1) // 2
        if measure_callable(text[:middle] + ellipsis) <= width:
            low = middle
        else:
            high = middle - 1
    return text[:low] + ellipsis
