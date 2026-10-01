"""Capture only our test window, including under XWayland without a root pixmap."""
import ctypes as c
from ctypes.util import find_library


def capture_window(window):
    from PIL import Image

    class XImage(c.Structure):
        _fields_ = [
            ("width", c.c_int), ("height", c.c_int), ("xoffset", c.c_int),
            ("format", c.c_int), ("data", c.c_void_p), ("byte_order", c.c_int),
            ("bitmap_unit", c.c_int), ("bitmap_bit_order", c.c_int),
            ("bitmap_pad", c.c_int), ("depth", c.c_int),
            ("bytes_per_line", c.c_int), ("bits_per_pixel", c.c_int),
            ("red_mask", c.c_ulong), ("green_mask", c.c_ulong), ("blue_mask", c.c_ulong),
        ]

    library = c.CDLL(find_library("X11"))
    library.XOpenDisplay.argtypes = [c.c_char_p]
    library.XOpenDisplay.restype = c.c_void_p
    library.XGetImage.argtypes = [c.c_void_p, c.c_ulong, c.c_int, c.c_int, c.c_uint, c.c_uint, c.c_ulong, c.c_int]
    library.XGetImage.restype = c.POINTER(XImage)
    library.XDestroyImage.argtypes = [c.POINTER(XImage)]
    library.XCloseDisplay.argtypes = [c.c_void_p]
    display = library.XOpenDisplay(None)
    if not display:
        raise RuntimeError("Cannot open test window display")
    pointer = None
    try:
        pointer = library.XGetImage(display, window.winfo_id(), 0, 0, window.winfo_width(), window.winfo_height(), c.c_ulong(-1), 2)
        if not pointer:
            raise RuntimeError("Cannot capture test window")
        image = pointer.contents
        if image.bits_per_pixel != 32 or image.byte_order != 0 or image.red_mask != 0xFF0000:
            raise RuntimeError("Unsupported test display pixel format")
        pixels = c.string_at(image.data, image.bytes_per_line * image.height)
        return Image.frombytes("RGB", (image.width, image.height), pixels, "raw", "BGRX", image.bytes_per_line, 1)
    finally:
        if pointer:
            library.XDestroyImage(pointer)
        library.XCloseDisplay(display)
