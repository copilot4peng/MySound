import gc
import unittest
import weakref

from ui.icons import ICON_NAMES, IconManager, draw_icon
from ui.theme import get_palette


class FakeImage:
    def __init__(self, **options):
        self.options = options


class FakeWidget:
    def __init__(self):
        self.options = {}
        self.exists = True

    def configure(self, **options):
        self.options.update(options)

    def winfo_exists(self):
        return self.exists


class IconTests(unittest.TestCase):
    def test_each_icon_is_transparent_and_visible_at_both_font_extremes(self):
        for name in ICON_NAMES:
            for size in (16, 52):
                image = draw_icon(name, "#ABCDEF", size)
                self.assertEqual(image.mode, "RGBA")
                self.assertEqual(image.size, (size, size))
                alpha = image.getchannel("A")
                self.assertIsNotNone(alpha.getbbox(), name)
                self.assertEqual(image.getpixel((0, 0))[3], 0, name)

    def test_theme_and_size_change_refresh_images_without_changing_text(self):
        manager = IconManager(get_palette("dark", "blue"), 14, image_factory=FakeImage)
        widget = FakeWidget()
        widget.configure(text="保存修改")
        self.assertIs(manager.bind(widget, "save", tone="on_accent"), widget)
        original = widget.options["image"]
        self.assertEqual(original.options["size"], (18, 18))
        manager.apply(get_palette("light", "orange"), 48)
        refreshed = widget.options["image"]
        self.assertIsNot(refreshed, original)
        self.assertEqual(refreshed.options["size"], (52, 52))
        self.assertEqual(widget.options["text"], "保存修改")
        self.assertEqual(widget.options["compound"], "left")
        # The default light/dark assets both use the selected palette, with
        # white strokes remaining legible on primary buttons.
        image = refreshed.options["light_image"]
        pixels = [image.getpixel((x, y)) for y in range(image.height) for x in range(image.width)]
        opaque = [pixel for pixel in pixels if pixel[3] == 255]
        self.assertTrue(opaque)
        self.assertTrue(all(pixel[:3] == (255, 255, 255) for pixel in opaque))

    def test_icons_follow_text_palette_and_aliases(self):
        manager = IconManager(get_palette("dark", "blue"), image_factory=FakeImage)
        widget = FakeWidget()
        manager.bind(widget, "microphone")
        dark = widget.options["image"].options["light_image"].tobytes()
        manager.apply(get_palette("light", "green"), 14)
        light = widget.options["image"].options["light_image"].tobytes()
        self.assertNotEqual(dark, light)
        self.assertEqual(draw_icon("hardware").tobytes(), draw_icon("monitor").tobytes())

    def test_destroyed_widgets_are_skipped_and_not_retained(self):
        manager = IconManager(get_palette(), image_factory=FakeImage)
        widget = FakeWidget()
        manager.bind(widget, "settings")
        widget.exists = False
        original = widget.options["image"]
        manager.apply(get_palette("light", "purple"), 24)
        self.assertIs(widget.options["image"], original)
        self.assertNotIn(widget, manager._widgets)
        live = FakeWidget()
        manager.bind(live, "folder")
        reference = weakref.ref(live)
        del live
        gc.collect()
        self.assertIsNone(reference())

    def test_unbinding_removes_icon_and_future_updates(self):
        manager = IconManager(get_palette(), image_factory=FakeImage)
        widget = FakeWidget()
        manager.bind(widget, "check")
        manager.unbind(widget)
        manager.apply(get_palette("light", "orange"), 20)
        self.assertIsNone(widget.options["image"])


if __name__ == "__main__":
    unittest.main()
