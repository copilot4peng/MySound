import unittest

from ui.theme import COLORS, MODES, THEMES, get_palette


def contrast(first, second):
    def luminance(color):
        rgb = [int(color[index:index + 2], 16) / 255 for index in (1, 3, 5)]
        rgb = [part / 12.92 if part <= 0.04045 else ((part + 0.055) / 1.055) ** 2.4 for part in rgb]
        return sum(part * weight for part, weight in zip(rgb, (0.2126, 0.7152, 0.0722)))
    values = sorted((luminance(first), luminance(second)))
    return (values[1] + 0.05) / (values[0] + 0.05)


class PaletteTests(unittest.TestCase):
    def test_presets_have_distinct_surfaces_and_keep_saved_keys(self):
        self.assertEqual(COLORS, ("blue", "green", "purple", "orange"))
        for mode in MODES:
            for role in ("background", "surface", "border", "secondary", "selected"):
                self.assertEqual(len({get_palette(mode, key)[role] for key in COLORS}), 4)
        self.assertEqual(THEMES["blue"]["name"], "深海蓝")
        self.assertEqual(THEMES["green"]["name"], "森林绿")
        self.assertEqual(THEMES["purple"]["name"], "鸢尾紫")
        self.assertEqual(THEMES["orange"]["name"], "暖沙橙")

    def test_all_modes_keep_readable_text_and_accent_buttons(self):
        for mode in MODES:
            for key in COLORS:
                palette = get_palette(mode, key)
                for foreground, background in (
                    ("text", "background"), ("text", "surface"), ("text", "input"),
                    ("text", "selected"), ("muted", "surface"), ("on_accent", "accent"),
                ):
                    self.assertGreaterEqual(contrast(palette[foreground], palette[background]), 4.5, (mode, key, foreground, background))

    def test_preview_palette_does_not_mutate_another_preset(self):
        palette = get_palette("dark", "green")
        original = palette["surface"]
        palette["surface"] = "#000000"
        self.assertEqual(get_palette("dark", "green")["surface"], original)
        self.assertEqual(get_palette("invalid", "invalid"), get_palette("dark", "blue"))


if __name__ == "__main__":
    unittest.main()
