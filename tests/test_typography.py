import unittest

from ui.typography import FONT_SIZES, FontManager, Typography, normalize_font_size


class FakeFont:
    def __init__(self, **options):
        self.options = options

    def configure(self, **options):
        self.options.update(options)


class TypographyTests(unittest.TestCase):
    def test_shared_fonts_update_in_place_and_keep_hierarchy(self):
        typography = Typography(14, font_factory=FakeFont)
        fonts = {role: typography.font(role) for role in ("body", "title", "brand", "section", "editor", "small")}
        self.assertIs(typography.font(), fonts["body"])
        self.assertIs(FontManager, Typography)
        typography.apply(20)
        expected_sizes = {"body": 20, "title": 32, "brand": 33, "section": 23, "editor": 24, "small": 16}
        for role, size in expected_sizes.items():
            self.assertIs(typography.font(role), fonts[role])
            self.assertEqual(fonts[role].options["size"], size)
        self.assertEqual(fonts["title"].options["weight"], "bold")
        self.assertEqual(fonts["body"].options["weight"], "normal")

    def test_font_created_after_resize_uses_new_size(self):
        typography = Typography(font_factory=FakeFont)
        typography.apply(48)
        self.assertEqual(typography.font("editor").options["size"], 52)
        self.assertEqual(typography.font("small").options["size"], 38)

    def test_small_text_stays_readable_at_minimum_base_size(self):
        typography = Typography(12, font_factory=FakeFont)
        self.assertEqual(typography.font("small").options["size"], 12)
        self.assertEqual(typography.apply(3), 12)
        self.assertEqual(typography.apply(99), 48)

    def test_invalid_values_fall_back_to_default_and_numbers_are_clamped(self):
        for invalid in (None, "invalid", float("nan"), float("inf"), True):
            self.assertEqual(normalize_font_size(invalid), 14)
        self.assertEqual(normalize_font_size("18"), 18)
        self.assertEqual(normalize_font_size(15), 15)
        self.assertEqual(normalize_font_size(2), 12)
        self.assertEqual(normalize_font_size(70), 48)
        self.assertEqual(FONT_SIZES, (12, 14, 16, 18, 20, 22, 24, 28, 32, 36, 40, 48))


if __name__ == "__main__":
    unittest.main()
