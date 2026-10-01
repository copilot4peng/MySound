import unittest
from unittest.mock import Mock, patch

from ui.layout import clamp_layout, clamp_sash, ellipsize
from ui.theme import ThemeManager


class LayoutTests(unittest.TestCase):
    def test_restore_clamps_to_current_monitor(self):
        restored = clamp_layout({"width": 3000, "height": 1800, "maximized": True}, 1366, 768)
        self.assertEqual((restored["width"], restored["height"]), (1286, 688))
        self.assertTrue(restored["maximized"])
        small = clamp_layout({"width": 10, "height": 10}, 800, 600)
        self.assertEqual((small["width"], small["height"]), (720, 520))

    def test_defaults_and_invalid_persisted_values(self):
        self.assertEqual(clamp_layout(None, 1920, 1080), {
            "width": 1180, "height": 780, "maximized": False,
            "sidebar_ratio": 0.23, "editor_ratio": 0.40,
        })
        result = clamp_layout({"width": "bad", "height": None, "sidebar_ratio": float("nan"),
                               "editor_ratio": 99, "maximized": "false"}, 1920, 1080)
        self.assertEqual(result["width"], 1180)
        self.assertEqual(result["height"], 780)
        self.assertEqual(result["sidebar_ratio"], 0.23)
        self.assertEqual(result["editor_ratio"], 1.0)
        self.assertFalse(result["maximized"])

    def test_restore_preserves_narrow_panes_on_large_displays(self):
        result = clamp_layout({"sidebar_ratio": 0.08, "editor_ratio": 0.91}, 3840, 2160)
        self.assertEqual(result["sidebar_ratio"], 0.08)
        self.assertEqual(result["editor_ratio"], 0.91)
        negative = clamp_layout({"sidebar_ratio": -0.2, "editor_ratio": -1}, 1920, 1080)
        self.assertEqual(negative["sidebar_ratio"], 0.0)
        self.assertEqual(negative["editor_ratio"], 0.0)

    def test_sash_respects_both_minimum_sizes_including_fractional_scaling(self):
        self.assertEqual(clamp_sash(1000, 0.01, 220, 600), 220)
        self.assertEqual(clamp_sash(1000, 0.99, 220, 600), 392)
        self.assertEqual(clamp_sash(1250, 0.23, 275, 750, sash_width=10), 288)
        self.assertEqual(clamp_sash(500, 0.9, 200, 600), 123)
        self.assertEqual(clamp_sash(0, 0.5, 220, 600), 0)

    def test_ellipsis_uses_measured_width_not_character_count(self):
        def measure(text):
            return sum(12 if ord(char) > 127 else 6 for char in text)

        self.assertEqual(ellipsize("中文文件名", 36, measure), "中文…")
        self.assertEqual(ellipsize("abcde", 36, measure), "abcde")
        self.assertEqual(ellipsize("abcdefg", 36, measure), "abcd…")
        self.assertEqual(ellipsize("中文", 11, measure), "")


class ThemeTests(unittest.TestCase):
    def setUp(self):
        self.ctk = Mock()
        self.modules = patch.dict("sys.modules", {"customtkinter": self.ctk})
        self.modules.start()
        self.addCleanup(self.modules.stop)

    def test_switch_updates_dropdown_panes_and_preserves_overrides(self):
        manager = ThemeManager()
        option, pane, title = Mock(), Mock(), Mock()
        self.assertIs(manager.style(option, "option"), option)
        manager.style(pane, "paned")
        manager.style(title, "title", text_color="#123456")
        manager.apply("light", "green")
        self.ctk.set_appearance_mode.assert_called_with("light")
        options = option.configure.call_args.kwargs
        self.assertEqual(options["dropdown_fg_color"], manager.palette["surface"])
        self.assertEqual(options["dropdown_text_color"], manager.palette["text"])
        self.assertEqual(pane.configure.call_args.kwargs, {"bg": manager.palette["border"]})
        self.assertEqual(title.configure.call_args.kwargs, {"text_color": "#123456"})

    def test_reregister_changes_role_and_destroyed_widgets_are_removed(self):
        manager = ThemeManager()
        alive, dead = Mock(), Mock()
        manager.style(alive, "history")
        manager.style(alive, "selected_history")
        manager.style(dead, "button")
        dead.winfo_exists.return_value = False
        dead.configure.reset_mock()
        manager.apply("light", "purple")
        self.assertEqual(alive.configure.call_args.kwargs["fg_color"], manager.palette["selected"])
        dead.configure.assert_not_called()
        self.assertNotIn(dead, manager._widgets)


if __name__ == "__main__":
    unittest.main()
