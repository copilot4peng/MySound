import unittest
from unittest.mock import Mock, call, patch

from ui.dpi import _read_x_resources, configure_tk_dpi, parse_xft_dpi


class ParseDpiTests(unittest.TestCase):
    def test_reads_only_exact_dpi_resource(self):
        resources = b"Xft.antialias:\t1\nXft.dpi:\t192\nXft.hinting:\t1\n"
        self.assertEqual(parse_xft_dpi(resources), 192)
        self.assertEqual(parse_xft_dpi("  Xft.dpi : 143.5\n"), 143.5)
        self.assertIsNone(parse_xft_dpi("Other.Xft.dpi: 192\nXft.dpiScale: 192\n! Xft.dpi: 192"))

    def test_invalid_or_missing_dpi_preserves_native_default(self):
        for resource in (None, b"", "Xft.dpi: nan", "Xft.dpi: inf", "Xft.dpi: -96",
                         "Xft.dpi: 47", "Xft.dpi: 769", "Xft.dpi: not-a-number"):
            with self.subTest(resource=resource):
                self.assertIsNone(parse_xft_dpi(resource))


class ConfigureDpiTests(unittest.TestCase):
    def setUp(self):
        self.root = Mock()
        self.root.tk.call.return_value = "x11"
        self.platform = patch("ui.dpi.sys.platform", "linux")
        self.platform.start()
        self.addCleanup(self.platform.stop)

    def test_reads_ctypes_resource_and_configures_tk_locally(self):
        x11 = Mock()
        x11.XOpenDisplay.return_value = 123
        x11.XResourceManagerString.return_value = b"Xft.dpi:\t192\n"
        with patch("ui.dpi.ctypes.util.find_library", return_value="libX11.so.6"), \
             patch("ui.dpi.ctypes.CDLL", return_value=x11):
            self.assertEqual(configure_tk_dpi(self.root), 192)
        self.assertEqual(self.root.tk.call.call_args_list, [
            call("tk", "windowingsystem"), call("tk", "scaling", 192 / 72),
        ])
        x11.XOpenDisplay.assert_called_once_with(None)
        x11.XResourceManagerString.assert_called_once_with(123)
        x11.XCloseDisplay.assert_called_once_with(123)

    def test_missing_resource_does_not_change_scaling(self):
        with patch("ui.dpi._read_x_resources", return_value=b"Xft.antialias: 1"):
            self.assertIsNone(configure_tk_dpi(self.root))
        self.root.tk.call.assert_called_once_with("tk", "windowingsystem")

    def test_display_is_closed_on_resource_failure(self):
        x11 = Mock()
        x11.XOpenDisplay.return_value = 456
        x11.XResourceManagerString.side_effect = RuntimeError("resource unavailable")
        with patch("ui.dpi.ctypes.util.find_library", return_value="libX11.so.6"), \
             patch("ui.dpi.ctypes.CDLL", return_value=x11):
            self.assertIsNone(configure_tk_dpi(self.root))
        x11.XCloseDisplay.assert_called_once_with(456)
        self.root.tk.call.assert_called_once_with("tk", "windowingsystem")

    def test_unavailable_x11_and_non_x11_are_noops(self):
        with patch("ui.dpi.ctypes.util.find_library", return_value=None):
            self.assertIsNone(_read_x_resources())
        with patch("ui.dpi._read_x_resources") as read:
            self.root.tk.call.return_value = "wayland"
            self.assertIsNone(configure_tk_dpi(self.root))
            read.assert_not_called()
        with patch("ui.dpi.sys.platform", "darwin"), patch("ui.dpi._read_x_resources") as read:
            self.root.tk.call.reset_mock()
            self.assertIsNone(configure_tk_dpi(self.root))
            self.root.tk.call.assert_not_called()
            read.assert_not_called()


if __name__ == "__main__":
    unittest.main()
