"""The dependency check reports missing ONNX native libraries before startup."""

from contextlib import redirect_stdout
from io import StringIO
import unittest
from unittest.mock import patch

from main import check_environment


class EnvironmentCheckTests(unittest.TestCase):
    def test_onnx_extension_is_imported_without_loading_a_model(self):
        output = StringIO()
        with patch("main.shutil.which", return_value="/usr/bin/ffmpeg"), \
             patch("main.importlib.util.find_spec", return_value=object()), \
             patch("main.importlib.import_module") as importer, redirect_stdout(output):
            self.assertEqual(check_environment(), 0)
        self.assertIn("sherpa_onnx", [call.args[0] for call in importer.call_args_list])
        self.assertNotIn("torch", [call.args[0] for call in importer.call_args_list])
        self.assertIn("OK      sherpa-onnx", output.getvalue())

    def test_onnx_native_library_failure_is_reported_as_missing_dependency(self):
        def import_module(name):
            if name == "sherpa_onnx":
                raise OSError("libonnxruntime.so cannot be opened")

        output = StringIO()
        with patch("main.shutil.which", return_value="/usr/bin/ffmpeg"), \
             patch("main.importlib.util.find_spec", return_value=object()), \
             patch("main.importlib.import_module", side_effect=import_module), redirect_stdout(output):
            self.assertEqual(check_environment(), 1)
        self.assertIn("MISSING sherpa-onnx", output.getvalue())
        self.assertIn("libonnxruntime.so", output.getvalue())


if __name__ == "__main__":
    unittest.main()
