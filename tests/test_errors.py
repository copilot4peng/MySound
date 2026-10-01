import unittest

from ui.errors import concise_error
from ui.i18n import Translator


class ErrorMessageTests(unittest.TestCase):
    def test_cuda_diagnostic_is_replaced_with_actionable_message(self):
        message = concise_error("CUDA out of memory. Tried to allocate 20 MiB. " * 80)
        self.assertIn("CPU", message)
        self.assertLess(len(message), 200)
        self.assertNotIn("Tried to allocate", message)

    def test_english_and_other_long_errors_stay_readable(self):
        tr = Translator("en_US").tr
        self.assertIn("CPU", concise_error("CUDA out of memory", tr))
        message = concise_error("backend detail " * 100, tr)
        self.assertLess(len(message), 500)
        self.assertIn("log", message)
        self.assertEqual(concise_error("File does not exist"), "File does not exist")
