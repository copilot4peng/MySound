"""Hardware inventory stays read-only and treats missing readings as unknown."""

import os
import subprocess
import unittest
from unittest.mock import patch

from core import hardware_info as hardware


class HardwareTests(unittest.TestCase):
    def test_nvidia_memory_is_mib_and_unknown_free_memory_is_not_zero(self):
        devices = hardware._nvidia_gpus('0, GeForce RTX 2060, GPU-a, 6144, 88\n1, Other GPU, GPU-b, 24576, N/A\n')
        self.assertEqual(devices[0]["total_vram_gib"], 6)
        self.assertAlmostEqual(devices[0]["free_vram_gib"], 0.09)
        self.assertIsNone(devices[1]["free_vram_gib"])

    def test_device_visibility_honors_indices_and_uuid(self):
        text = '0, First, GPU-first, 6144, 4096\n1, Second, GPU-second, 12288, 10240\n'
        self.assertEqual([gpu["visible_to_process"] for gpu in hardware._nvidia_gpus(text, "1")], [False, True])
        self.assertEqual([gpu["visible_to_process"] for gpu in hardware._nvidia_gpus(text, "GPU-first")], [True, False])
        self.assertFalse(any(gpu["visible_to_process"] for gpu in hardware._nvidia_gpus(text, "")))
        self.assertEqual([gpu["cuda_ordinal"] for gpu in hardware._nvidia_gpus(text, "1,0")], [1, 0])
        self.assertEqual([gpu["cuda_ordinal"] for gpu in hardware._nvidia_gpus(text, "GPU-second")], [None, 0])
        self.assertTrue(all(gpu["cuda_order_verified"] for gpu in hardware._nvidia_gpus(text, "GPU-second,GPU-first")))
        self.assertFalse(any(gpu["cuda_order_verified"] for gpu in hardware._nvidia_gpus(text)))

    def test_memavailable_is_distinct_from_total_or_memfree(self):
        total, available = hardware._memory('MemTotal: 8388608 kB\nMemAvailable: 2097152 kB\nMemFree: 1024 kB\n')
        self.assertEqual((total, available), (8, 2))
        self.assertEqual(hardware._memory('MemTotal: 8388608 kB\nMemFree: 1024 kB\n'), (8, None))

    def test_query_timeout_is_bounded_and_returns_unknown(self):
        with patch.object(hardware.subprocess, "run", side_effect=subprocess.TimeoutExpired("nvidia-smi", 2)) as run:
            self.assertIsNone(hardware._command(["nvidia-smi"]))
        self.assertEqual(run.call_args.kwargs["timeout"], 2)
        self.assertNotIn("shell", run.call_args.kwargs)

    def test_driver_failure_preserves_pci_identity_without_fake_vram(self):
        texts = {"/proc/cpuinfo": "model name : Test CPU\n", "/proc/meminfo": "MemTotal: 8388608 kB\nMemAvailable: 2097152 kB\n", "/etc/os-release": 'PRETTY_NAME="Ubuntu Test"\n'}
        pci = "01:00.0 VGA compatible controller: NVIDIA Corporation RTX Test\n02:00.0 Display controller: Advanced Micro Devices AMD Test\n"
        with patch.object(hardware, "_read_text", side_effect=lambda path: texts.get(path, "")), patch.object(hardware, "_command", side_effect=[None, pci]), patch.dict(os.environ, {}, clear=True):
            result = hardware.detect_hardware()
        self.assertEqual(result["cpu_model"], "Test CPU")
        self.assertEqual(result["system"], "Ubuntu Test")
        self.assertIsNone(result["gpus"][0]["free_vram_gib"])
        self.assertFalse(result["gpus"][1]["cuda"])
        self.assertIn("nvidia_query_unavailable", result["warnings"])
        self.assertIn("non_cuda_gpu", result["warnings"])


if __name__ == "__main__":
    unittest.main()
