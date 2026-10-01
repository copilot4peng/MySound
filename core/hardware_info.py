"""Read a lightweight hardware snapshot without importing ML runtimes."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import io
import math
import os
from pathlib import Path
import platform
import subprocess


GIB = 1024 ** 3


def _read_text(path) -> str:
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _command(arguments, timeout=2.0):
    try:
        result = subprocess.run(arguments, capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout if result.returncode == 0 else None


def _number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number >= 0 else None


def _memory(text):
    values = {}
    for line in text.splitlines():
        name, separator, value = line.partition(":")
        if separator and value.split():
            kib = _number(value.split()[0])
            if kib is not None:
                values[name] = round(kib / (1024 ** 2), 2)
    # MemFree excludes reclaimable filesystem cache. MemAvailable is the
    # relevant Linux allocation headroom; never substitute the total capacity.
    return values.get("MemTotal"), values.get("MemAvailable")


def _cpu_name(text):
    for key in ("model name", "Hardware", "Processor"):
        for line in text.splitlines():
            label, separator, value = line.partition(":")
            if separator and label.strip() == key and value.strip():
                return value.strip()
    return platform.processor() or None


def _nvidia_gpus(text, visible_devices=None):
    result = []
    visibility = None if visible_devices is None else [part.strip() for part in visible_devices.split(",") if part.strip()]
    for row in csv.reader(io.StringIO(text or "")):
        if len(row) != 5:
            continue
        index, name, uuid, total, free = [part.strip() for part in row]
        total_mib, free_mib = _number(total), _number(free)
        ordinal = len(result) if visibility is None else next((position for position, item in enumerate(visibility) if item != "-1" and (item == index or uuid.startswith(item))), None)
        visible = ordinal is not None
        result.append({
            "index": index, "name": name, "uuid": uuid, "vendor": "NVIDIA",
            "total_vram_gib": round(total_mib / 1024, 2) if total_mib is not None else None,
            "free_vram_gib": round(free_mib / 1024, 2) if free_mib is not None else None,
            "cuda": True, "visible_to_process": visible, "cuda_ordinal": ordinal,
            "driver_reported": True,
        })
    # CUDA's default FASTEST_FIRST order need not match nvidia-smi's physical
    # indices. UUID selection is unambiguous without initializing CUDA.
    order_verified = len(result) <= 1 or (visibility is not None and all(item.startswith("GPU-") for item in visibility if item != "-1"))
    for gpu in result:
        gpu["cuda_order_verified"] = order_verified
    return result


def _pci_gpus(text, nvidia_already_reported=False):
    result = []
    for line in (text or "").splitlines():
        if not any(kind in line for kind in ("VGA compatible controller:", "3D controller:", "Display controller:")):
            continue
        name = line.split(": ", 1)[-1].strip()
        lowered = name.lower()
        vendor = "NVIDIA" if "nvidia" in lowered else "AMD" if "amd" in lowered or "advanced micro devices" in lowered else "Intel" if "intel" in lowered else "Other"
        if vendor == "NVIDIA" and nvidia_already_reported:
            continue
        result.append({"index": None, "name": name, "uuid": None, "vendor": vendor,
                       "total_vram_gib": None, "free_vram_gib": None, "cuda": vendor == "NVIDIA",
                       "visible_to_process": None, "cuda_ordinal": None, "driver_reported": False})
    return result


def detect_hardware() -> dict:
    """Return JSON-compatible inventory. None means unknown, never zero bytes.

    CUDA identifies an NVIDIA device family, not a successful PyTorch runtime
    check. nvidia-smi is queried with a timeout; no model/GPU allocations occur.
    """
    total, available = _memory(_read_text("/proc/meminfo"))
    cpu = _cpu_name(_read_text("/proc/cpuinfo"))
    system_name = platform.system()
    os_release = {}
    for line in _read_text("/etc/os-release").splitlines():
        key, separator, value = line.partition("=")
        if separator:
            os_release[key] = value.strip().strip('"')
    gpu_output = _command(["nvidia-smi", "--query-gpu=index,name,uuid,memory.total,memory.free", "--format=csv,noheader,nounits"])
    gpus = _nvidia_gpus(gpu_output, os.environ.get("CUDA_VISIBLE_DEVICES"))
    pci_output = _command(["lspci"])
    gpus.extend(_pci_gpus(pci_output, bool(gpus)))
    if os.environ.get("CUDA_VISIBLE_DEVICES") in {"", "-1"}:
        for gpu in gpus:
            if gpu["cuda"]:
                gpu["visible_to_process"] = False
                gpu["cuda_ordinal"] = None
    warnings = []
    if total is None:
        warnings.append("ram_total_unknown")
    if available is None:
        warnings.append("ram_available_unknown")
    if gpu_output is None:
        warnings.append("nvidia_query_unavailable")
    if any(gpu["vendor"] != "NVIDIA" for gpu in gpus):
        warnings.append("non_cuda_gpu")
    if any(gpu["visible_to_process"] is False for gpu in gpus):
        warnings.append("cuda_hidden")
    if any(gpu["cuda"] for gpu in gpus):
        warnings.append("cuda_runtime_unverified")
    if any(gpu.get("cuda_order_verified") is False for gpu in gpus):
        warnings.append("cuda_order_unverified")
    return {
        "cpu_model": cpu, "logical_cores": os.cpu_count(),
        "ram_total_gib": total, "ram_available_gib": available,
        "system": os_release.get("PRETTY_NAME") or f"{system_name} {platform.release()}",
        "os": system_name, "architecture": platform.machine(),
        "gpus": gpus, "gpu_query_ok": gpu_output is not None,
        "warnings": warnings, "captured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
