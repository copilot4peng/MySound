"""Persistent user preferences."""

from __future__ import annotations

from copy import deepcopy

from .json_store import JsonStore


DEFAULT_CONFIG = {
    "model_root": "~/asr_models",
    "model_type": "qwen3",
    "model_path": "",
    "model_paths": {"whisper": "", "whisper_onnx": "", "qwen3": "", "confucius": ""},
    "appearance_mode": "dark",
    "color_theme": "blue",
    "ui_language": "zh_CN",
    "font_size": 14,
    "work_dir": "",
    "window_layout": {
        "width": 1180,
        "height": 780,
        "maximized": False,
        "sidebar_ratio": 0.23,
        "editor_ratio": 0.40,
    },
    "device": "auto",
    "language": None,
    "mic_device": None,
    "vad_target_seconds": 30,
    "vad_max_seconds": 60,
    "mic_block_seconds": 2.0,
}


def normalize_font_size(value) -> int:
    """Keep saved base text sizes within the supported, readable range."""
    try:
        size = int(value) if not isinstance(value, bool) else 14
    except (TypeError, ValueError, OverflowError):
        size = 14
    return max(12, min(48, size))


def _normalize_presentation(config: dict) -> None:
    if config.get("ui_language") not in ("zh_CN", "en_US"):
        config["ui_language"] = "zh_CN"
    config["font_size"] = normalize_font_size(config.get("font_size", 14))


def _merge(base: dict, values: dict) -> dict:
    """Merge nested preferences without sharing mutable defaults."""
    result = deepcopy(base)
    for key, value in values.items():
        if isinstance(result.get(key), dict) and isinstance(value, dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _with_defaults(data: dict) -> dict:
    result = _merge(DEFAULT_CONFIG, data)
    # Older installations have only model_type/model_path. A path in the new
    # mapping (including an explicitly empty path) takes precedence on load.
    model_type = result["model_type"]
    saved_paths = data.get("model_paths")
    if not isinstance(saved_paths, dict):
        saved_paths = {}
        result["model_paths"] = deepcopy(DEFAULT_CONFIG["model_paths"])
    if model_type not in saved_paths and data.get("model_path"):
        result["model_paths"][model_type] = data["model_path"]
    result["model_path"] = result["model_paths"].get(model_type, "")
    if not isinstance(result["window_layout"], dict):
        result["window_layout"] = deepcopy(DEFAULT_CONFIG["window_layout"])
    _normalize_presentation(result)
    return result


class ConfigStore:
    def __init__(self, data_dir=None):
        self._store = JsonStore("config.json", {}, lambda value: isinstance(value, dict), data_dir)
        self.path = self._store.path

    def load(self) -> dict:
        return _with_defaults(self._store.read())

    def update(self, **kwargs) -> dict:
        def apply(data):
            merged = _merge(_with_defaults(data), kwargs)
            if not isinstance(merged["model_paths"], dict):
                raise ValueError("model_paths 必须为模型类型到路径的映射。")
            if not isinstance(merged["window_layout"], dict):
                raise ValueError("window_layout 必须为窗口布局设置。")
            active_type = merged["model_type"]
            if "model_path" in kwargs:
                merged["model_paths"][active_type] = kwargs["model_path"]
            merged["model_path"] = merged["model_paths"].get(active_type, "")
            _normalize_presentation(merged)
            data.clear()
            data.update(merged)
            return data

        return self._store.modify(apply)
