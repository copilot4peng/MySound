#!/usr/bin/env python3
"""MySound entry point. Heavy ASR modules load only when a job starts."""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import logging
from logging.handlers import RotatingFileHandler
import os
import shutil
import sys

# All supported model loaders operate exclusively on local files.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def check_environment():
    required = {"tkinter": "python3-tk (系统包)", "customtkinter": "customtkinter", "tkinterdnd2": "tkinterdnd2", "numpy": "numpy", "soundfile": "soundfile", "sounddevice": "sounddevice", "torch": "torch", "silero_vad": "silero-vad", "transformers": "transformers", "qwen_asr": "qwen-asr", "whisper": "openai-whisper", "sherpa_onnx": "sherpa-onnx"}
    missing = []
    print(f"Python: {sys.version.split()[0]}")
    for module, package in required.items():
        found = importlib.util.find_spec(module) is not None
        detail = ""
        if found and module in {"tkinter", "sounddevice", "soundfile", "sherpa_onnx"}:
            try:
                importlib.import_module(module)
            except (ImportError, OSError) as exc:
                found = False
                detail = f" ({exc})"
        print(f"{'OK     ' if found else 'MISSING'} {package}{detail}")
        if not found:
            missing.append(package)
    ffmpeg = shutil.which("ffmpeg")
    print(f"{'OK     ' if ffmpeg else 'MISSING'} ffmpeg{': ' + ffmpeg if ffmpeg else ''}")
    print("离线模式：已启用；完整验证仍需实际加载模型和录音设备。")
    return 1 if missing or not ffmpeg else 0


def main():
    parser = argparse.ArgumentParser(description="MySound · 本地语音转文字桌面应用")
    parser.add_argument("--data-dir", help="config.json / history.json / 日志保存目录")
    parser.add_argument("--check", action="store_true", help="检查运行依赖，不启动界面")
    parser.add_argument("--smoke-test", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.check:
        return check_environment()
    try:
        from storage import BootstrapStore, resolve_data_directory
        bootstrap = BootstrapStore()
        selection = resolve_data_directory(args.data_dir, bootstrap)
        selection.path.mkdir(parents=True, exist_ok=True)
        handler = RotatingFileHandler(selection.path / "mysound.log", maxBytes=2_000_000, backupCount=2, encoding="utf-8")
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s", handlers=[handler, logging.StreamHandler()])
        from ui.app import MySoundApp
        app = MySoundApp(data_dir=selection.path, data_selection=selection, bootstrap=bootstrap)
        if args.smoke_test:
            app.after(1000, app.destroy)
        app.mainloop()
    except ImportError as exc:
        print(f"缺少运行依赖：{exc}\n请按 README 安装系统依赖及 requirements.txt，再运行 python main.py --check。", file=sys.stderr)
        return 1
    except Exception:
        logging.exception("启动失败")
        return 1
    return 0


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    raise SystemExit(main())
