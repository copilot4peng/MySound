#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$project_dir"

if [[ -n "${PYTHON:-}" ]]; then
    python_bin="$PYTHON"
elif [[ -x "$project_dir/.venv/bin/python" ]]; then
    python_bin="$project_dir/.venv/bin/python"
else
    python_bin="python3"
fi

if [[ "$(uname -s)" != "Linux" ]]; then
    printf '%s\n' '请在 Ubuntu/Linux 上构建 Linux 发行目录。' >&2
    exit 1
fi

"$python_bin" - <<'PY'
import importlib.util
import sys

required = {
    "tkinter": "python3-tk（系统包）",
    "PyInstaller": "pyinstaller",
    "customtkinter": "customtkinter",
    "tkinterdnd2": "tkinterdnd2",
    "numpy": "numpy",
    "sounddevice": "sounddevice",
    "soundfile": "soundfile",
    "torch": "torch",
    "torchaudio": "torchaudio",
    "silero_vad": "silero-vad",
    "whisper": "openai-whisper",
    "sherpa_onnx": "sherpa-onnx",
    "qwen_asr": "qwen-asr",
    "transformers": "transformers",
    "accelerate": "accelerate",
    "librosa": "librosa",
}
missing = [package for module, package in required.items() if importlib.util.find_spec(module) is None]
if missing:
    print("构建环境缺少：" + "、".join(missing), file=sys.stderr)
    print("请先安装系统依赖，并执行 python -m pip install -r requirements-dev.txt。", file=sys.stderr)
    sys.exit(1)
import tkinter  # Catch a missing _tkinter extension without requiring a display.
print("构建 Python：" + sys.executable)
PY

"$python_bin" -m PyInstaller \
    --noconfirm \
    --clean \
    --distpath "$project_dir/dist" \
    --workpath "$project_dir/build/pyinstaller" \
    "$project_dir/packaging/mysound.spec"

printf '%s\n' \
    "构建完成：$project_dir/dist/ubuntu_asr_app/" \
    '运行：dist/ubuntu_asr_app/ubuntu_asr_app' \
    '分发时请复制整个目录并保留符号链接；ASR 模型文件保持在目录外。' \
    '目标系统需要 FFmpeg 和可用的音频设备/PortAudio。'
