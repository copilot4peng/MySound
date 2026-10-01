"""Resolve local model selections and describe incomplete downloads, offline."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re


ARCHIVES = {".zip", ".7z", ".tar", ".gz"}
MODEL_TYPES = {"whisper", "whisper_onnx", "qwen3", "confucius"}
_WHISPER_ONNX = re.compile(r"^(?P<variant>.+)-(?P<part>encoder|decoder)(?P<int8>\.int8)?\.onnx$")


class ModelPathError(ValueError):
    """A path diagnostic with an English message for the desktop language UI."""

    def __init__(self, message: str, message_en: str):
        super().__init__(message)
        self.message_en = message_en


def _english_issues(issues: list[str]) -> str:
    replacements = {
        "（该文件尚未下载完成）": " (download is incomplete)",
        "分片权重索引为空或损坏": "The sharded-weight index is empty or invalid",
        "权重分片 ": "Weight shard ",
        "模型权重（model.safetensors / pytorch_model.bin 或分片索引）": "Model weights (model.safetensors / pytorch_model.bin or a shard index)",
        "分词器（tokenizer.json 或 vocab.json + merges.txt）": "Tokenizer (tokenizer.json or vocab.json + merges.txt)",
        "模型类型无法识别": "Unrecognized model type",
    }
    translated = []
    for issue in issues:
        for source, target in replacements.items():
            issue = issue.replace(source, target)
        if issue.startswith("无法读取模型配置") or issue.startswith("模型配置"):
            issue = "Invalid or unreadable model JSON configuration"
        translated.append(issue)
    return ", ".join(translated)


def infer_model_type(path: Path) -> str:
    directory = path if path.is_dir() else path.parent
    metadata = path.name.lower() + " " + directory.name.lower()
    for filename in ("config.json", "README.md"):
        try:
            with (directory / filename).open(encoding="utf-8") as stream:
                metadata += " " + stream.read(8192).lower()
        except (OSError, UnicodeError):
            pass
    if _WHISPER_ONNX.match(path.name) or "sherpa-onnx-whisper" in metadata:
        return "whisper_onnx"
    try:
        if any(_WHISPER_ONNX.match(item.name) for item in directory.iterdir()):
            return "whisper_onnx"
    except OSError:
        pass
    if "confucius" in metadata or "r2t2" in metadata:
        return "confucius"
    if "qwen3" in metadata or "qwen3_asr" in metadata:
        return "qwen3"
    if "whisper" in metadata or path.name.endswith((".pt", ".pt.incomplete")):
        return "whisper"
    return "unknown"


def resolve_whisper_onnx_files(raw_path: str | Path) -> dict:
    """Select matching encoder/decoder/tokens by names; never read ONNX bytes.

    Explicit ONNX files select their exact variant and precision. A directory
    prefers a complete INT8 pair and otherwise uses a complete FP32 pair.
    """
    selected = Path(raw_path).expanduser().resolve()
    explicit = _WHISPER_ONNX.match(selected.name) if selected.is_file() else None
    directory = selected.parent if selected.is_file() else selected
    try:
        names = {item.name for item in directory.iterdir() if item.is_file()}
    except OSError as error:
        raise ModelPathError(f"无法读取 ONNX 模型目录：{directory}", f"Cannot read the ONNX model folder: {directory}") from error
    if selected.is_file() and not explicit and not selected.name.endswith("-tokens.txt"):
        raise ModelPathError("请选择 Whisper ONNX encoder / decoder 文件，或完整模型目录。", "Select a Whisper ONNX encoder / decoder file or a complete model folder.")
    variants = sorted({match.group("variant") for name in names if (match := _WHISPER_ONNX.match(name))})
    if explicit:
        variants = [explicit.group("variant")]
    candidates, diagnostics = [], []
    for variant in variants:
        precisions = ["int8" if explicit.group("int8") else "fp32"] if explicit else ["int8", "fp32"]
        for precision in precisions:
            suffix = ".int8.onnx" if precision == "int8" else ".onnx"
            files = {"encoder": directory / (variant + "-encoder" + suffix),
                     "decoder": directory / (variant + "-decoder" + suffix),
                     "tokens": directory / (variant + "-tokens.txt")}
            missing = [path.name for path in files.values() if path.name not in names]
            if not missing:
                candidates.append(dict(files, precision=precision, variant=variant))
                break
            diagnostics.append(f"{variant} {precision.upper()}: " + ", ".join(missing))
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        variants = ", ".join(item["variant"] for item in candidates)
        raise ModelPathError(f"此目录有多个 Whisper ONNX 型号（{variants}），请选择具体 encoder 或 decoder 文件。", f"Multiple Whisper ONNX variants ({variants}) were found; select the exact encoder or decoder file.")
    detail = "; ".join(diagnostics) or "<variant>-encoder[.int8].onnx, <variant>-decoder[.int8].onnx, <variant>-tokens.txt"
    raise ModelPathError("Whisper ONNX 模型不完整；必须有同型号、同精度的 encoder / decoder 和 tokens。缺少：" + detail,
                         "Incomplete Whisper ONNX model; matching variant/precision encoder and decoder plus tokens are required. Missing: " + detail)


def _read_config(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeError) as exc:
        raise ValueError(f"无法读取模型配置 {path.name}：{exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"模型配置 {path.name} 不是有效的 JSON 对象。")
    return value


def directory_issues(path: Path) -> list[str]:
    """Check small metadata/files only; never deserialize model weights."""
    missing = []
    config = path / "config.json"
    if not config.is_file():
        suffix = "（该文件尚未下载完成）" if (path / "config.json.incomplete").exists() else ""
        missing.append("config.json" + suffix)
    else:
        try:
            _read_config(config)
        except ValueError as exc:
            missing.append(str(exc))
    weight_files = ("model.safetensors", "pytorch_model.bin")
    index_files = ("model.safetensors.index.json", "pytorch_model.bin.index.json")
    if not any((path / name).is_file() for name in weight_files):
        indices = [path / name for name in index_files if (path / name).is_file()]
        if indices:
            try:
                weight_map = _read_config(indices[0]).get("weight_map", {})
                if not isinstance(weight_map, dict) or not weight_map:
                    missing.append("分片权重索引为空或损坏")
                else:
                    for filename in sorted(set(weight_map.values())):
                        if not isinstance(filename, str) or not (path / filename).is_file():
                            missing.append(f"权重分片 {filename}")
            except (TypeError, ValueError) as exc:
                missing.append(str(exc))
        else:
            missing.append("模型权重（model.safetensors / pytorch_model.bin 或分片索引）")
    for filename in ("preprocessor_config.json", "tokenizer_config.json"):
        if not (path / filename).is_file():
            missing.append(filename)
    if not ((path / "tokenizer.json").is_file() or ((path / "vocab.json").is_file() and (path / "merges.txt").is_file())):
        missing.append("分词器（tokenizer.json 或 vocab.json + merges.txt）")
    return missing


def _entry(path: Path, model_type: str, ready: bool, status: str, status_en: str) -> dict:
    return {"name": path.name, "path": str(path.resolve()), "type": model_type, "ready": ready, "status": status, "status_en": status_en}


def scan_models(root: str | Path) -> list[dict]:
    """Find nested model directories/checkpoints and annotate incomplete ones."""
    root_path = Path(root).expanduser()
    if not root_path.is_dir():
        return []
    found = []
    for folder, directories, files in os.walk(root_path):
        path = Path(folder)
        directories[:] = [name for name in directories if name not in {".git", ".venv", "__pycache__"}]
        if len(path.relative_to(root_path).parts) >= 6:
            directories[:] = []
        markers = {"config.json", "config.json.incomplete", "model.safetensors", "model.safetensors.index.json", "pytorch_model.bin"}
        onnx_markers = any(_WHISPER_ONNX.match(filename) or filename.endswith("-tokens.txt") for filename in files)
        if onnx_markers:
            try:
                selected = resolve_whisper_onnx_files(path)
            except ModelPathError as error:
                found.append(_entry(path, "whisper_onnx", False, str(error), error.message_en))
            else:
                precision = selected["precision"].upper()
                found.append(_entry(path, "whisper_onnx", True, f"可用的 Whisper ONNX 模型（{precision}，CPU）", f"Ready Whisper ONNX model ({precision}, CPU)"))
        elif markers.intersection(files):
            model_type = infer_model_type(path)
            issues = directory_issues(path)
            if model_type == "unknown":
                issues.insert(0, "模型类型无法识别")
            found.append(_entry(path, model_type, not issues, "可用的本地模型目录" if not issues else "缺少或损坏：" + "、".join(issues[:4]), "Ready local model folder" if not issues else "Missing or invalid: " + _english_issues(issues[:4])))
        for filename in sorted(files):
            item = path / filename
            model_type = infer_model_type(item)
            if filename.endswith(".pt"):
                found.append(_entry(item, "whisper", True, "可用的 Whisper 权重", "Ready Whisper checkpoint"))
            elif item.suffix.lower() in ARCHIVES and model_type in MODEL_TYPES:
                found.append(_entry(item, model_type, False, "模型压缩包尚未解压；请先解压再选择模型目录", "Model archive: extract it first, then select its model folder"))
            elif filename.endswith(".pt.incomplete"):
                found.append(_entry(item, "whisper", False, "Whisper 权重尚未下载完成", "Whisper checkpoint download is incomplete"))
    return sorted(found, key=lambda item: (not item["ready"], item["name"].lower()))


def _compatible(path: Path, model_type: str) -> bool:
    detected = infer_model_type(path)
    if detected in {"unknown", model_type}:
        return True
    # The two families share the qwen3_asr architecture. A renamed Confucius
    # folder can have only that generic architecture marker in its config.
    return model_type == "confucius" and detected == "qwen3" and "qwen" not in path.name.lower()


def resolve_model_path(raw_path: str | Path, model_type: str) -> Path:
    """Accept exact models, a model's metadata/weight file, or one-model wrappers.

    No files are downloaded, modified, extracted or moved by this resolver.
    """
    if model_type not in MODEL_TYPES:
        raise ModelPathError(f"不支持的模型类型：{model_type}", f"Unsupported model type: {model_type}")
    if not str(raw_path).strip():
        if model_type == "whisper_onnx":
            raise ModelPathError("请选择 Whisper ONNX 模型目录或 encoder / decoder .onnx 文件。", "Select a Whisper ONNX model folder or encoder / decoder .onnx file.")
        raise ModelPathError("请选择本地模型目录或 Whisper .pt 文件。", "Select a local model folder or a Whisper .pt checkpoint.")
    path = Path(raw_path).expanduser().resolve()
    if not path.exists():
        error = FileNotFoundError(f"本地模型不存在：{path}")
        error.message_en = f"Local model path does not exist: {path}"
        raise error
    if path.name.endswith((".incomplete", ".part")):
        raise ModelPathError(f"模型文件尚未下载完整：{path.name}", f"Model download is incomplete: {path.name}")
    if path.suffix.lower() in ARCHIVES:
        raise ModelPathError(f"模型仍在压缩包中，请先解压 {path.name}，再选择解压后的模型目录。", f"Extract {path.name} first, then select the extracted model folder.")
    if path.is_file():
        if model_type == "whisper_onnx" and (path.suffix.lower() == ".onnx" or path.name.endswith("-tokens.txt")):
            resolve_whisper_onnx_files(path)
            return path if path.suffix.lower() == ".onnx" else path.parent
        if path.suffix.lower() == ".pt" and model_type == "whisper":
            return path
        if path.suffix.lower() in {".json", ".safetensors", ".bin"}:
            path = path.parent
        else:
            raise ModelPathError("请选择完整模型目录、其中的 config.json / 权重文件，或 Whisper .pt 文件。", "Select a model folder, its config.json / weight file, or a Whisper .pt checkpoint.")
    if model_type == "whisper_onnx":
        try:
            has_onnx_files = any(_WHISPER_ONNX.match(item.name) or item.name.endswith("-tokens.txt") for item in path.iterdir())
        except OSError:
            has_onnx_files = False
        if has_onnx_files:
            resolve_whisper_onnx_files(path)
            return path
    markers = ("config.json", "model.safetensors", "model.safetensors.index.json", "pytorch_model.bin", "config.json.incomplete")
    if any((path / name).exists() for name in markers):
        if not _compatible(path, model_type):
            raise ModelPathError(f"此目录识别为 {infer_model_type(path)}，与所选 {model_type} 不一致。请选择对应模型类型。", f"This folder contains {infer_model_type(path)}, but {model_type} is selected. Choose the matching model type.")
        issues = directory_issues(path)
        if issues:
            raise ModelPathError(f"模型目录不完整：{path}\n缺少或损坏：" + "、".join(issues), f"Incomplete model folder: {path}\nMissing or invalid: " + _english_issues(issues))
        return path
    candidates = [item for item in scan_models(path) if item["type"] == model_type]
    ready = [item for item in candidates if item["ready"]]
    if len(ready) == 1:
        return Path(ready[0]["path"])
    if len(ready) > 1:
        choices = "\n".join(item["path"] for item in ready[:5])
        if model_type == "whisper_onnx":
            raise ModelPathError(f"此目录下发现多个 Whisper ONNX 模型，请选择具体目录或 .onnx 文件：\n{choices}", f"Multiple Whisper ONNX models were found. Select an exact model folder or .onnx file:\n{choices}")
        raise ModelPathError(f"此目录下发现多个 {model_type} 模型，请选择具体模型目录或 .pt 文件：\n{choices}", f"Multiple {model_type} models were found. Select an exact model folder or .pt checkpoint:\n{choices}")
    if candidates:
        details = "\n".join(f"{item['path']}：{item['status']}" for item in candidates[:3])
        details_en = "\n".join(f"{item['path']}: {item['status_en']}" for item in candidates[:3])
        raise ModelPathError("未找到完整可用的模型：\n" + details, "No complete model was found:\n" + details_en)
    if model_type == "whisper_onnx":
        raise ModelPathError(f"此目录下未找到 Whisper ONNX 模型：{path}。需要同型号、同精度的 encoder / decoder .onnx 文件和对应 tokens.txt。", f"No Whisper ONNX model was found in {path}. A matching variant/precision encoder / decoder .onnx pair and corresponding tokens.txt are required.")
    raise ModelPathError(f"此目录下未找到 {model_type} 模型：{path}。请选择含配置、分词器和权重的目录，或 Whisper .pt 文件。", f"No {model_type} model was found in {path}. Select a folder containing configuration, tokenizer and weights, or a Whisper .pt checkpoint.")
