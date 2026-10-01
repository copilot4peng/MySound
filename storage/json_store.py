"""Small atomic JSON store shared by configuration and history."""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
import threading
from typing import Any, Callable, Iterator
import uuid

try:
    import fcntl
except ImportError:  # Thread safety still works outside Unix.
    fcntl = None


class StorageError(RuntimeError):
    """Invalid stored data was preserved for manual recovery."""


def default_data_dir() -> Path:
    """Use a writable user directory, including inside a frozen application."""
    override = os.environ.get("MYSOUND_DATA_DIR")
    if override:
        return Path(override).expanduser()
    return Path(os.environ.get("XDG_DATA_HOME", "~/.local/share")).expanduser() / "mysound"


_locks_guard = threading.Lock()
_locks: dict[str, threading.RLock] = {}


class JsonStore:
    def __init__(self, filename: str, default: Any, validator: Callable[[Any], bool], data_dir=None):
        self.data_dir = Path(data_dir).expanduser() if data_dir is not None else default_data_dir()
        self.path = self.data_dir / filename
        self._default = default
        self._validator = validator
        with _locks_guard:
            self._lock = _locks.setdefault(str(self.path.resolve()), threading.RLock())

    @contextmanager
    def _locked(self) -> Iterator[None]:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        with self._lock:
            with (self.data_dir / f".{self.path.name}.lock").open("a") as lock_file:
                if fcntl is not None:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    if fcntl is not None:
                        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def _read(self):
        if not self.path.exists():
            return deepcopy(self._default)
        try:
            with self.path.open("r", encoding="utf-8") as stream:
                data = json.load(stream)
            if not self._validator(data):
                raise ValueError("unexpected JSON structure")
            return data
        except (ValueError, UnicodeError) as exc:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            backup = self.path.with_name(f"{self.path.name}.corrupt-{timestamp}-{uuid.uuid4().hex[:6]}")
            self.path.replace(backup)
            raise StorageError(
                f"无法读取 {self.path.name}，原文件已保留到 {backup}。"
                "请检查备份内容后重试；重试将使用新的空数据。"
            ) from exc

    def _write(self, data):
        fd, temp_name = tempfile.mkstemp(prefix=f".{self.path.name}.", suffix=".tmp", dir=self.data_dir)
        temp_path = Path(temp_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp_path, self.path)
        finally:
            temp_path.unlink(missing_ok=True)

    def read(self):
        with self._locked():
            return self._read()

    def modify(self, operation: Callable[[Any], Any]):
        with self._locked():
            data = self._read()
            result = operation(data)
            self._write(data)
            return deepcopy(result)
