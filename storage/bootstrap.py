"""Locate independent data directories without putting the pointer inside one.

The bootstrap only remembers the selected data directory. Selecting another
directory never copies or merges its configuration, history, or model weights.
The running application keeps its existing stores until the next launch.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import tempfile

from .json_store import JsonStore, default_data_dir


@dataclass(frozen=True)
class DataDirectorySelection:
    path: Path
    source: str
    overridden: bool


def validate_data_directory(path: str | Path) -> Path:
    """Create/probe a data directory without changing the startup selection."""
    if not str(path).strip():
        raise ValueError("请选择数据目录。")
    directory = Path(path).expanduser().resolve()
    if directory.exists() and not directory.is_dir():
        raise ValueError("数据目录必须是文件夹。")
    directory.mkdir(parents=True, exist_ok=True)
    if not os.access(directory, os.R_OK | os.W_OK | os.X_OK):
        raise PermissionError(f"数据目录不可读写：{directory}")
    # Probe actual operations so ACL/read-only filesystem errors appear in
    # settings, before replacing the last usable directory pointer.
    with os.scandir(directory) as entries:
        next(entries, None)
    with tempfile.TemporaryFile(prefix=".mysound-access-", dir=directory) as probe:
        probe.write(b"mysound")
        probe.flush()
    return directory


class BootstrapStore:
    validate_data_dir = staticmethod(validate_data_directory)

    def __init__(self, config_dir=None):
        directory = (
            Path(config_dir).expanduser()
            if config_dir is not None
            else Path(os.environ.get("XDG_CONFIG_HOME") or "~/.config").expanduser() / "mysound"
        )
        self._store = JsonStore(
            "bootstrap.json", {},
            lambda value: isinstance(value, dict)
            and ("data_dir" not in value or isinstance(value["data_dir"], str)),
            directory,
        )
        self.path = self._store.path

    def load(self) -> dict:
        value = self._store.read()
        return {"data_dir": value["data_dir"]} if value.get("data_dir") else {}

    def select_data_dir(self, path: str | Path) -> Path:
        """Validate a writable directory and persist its pointer for next launch."""
        directory = self.validate_data_dir(path)

        def select(data):
            data.clear()
            data["data_dir"] = str(directory)

        self._store.modify(select)
        return directory


def resolve_data_directory(cli_data_dir=None, bootstrap: BootstrapStore | None = None) -> DataDirectorySelection:
    """Preserve explicit launch overrides ahead of the saved GUI selection."""
    if cli_data_dir:
        return DataDirectorySelection(Path(cli_data_dir).expanduser().resolve(), "cli", True)
    environment_path = os.environ.get("MYSOUND_DATA_DIR")
    if environment_path:
        return DataDirectorySelection(Path(environment_path).expanduser().resolve(), "env", True)
    saved = (bootstrap if bootstrap is not None else BootstrapStore()).load().get("data_dir")
    if saved:
        return DataDirectorySelection(Path(saved).expanduser().resolve(), "bootstrap", False)
    return DataDirectorySelection(default_data_dir().resolve(), "default", False)
