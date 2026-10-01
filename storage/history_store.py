"""Transcription history stored independently from model weights."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import uuid

from .json_store import JsonStore


class HistoryStore:
    def __init__(self, data_dir=None):
        self._store = JsonStore(
            "history.json", [],
            lambda value: isinstance(value, list) and all(isinstance(item, dict) for item in value),
            data_dir,
        )
        self.path = self._store.path

    def list(self, query: str = "") -> list[dict]:
        records = self._store.read()
        needle = query.strip().casefold()
        if needle:
            records = [
                record for record in records
                if needle in " ".join(str(record.get(key, "")) for key in ("text", "source", "model")).casefold()
            ]
        return sorted(records, key=lambda record: record.get("created_at", ""), reverse=True)

    def get(self, record_id: str) -> dict | None:
        return next((record for record in self._store.read() if record.get("id") == record_id), None)

    def add(self, source: str, model: str, text: str, segments=None) -> dict:
        record = {
            "id": uuid.uuid4().hex,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="microseconds"),
            "source": str(source),
            "model": str(model),
            "text": str(text),
            "segments": deepcopy(segments) if segments is not None else [],
        }

        def apply(records):
            records.append(record)
            return record

        return self._store.modify(apply)

    def update(self, record_id: str, text: str, segments=None) -> dict:
        def apply(records):
            for record in records:
                if record.get("id") == record_id:
                    record["text"] = str(text)
                    if segments is not None:
                        record["segments"] = deepcopy(segments)
                    record["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="microseconds")
                    return record
            raise KeyError(f"历史记录不存在：{record_id}")

        return self._store.modify(apply)

    def delete(self, record_id: str) -> None:
        def apply(records):
            records[:] = [record for record in records if record.get("id") != record_id]

        self._store.modify(apply)
