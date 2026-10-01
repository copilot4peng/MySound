"""Export edited transcripts without silently reverting subtitle text.

SRT uses the original segment timings when the transcript is unchanged. After
editing, one nonempty line per segment retains those timings; any other edit is
exported as one cue over the complete recording span. This is intentionally
coarse because reliable word alignment requires another model pass.
"""

from __future__ import annotations

import math
from pathlib import Path


def _timestamp(seconds: float) -> str:
    millis = max(0, round(float(seconds) * 1000))
    hours, remainder = divmod(millis, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"


def _subtitle_segments(record: dict) -> list[dict]:
    text = str(record.get("text", "")).strip()
    if not text:
        return []
    segments = []
    for segment in record.get("segments") or []:
        start, end = float(segment["start"]), float(segment["end"])
        if not (math.isfinite(start) and math.isfinite(end)) or start < 0 or end <= start:
            raise ValueError("字幕片段时间无效，无法导出 SRT。")
        segments.append({"start": start, "end": end, "text": str(segment.get("text", "")).strip()})
    if not segments:
        raise ValueError("该记录没有时间戳，请导出为 TXT 或 Markdown。")

    original_text = "\n".join(segment["text"] for segment in segments)
    if text == original_text:
        return [segment for segment in segments if segment["text"]]

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) == len(segments):
        return [{**segment, "text": line} for segment, line in zip(segments, lines)]

    return [{
        "start": min(segment["start"] for segment in segments),
        "end": max(segment["end"] for segment in segments),
        "text": text,
    }]


def export_record(record: dict, path: str | Path) -> None:
    """Write UTF-8 TXT, Markdown or SRT, selecting format from the suffix."""
    path = Path(path).expanduser()
    suffix = path.suffix.lower()
    text = str(record.get("text", ""))
    if suffix == ".txt":
        content = text
    elif suffix == ".md":
        source = str(record.get("source", "转写记录")).replace("\n", " ")
        content = f"# {source}\n\n{text}"
    elif suffix == ".srt":
        cues = []
        for index, segment in enumerate(_subtitle_segments(record), start=1):
            cue_text = "\n".join(line for line in segment["text"].splitlines() if line.strip())
            cues.append(
                f"{index}\n{_timestamp(segment['start'])} --> {_timestamp(segment['end'])}\n{cue_text}"
            )
        content = "\n\n".join(cues)
    else:
        raise ValueError("仅支持 .txt、.md 和 .srt 格式。")
    path.write_text(content.rstrip() + "\n" if content else "", encoding="utf-8")
