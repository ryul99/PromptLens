from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from promptlens.config import LoggingConfig


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class LogWriteResult:
    bytes_written: int
    rotated: bool
    path: Path


class JsonlLogger:
    def __init__(self, path: Path, *, max_file_bytes: int) -> None:
        self._path = path
        self._max_file_bytes = max_file_bytes
        self._lock = asyncio.Lock()

    @classmethod
    def from_config(cls, cfg: LoggingConfig) -> "JsonlLogger":
        log_dir = Path(os.path.expanduser(cfg.log_dir))
        log_dir.mkdir(parents=True, exist_ok=True)
        return cls(log_dir / cfg.filename, max_file_bytes=cfg.max_file_bytes)

    @property
    def path(self) -> Path:
        return self._path

    async def write_event(self, event: dict[str, Any]) -> LogWriteResult:
        event.setdefault("timestamp", _utc_now_iso())
        line = (
            json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n"
        ).encode("utf-8")

        async with self._lock:
            rotated = await asyncio.to_thread(
                self._rotate_if_needed, incoming_bytes=len(line)
            )
            bytes_written = await asyncio.to_thread(self._append_bytes, line)
            return LogWriteResult(
                bytes_written=bytes_written, rotated=rotated, path=self._path
            )

    def _rotate_if_needed(self, *, incoming_bytes: int) -> bool:
        try:
            current = self._path.stat().st_size
        except FileNotFoundError:
            current = 0

        if current + incoming_bytes <= self._max_file_bytes:
            return False

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        rotated_path = self._path.with_name(
            f"{self._path.stem}-{timestamp}{self._path.suffix}"
        )
        counter = 1
        while rotated_path.exists():
            rotated_path = self._path.with_name(
                f"{self._path.stem}-{timestamp}-{counter}{self._path.suffix}"
            )
            counter += 1

        if self._path.exists():
            self._path.replace(rotated_path)
        return True

    def _append_bytes(self, payload: bytes) -> int:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self._path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
        try:
            return os.write(fd, payload)
        finally:
            os.close(fd)


def truncate_bytes(data: bytes, max_bytes: int) -> tuple[bytes, bool]:
    if max_bytes <= 0:
        return b"", True
    if len(data) <= max_bytes:
        return data, False
    return data[:max_bytes], True


def safe_json_loads(payload: bytes) -> Optional[Any]:
    try:
        return json.loads(payload.decode("utf-8"))
    except Exception:
        return None
