from __future__ import annotations

import atexit
import os
from pathlib import Path


class PidFile:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._pid = os.getpid()

    def __enter__(self) -> "PidFile":
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if self._path.exists():
            existing = self._read_pid()
            if existing is not None and _pid_is_running(existing):
                raise RuntimeError(
                    f"PID file exists and process appears running (pid={existing}): {self._path}"
                )
            self._path.unlink(missing_ok=True)

        self._path.write_text(f"{self._pid}\n", encoding="utf-8")
        atexit.register(self._cleanup)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._cleanup()

    def _cleanup(self) -> None:
        try:
            self._path.unlink()
        except FileNotFoundError:
            return
        except Exception:
            return

    def _read_pid(self) -> int | None:
        try:
            text = self._path.read_text(encoding="utf-8").strip()
            return int(text) if text else None
        except Exception:
            return None


def _pid_is_running(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    else:
        return True
