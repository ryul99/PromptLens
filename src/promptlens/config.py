from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

from pydantic import BaseModel, Field, field_validator


class UpstreamConfig(BaseModel):
    base_url: str = "http://127.0.0.1:4000"
    timeout_s: float = 60.0
    verify_ssl: bool = True
    headers: dict[str, str] = Field(default_factory=dict)

    @field_validator("base_url")
    @classmethod
    def _validate_base_url(cls, value: str) -> str:
        parsed = urlparse(value)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError(
                "base_url must be a full http(s) URL, e.g. http://127.0.0.1:4000"
            )
        return value.rstrip("/")


class LoggingConfig(BaseModel):
    log_dir: str = str(Path("~/.promptlens/logs").expanduser())
    filename: str = "promptlens.jsonl"
    max_file_bytes: int = 50 * 1024 * 1024
    max_prompt_bytes: int = 256 * 1024


class ServerConfig(BaseModel):
    log_level: str = "info"


class AppConfig(BaseModel):
    upstream: UpstreamConfig = Field(default_factory=UpstreamConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)


def _load_from_path(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    raw = path.read_bytes()

    if suffix == ".toml":
        return tomllib.loads(raw.decode("utf-8"))

    raise ValueError(f"Unsupported config type: {suffix} (supported: .toml)")


def load_config(path: Optional[Path]) -> AppConfig:
    if path is None:
        return AppConfig()
    data = _load_from_path(path)
    return AppConfig.model_validate(data)
