from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated, Optional

import typer
import uvicorn

from promptlens.config import AppConfig, load_config
from promptlens.logging_jsonl import JsonlLogger
from promptlens.pidfile import PidFile
from promptlens.proxy_app import create_app


def _default_config_path() -> Optional[Path]:
    candidate = Path.cwd() / "promptlens.toml"
    return candidate if candidate.exists() else None


def _default_log_dir() -> Path:
    return Path(os.path.expanduser("~/.promptlens/logs"))


def _version_callback(value: bool) -> None:
    if not value:
        return
    from promptlens import __version__

    typer.echo(__version__)
    raise typer.Exit()


def cli(
    config: Annotated[
        Optional[Path],
        typer.Option(
            "--config",
            "-c",
            exists=True,
            dir_okay=False,
            readable=True,
            help="Path to TOML config file. Defaults to ./promptlens.toml if present.",
        ),
    ] = None,
    log_dir: Annotated[
        Path,
        typer.Option("--log-dir", help="Directory to write JSONL logs."),
    ] = _default_log_dir(),
    host: Annotated[str, typer.Option("--host", help="Bind host.")] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port", help="Bind port.")] = 8000,
    llm_endpoint: Annotated[
        Optional[str],
        typer.Option(
            "--llm-endpoint",
            help="Upstream OpenAI-compatible base URL (overrides config).",
        ),
    ] = None,
    timeout_s: Annotated[
        Optional[float],
        typer.Option(
            "--timeout", help="Upstream request timeout in seconds (overrides config)."
        ),
    ] = None,
    max_log_file_bytes: Annotated[
        Optional[int],
        typer.Option(
            "--max-log-file-bytes",
            help="Rotate logs when file exceeds this size (overrides config).",
        ),
    ] = None,
    max_prompt_bytes: Annotated[
        Optional[int],
        typer.Option(
            "--max-prompt-bytes",
            help="Max bytes of prompt content to store per log entry (overrides config).",
        ),
    ] = None,
    pid_file: Annotated[
        Optional[Path],
        typer.Option(
            "--pid-file", help="Optional PID file path (default: <log-dir>/plens.pid)."
        ),
    ] = None,
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            callback=_version_callback,
            is_eager=True,
            help="Print version and exit.",
        ),
    ] = False,
) -> None:
    """
    PromptLens: OpenAI-compatible HTTP proxy for logging LLM interactions.

    Captures user inputs and model responses (including tool use) as JSONL logs.
    Supports streaming responses, multi-turn conversations, and function calling.

    Examples:

        # Run with upstream endpoint
        plens --llm-endpoint http://127.0.0.1:4000 --port 8080

        # Use config file
        plens --config ./promptlens.toml

        # Custom log directory
        plens --llm-endpoint http://localhost:4000 --log-dir ./logs
    """
    config_path = config or _default_config_path()
    if config_path is None and llm_endpoint is None:
        raise typer.BadParameter(
            "Provide --config or --llm-endpoint (or create ./promptlens.toml)."
        )

    cfg: AppConfig = load_config(config_path) if config_path else AppConfig()

    if llm_endpoint is not None:
        cfg = cfg.model_copy(
            update={
                "upstream": cfg.upstream.model_copy(update={"base_url": llm_endpoint})
            }
        )
    if timeout_s is not None:
        cfg = cfg.model_copy(
            update={
                "upstream": cfg.upstream.model_copy(update={"timeout_s": timeout_s})
            }
        )
    if max_log_file_bytes is not None:
        cfg = cfg.model_copy(
            update={
                "logging": cfg.logging.model_copy(
                    update={"max_file_bytes": max_log_file_bytes}
                )
            }
        )
    if max_prompt_bytes is not None:
        cfg = cfg.model_copy(
            update={
                "logging": cfg.logging.model_copy(
                    update={"max_prompt_bytes": max_prompt_bytes}
                )
            }
        )

    log_dir.mkdir(parents=True, exist_ok=True)
    cfg = cfg.model_copy(
        update={"logging": cfg.logging.model_copy(update={"log_dir": str(log_dir)})}
    )

    pid_path = pid_file or (log_dir / "plens.pid")
    with PidFile(pid_path):
        logger = JsonlLogger.from_config(cfg.logging)
        fastapi_app = create_app(cfg, logger)
        uvicorn.run(
            fastapi_app,
            host=host,
            port=port,
            log_level=cfg.server.log_level,
            access_log=False,
        )


def main() -> None:
    typer.run(cli)


if __name__ == "__main__":  # pragma: no cover
    main()
