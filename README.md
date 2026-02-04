PromptLens (`plens`) is a local-first, OpenAI-compatible HTTP proxy that logs prompts as append-only JSONL.

## Quickstart

Requirements:

- Python 3.12+
- `uv` (recommended)

Install dependencies:

```bash
uv sync
```

Run against an existing OpenAI-compatible upstream:

```bash
uv run plens --llm-endpoint http://127.0.0.1:4000 --port 8080 --log-dir ./.promptlens/logs
```

Logs are written to `./.promptlens/logs/promptlens.jsonl` (one JSON object per line).

Basic end-to-end test (requires your upstream to expose OpenAI-compatible routes):

```bash
curl -sS http://127.0.0.1:8080/v1/models
```

## Config file

Create `promptlens.toml` in your project directory:

```toml
[upstream]
base_url = "http://127.0.0.1:4000"
timeout_s = 60.0
verify_ssl = true

[logging]
filename = "promptlens.jsonl"
max_file_bytes = 52428800
max_prompt_bytes = 262144

[server]
log_level = "info"
```

Then run:

```bash
uv run plens --config ./promptlens.toml --port 8080 --log-dir ./.promptlens/logs
```

## Inspect logs with Unix tools

Follow logs:

```bash
tail -f ./.promptlens/logs/promptlens.jsonl
```

Pretty-print the most recent event:

```bash
tail -n 1 ./.promptlens/logs/promptlens.jsonl | jq .
```
