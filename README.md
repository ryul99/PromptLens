PromptLens (`plens`) is a local-first, OpenAI-compatible HTTP proxy that logs prompts as append-only JSONL.

## Logging Format

Each request generates one or two log entries:

- **User input**: `{"role": "user", "type": "...", "content": ...}`
- **Model response**: `{"role": "assistant", "type": "...", "content": ..., "tool_calls": [...]}` (if applicable)

### Multi-Turn Conversations

For chat completions, the **entire conversation history** is logged in the input field. This means previous exchanges are duplicated across requests.

Example log entries for a 3-turn conversation:

```json
{"timestamp": "...", "input": {"role": "user", "type": "chat", "content": [{"role": "user", "content": "Hello"}]}, "truncated": false}
{"timestamp": "...", "output": {"role": "assistant", "type": "chat", "content": "Hi there!"}, "truncated": false}

{"timestamp": "...", "input": {"role": "user", "type": "chat", "content": [
  {"role": "user", "content": "Hello"},
  {"role": "assistant", "content": "Hi there!"},
  {"role": "user", "content": "How are you?"}
]}, "truncated": false}
{"timestamp": "...", "output": {"role": "assistant", "type": "chat", "content": "I'm doing well!"}, "truncated": false}
```

### Tool Use

Tool calls are logged in the `tool_calls` field:

```json
{
  "timestamp": "...",
  "output": {
    "role": "assistant",
    "type": "chat",
    "content": "Let me check that for you.",
    "tool_calls": [
      {
        "id": "call_abc123",
        "type": "function",
        "function": {
          "name": "get_weather",
          "arguments": "{\"location\": \"Tokyo\"}"
        }
      }
    ]
  },
  "truncated": false
}
```

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

## Test Scenarios

Individual test scripts are provided to verify logging behavior for different scenarios:

```bash
# Install the OpenAI package
uv add openai

# Run individual tests
uv run python scripts/test_single_turn.py
uv run python scripts/test_multi_turn.py
uv run python scripts/test_tool_use.py
uv run python scripts/test_streaming.py
```

### Configuration

Configure the model and proxy URL via environment variables:

```bash
# Set model (default: gpt-4o-mini)
export OPENAI_MODEL="llama-3.1-8b"

# Set proxy URL (default: http://127.0.0.1:8080)
export OPENAI_BASE_URL="http://127.0.0.1:8080"

# Run test with custom configuration
uv run python scripts/test_multi_turn.py
```

### Test Descriptions

| Script | Description |
|--------|-------------|
| `test_single_turn.py` | Simple question and answer |
| `test_multi_turn.py` | 3-turn conversation showing history accumulation |
| `test_tool_use.py` | Function calling with tool_calls |
| `test_streaming.py` | Real-time streaming responses |
