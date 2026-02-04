"""Common utilities for test scripts."""

import os

from openai import OpenAI


def get_client() -> OpenAI:
    """Create and return an OpenAI client configured for the proxy."""
    base_url = os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:8080")
    api_key = os.getenv("OPENAI_API_KEY", "sk-test-key")

    return OpenAI(base_url=base_url, api_key=api_key)


def get_model() -> str:
    """Get the model name from environment or use default."""
    return os.getenv("OPENAI_MODEL", "gpt-4o-mini")


def print_section(title: str) -> None:
    """Print a section header."""
    print(f"\n{'=' * 60}")
    print(f" {title}")
    print(f"{'=' * 60}")
    print(f"Model: {get_model()}")
    print(f"Proxy: {os.getenv('OPENAI_BASE_URL', 'http://127.0.0.1:8080')}")


def print_log_hint() -> None:
    """Print hint for viewing logs."""
    print(f"\n{'=' * 60}")
    print("View logs:")
    print("  tail -n 20 ./.promptlens/logs/promptlens.jsonl | jq .")
    print(f"{'=' * 60}")
