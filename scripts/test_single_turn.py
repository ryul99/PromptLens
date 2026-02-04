#!/usr/bin/env python3
"""Test single-turn conversation scenario."""

import sys

from common import get_client, get_model, print_log_hint, print_section


def test_single_turn() -> None:
    """Test a single-turn conversation."""
    print_section("Single-Turn Conversation Test")

    client = get_client()
    model = get_model()

    try:
        print("\nSending request...")
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": "What is the capital of France? Answer in one word.",
                },
            ],
        )

        content = response.choices[0].message.content
        print(f"\nResponse: {content}")
        print("\n✅ Single-turn test completed")

        print_log_hint()
    except Exception as e:
        print(f"\n❌ Single-turn test failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    test_single_turn()
