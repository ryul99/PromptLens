#!/usr/bin/env python3
"""Test streaming response scenario."""

import sys

from common import get_client, get_model, print_log_hint, print_section


def test_streaming() -> None:
    """Test streaming responses."""
    print_section("Streaming Response Test")

    client = get_client()
    model = get_model()

    try:
        print("\nStreaming response: ", end="", flush=True)

        stream = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": "Count from 1 to 5, with each number on a new line.",
                }
            ],
            stream=True,
        )

        for chunk in stream:
            if chunk.choices[0].delta.content:
                print(chunk.choices[0].delta.content, end="", flush=True)

        print("\n\n✅ Streaming test completed")
        print(
            "\n💡 Check the log file - streaming responses are accumulated and logged after completion."
        )
        print("   Tool calls in streaming are also captured and parsed.")

        print_log_hint()
    except Exception as e:
        print(f"\n❌ Streaming test failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    test_streaming()
