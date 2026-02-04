#!/usr/bin/env python3
"""Test multi-turn conversation scenario."""

import sys
from time import sleep

from common import get_client, get_model, print_log_hint, print_section


def test_multi_turn() -> None:
    """Test a multi-turn conversation."""
    print_section("Multi-Turn Conversation Test")

    client = get_client()
    model = get_model()

    try:
        # Turn 1
        print("\n[Turn 1] User asks about Python")
        response1 = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": "What is Python? Answer in one sentence."},
            ],
        )
        content1 = response1.choices[0].message.content
        print(f"Assistant: {content1}")

        sleep(0.5)

        # Turn 2
        print("\n[Turn 2] User asks a follow-up question")
        response2 = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": "What is Python? Answer in one sentence."},
                {"role": "assistant", "content": content1},
                {
                    "role": "user",
                    "content": "Is it easy to learn? Answer in one sentence.",
                },
            ],
        )
        content2 = response2.choices[0].message.content
        print(f"Assistant: {content2}")

        sleep(0.5)

        # Turn 3
        print("\n[Turn 3] User asks another follow-up question")
        response3 = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": "What is Python? Answer in one sentence."},
                {"role": "assistant", "content": content1},
                {
                    "role": "user",
                    "content": "Is it easy to learn? Answer in one sentence.",
                },
                {"role": "assistant", "content": content2},
                {
                    "role": "user",
                    "content": "What are its main uses? Answer in one sentence.",
                },
            ],
        )
        print(f"Assistant: {response3.choices[0].message.content}")

        print("\n✅ Multi-turn test completed")
        print(
            "\n💡 Check the log file to see how conversation history is captured in each input."
        )

        print_log_hint()
    except Exception as e:
        print(f"\n❌ Multi-turn test failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    test_multi_turn()
