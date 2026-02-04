#!/usr/bin/env python3
"""Test tool use (function calling) scenario."""

import sys

from common import get_client, get_model, print_log_hint, print_section


def test_tool_use() -> None:
    """Test tool use (function calling)."""
    print_section("Tool Use Test")

    client = get_client()
    model = get_model()

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "The temperature unit",
                        },
                    },
                    "required": ["location"],
                },
            },
        }
    ]

    try:
        print("\nSending request with tools defined...")
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": "What's the weather like in Tokyo and Paris?",
                }
            ],
            tools=tools,
        )

        message = response.choices[0].message

        if message.tool_calls:
            print(f"\n✅ Tool calls requested: {len(message.tool_calls)}")
            for tool_call in message.tool_calls:
                function_name = tool_call.function.name
                function_args = tool_call.function.arguments
                print(f"  - {function_name}({function_args})")

            print("\n💡 Check the log file for the 'tool_calls' field in the output.")
            print("   Non-streaming responses include structured tool call data.")
        else:
            print("\n⚠️  No tool calls made.")
            print(f"Response: {message.content}")
            print(
                "\nNote: The model may not support tools or may not have triggered them."
            )

        print_log_hint()
    except Exception as e:
        print(f"\n❌ Tool use test failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    test_tool_use()
