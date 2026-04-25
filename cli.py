"""
Interactive CLI for the Payment Collection AI Agent.

Run:
    python cli.py
"""

from __future__ import annotations

from agent import Agent


def main() -> None:
    agent = Agent()

    print("Payment Collection AI Agent")
    print("Type 'exit' or 'quit' to end.\n")

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() in {"exit", "quit"}:
            print("Agent: Goodbye.")
            break

        if not user_input:
            continue

        response = agent.next(user_input)
        print(f"Agent: {response['message']}")


if __name__ == "__main__":
    main()