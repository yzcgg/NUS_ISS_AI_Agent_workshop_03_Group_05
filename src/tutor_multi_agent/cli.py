from __future__ import annotations

import argparse
import json

from .runner import run_turn


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one tutoring agent turn.")
    parser.add_argument("--input", required=True, help="User input text")
    parser.add_argument("--session", required=True, help="Session id")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model name")
    args = parser.parse_args()

    result = run_turn(
        user_input=args.input,
        session_id=args.session,
        model_name=args.model,
    )
    print("=== FINAL ANSWER ===")
    print(result["final_answer"])
    print("\n=== TOOL EVENTS ===")
    print(json.dumps(result["tool_events"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
