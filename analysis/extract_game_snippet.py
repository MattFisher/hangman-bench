#!/usr/bin/env python3
"""
Extract a short snippet from a Hangman game for presentation slides.

Shows just a few turns with model commentary to illustrate behavior.

Usage:
    uv run analysis/extract_game_snippet.py --log logs/your-log.eval --sample apple --turns 2
"""

import argparse
import sys
from pathlib import Path

from inspect_ai.log import read_eval_log


def extract_snippet(sample, num_turns: int = 2):
    """Extract a short snippet showing model commentary pattern."""
    lines = []
    turn_count = 0
    skip_next_user = True  # Skip initial user prompt
    
    for msg in sample.messages:
        if msg.role == "system":
            continue
        
        if msg.role == "user":
            if skip_next_user:
                skip_next_user = False
                continue
            # Show continue prompts
            if "Continue by calling" in msg.content:
                lines.append(f'User:  "{msg.content}"')
        
        elif msg.role == "assistant":
            if turn_count >= num_turns:
                break
            
            # Show model commentary (first sentence only)
            if msg.content and msg.content.strip():
                first_sentence = msg.content.strip().split('.')[0] + '."'
                if len(first_sentence) > 80:
                    first_sentence = first_sentence[:77] + '..."'
                lines.append(f'Model: "{first_sentence}')
            
            # Show tool calls
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    if tc.function == "hangman_guess":
                        import json
                        try:
                            args = json.loads(tc.arguments) if isinstance(tc.arguments, str) else tc.arguments
                            letter = args.get("letter", "?")
                            lines.append(f'Model: [calls hangman_guess("{letter}")]')
                            turn_count += 1
                        except:
                            pass
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Extract short game snippet for slides"
    )
    parser.add_argument("--log", required=True, help="Path to eval log file")
    parser.add_argument("--sample", help="Sample ID or index (default: 0)")
    parser.add_argument("--turns", type=int, default=2, help="Number of turns to show")
    
    args = parser.parse_args()
    
    log_path = Path(args.log)
    if not log_path.exists():
        print(f"Error: Log file not found: {log_path}", file=sys.stderr)
        return 1
    
    log = read_eval_log(str(log_path))
    
    if not log.samples:
        print("Error: No samples found", file=sys.stderr)
        return 1
    
    # Get sample
    if args.sample:
        try:
            idx = int(args.sample)
            sample = log.samples[idx]
        except (ValueError, IndexError):
            matching = [s for s in log.samples if s.id == args.sample]
            if not matching:
                print(f"Error: Sample not found: {args.sample}", file=sys.stderr)
                return 1
            sample = matching[0]
    else:
        sample = log.samples[0]
    
    print(f"# Sample: {sample.id}\n")
    print("```text")
    print(extract_snippet(sample, args.turns))
    print("```")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
