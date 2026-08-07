#!/usr/bin/env python3
"""
Extract game transcripts from Hangman eval logs.

Shows the sequence of model commentary, tool calls, and tool responses
to illustrate model behavior patterns.

Usage:
    uv run analysis/extract_game_transcript.py --log logs/your-log.eval --sample apple
    uv run analysis/extract_game_transcript.py --log logs/your-log.eval --sample 0
    uv run analysis/extract_game_transcript.py --log logs/your-log.eval --all
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

from inspect_ai.log import read_eval_log


def format_tool_call(tool_call) -> str:
    """Format a tool call for display."""
    if tool_call.function == "hangman_guess":
        args = tool_call.arguments
        if isinstance(args, dict):
            letter = args.get("letter", "?")
        else:
            # Try to parse as string
            import json
            try:
                parsed = json.loads(args)
                letter = parsed.get("letter", "?")
            except:
                letter = "?"
        return f'hangman_guess("{letter}")'
    elif tool_call.function == "submit":
        args = tool_call.arguments
        if isinstance(args, dict):
            word = args.get("answer", "?")
        else:
            import json
            try:
                parsed = json.loads(args)
                word = parsed.get("answer", "?")
            except:
                word = "?"
        return f'submit("{word}")'
    else:
        return f"{tool_call.function}(...)"


def extract_transcript(sample, max_turns: Optional[int] = None, show_system: bool = False):
    """Extract a readable transcript from a sample."""
    lines = []
    
    if show_system and sample.messages and sample.messages[0].role == "system":
        lines.append("=" * 60)
        lines.append("SYSTEM MESSAGE:")
        lines.append(sample.messages[0].content[:200] + "..." if len(sample.messages[0].content) > 200 else sample.messages[0].content)
        lines.append("=" * 60)
        lines.append("")
    
    turn_count = 0
    for i, msg in enumerate(sample.messages):
        if msg.role == "system":
            continue
            
        if msg.role == "user":
            # Skip initial prompt, show continue prompts
            if i > 1:  # Not the first user message
                lines.append(f"User: {msg.content}")
        
        elif msg.role == "assistant":
            turn_count += 1
            if max_turns and turn_count > max_turns:
                lines.append(f"... ({len(sample.messages) - i} more messages)")
                break
                
            # Show model commentary if present
            if msg.content and msg.content.strip():
                lines.append(f"Model: \"{msg.content.strip()}\"")
            
            # Show tool calls
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    lines.append(f"Model: [calls {format_tool_call(tc)}]")
        
        elif msg.role == "tool":
            # Show tool response (truncated)
            content = msg.content
            if len(content) > 150:
                # Just show the key parts
                if "Word:" in content:
                    word_line = [line for line in content.split('\n') if line.startswith("Word:")][0]
                    status_line = [line for line in content.split('\n') if line.startswith("Status:")][0]
                    lines.append(f"Tool: {word_line}")
                    lines.append(f"      {status_line}")
                else:
                    lines.append(f"Tool: {content[:100]}...")
            else:
                lines.append(f"Tool: {content}")
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Extract game transcripts from Hangman eval logs"
    )
    parser.add_argument(
        "--log",
        required=True,
        help="Path to eval log file (.eval)"
    )
    parser.add_argument(
        "--sample",
        help="Sample ID (word) or index number to extract. Use --all for all samples."
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Extract all samples"
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        help="Maximum number of turns to show per sample"
    )
    parser.add_argument(
        "--show-system",
        action="store_true",
        help="Show system message"
    )
    parser.add_argument(
        "--format",
        choices=["text", "markdown"],
        default="text",
        help="Output format"
    )
    
    args = parser.parse_args()
    
    log_path = Path(args.log)
    if not log_path.exists():
        print(f"Error: Log file not found: {log_path}", file=sys.stderr)
        return 1
    
    # Read log
    log = read_eval_log(str(log_path))
    
    if not log.samples:
        print("Error: No samples found in log", file=sys.stderr)
        return 1
    
    # Determine which samples to extract
    if args.all:
        samples = log.samples
    elif args.sample:
        # Try as index first
        try:
            idx = int(args.sample)
            if idx < 0 or idx >= len(log.samples):
                print(f"Error: Sample index {idx} out of range (0-{len(log.samples)-1})", file=sys.stderr)
                return 1
            samples = [log.samples[idx]]
        except ValueError:
            # Try as word ID
            matching = [s for s in log.samples if s.id == args.sample]
            if not matching:
                print(f"Error: Sample '{args.sample}' not found", file=sys.stderr)
                print(f"Available samples: {', '.join(s.id for s in log.samples[:10])}...", file=sys.stderr)
                return 1
            samples = matching
    else:
        # Default to first sample
        samples = [log.samples[0]]
    
    # Extract and print transcripts
    for i, sample in enumerate(samples):
        if args.format == "markdown":
            print(f"## Sample: {sample.id}")
            print()
            print("```text")
            print(extract_transcript(sample, args.max_turns, args.show_system))
            print("```")
            print()
        else:
            if len(samples) > 1:
                print(f"\n{'='*60}")
                print(f"Sample {i+1}/{len(samples)}: {sample.id}")
                print('='*60)
            print(extract_transcript(sample, args.max_turns, args.show_system))
            if len(samples) > 1:
                print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
