#!/usr/bin/env python3
"""
Extract game transcript as a table for presentation slides.

Shows Turn | Commentary | Guess | Game State in a clean table format.

Usage:
    uv run analysis/extract_game_table.py --log logs/your-log.eval --sample apple --turns 5
"""

import argparse
import sys
import textwrap
from pathlib import Path

from inspect_ai.log import read_eval_log


def extract_game_state(tool_response: str) -> str:
    """Extract the current word state from tool response."""
    for line in tool_response.split('\n'):
        if line.startswith("Word:"):
            return line.replace("Word:", "").strip()
    return "?"


def extract_table(sample, num_turns: int = 5, format: str = "markdown"):
    """Extract game as a table."""
    rows = []
    turn_count = 0
    current_commentary = None
    done = False
    
    for msg in sample.messages:
        if done:
            break
            
        if msg.role == "system":
            continue
        
        if msg.role == "assistant":
            # Capture full commentary
            if msg.content and msg.content.strip():
                # Clean up the commentary - remove extra whitespace and newlines
                commentary = ' '.join(msg.content.strip().split())
                current_commentary = commentary
            
            # Process tool calls
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    if tc.function == "hangman_guess":
                        import json
                        try:
                            args = json.loads(tc.arguments) if isinstance(tc.arguments, str) else tc.arguments
                            letter = args.get("letter", "?")
                            
                            # We'll get the game state from the next tool response
                            rows.append({
                                'turn': turn_count + 1,
                                'commentary': current_commentary or "(no commentary)",
                                'guess': letter,
                                'state': None  # Will be filled by tool response
                            })
                            turn_count += 1
                            current_commentary = None
                            
                            if turn_count >= num_turns:
                                done = True
                                break
                        except:
                            pass
        
        elif msg.role == "tool":
            # Fill in the game state for the last row
            if rows and rows[-1]['state'] is None:
                rows[-1]['state'] = extract_game_state(msg.content)
    
    # Format as table
    if format == "markdown":
        lines = []
        lines.append("| Turn | Commentary | Guess | Game State |")
        lines.append("|------|------------|-------|------------|")
        for row in rows:
            state = row['state'] or "?"
            commentary = row['commentary']
            
            # Wrap long commentary - use <br> for line breaks in markdown tables
            if len(commentary) > 80:
                wrapped = textwrap.wrap(commentary, width=80)
                commentary = '<br>'.join(wrapped)
            
            lines.append(f"| {row['turn']} | {commentary} | `{row['guess']}` | `{state}` |")
        return "\n".join(lines)
    
    elif format == "ascii":
        # Use wider column for commentary
        commentary_width = 60
        
        lines = []
        header = f"{'Turn':<6} | {'Commentary':<{commentary_width}} | {'Guess':<6} | {'Game State':<20}"
        lines.append(header)
        lines.append("-" * len(header))
        
        for row in rows:
            commentary = row['commentary']
            state = row['state'] or "?"
            
            # Wrap long commentary into multiple lines
            if len(commentary) > commentary_width:
                wrapped_lines = textwrap.wrap(commentary, width=commentary_width)
                # First line with all columns
                lines.append(f"{row['turn']:<6} | {wrapped_lines[0]:<{commentary_width}} | {row['guess']:<6} | {state:<20}")
                # Subsequent lines with only commentary column filled
                for wrapped_line in wrapped_lines[1:]:
                    lines.append(f"{'':6} | {wrapped_line:<{commentary_width}} | {'':6} | {'':20}")
            else:
                lines.append(f"{row['turn']:<6} | {commentary:<{commentary_width}} | {row['guess']:<6} | {state:<20}")
        
        return "\n".join(lines)
    
    else:  # csv
        lines = []
        lines.append("Turn,Commentary,Guess,Game State")
        for row in rows:
            state = row['state'] or "?"
            commentary = row['commentary'].replace('"', '""')  # Escape quotes
            lines.append(f'{row["turn"]},"{commentary}",{row["guess"]},"{state}"')
        return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Extract game transcript as a table"
    )
    parser.add_argument("--log", required=True, help="Path to eval log file")
    parser.add_argument("--sample", help="Sample ID or index (default: 0)")
    parser.add_argument("--turns", type=int, default=5, help="Number of turns to show")
    parser.add_argument(
        "--format",
        choices=["markdown", "ascii", "csv"],
        default="markdown",
        help="Output format"
    )
    
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
    
    if args.format == "markdown":
        print(f"### Sample: {sample.id}\n")
    else:
        print(f"Sample: {sample.id}\n")
    
    print(extract_table(sample, args.turns, args.format))
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
