"""
agent/loop.py
=============
The agent reasoning loop.

Architecture
------------
This is a standard Anthropic tool-use loop:

  1. Send user task + system prompt + tool definitions to Claude
  2. Claude returns either:
     a. A final text response  →  done
     b. One or more tool_use blocks  →  execute tools, send results back
  3. Repeat until a final text response is received

The loop streams events back to the caller via a generator so the Streamlit
UI can display progress in real time.

Usage
-----
    from agent.loop import run_agent

    for event in run_agent(df=df, task_type="full_audit"):
        if event["type"] == "text_delta":
            print(event["text"], end="", flush=True)
        elif event["type"] == "tool_call":
            print(f"\n[Tool] {event['tool']}({event['params']})")
        elif event["type"] == "tool_result":
            print(f"[Result] {event['result']}")
        elif event["type"] == "done":
            break
        elif event["type"] == "error":
            print(f"Error: {event['message']}")
            break
"""

from __future__ import annotations

import json
from typing import Generator, Any

import pandas as pd

from agent.tools   import get_tool_definitions, execute_tool
from agent.prompts import build_task_prompt
from agent.memory  import get_findings_summary


# ── Anthropic client (lazy import so the module loads without the key) ────────
def _get_client():
    import anthropic
    return anthropic.Anthropic()   # reads ANTHROPIC_API_KEY from env


MODEL     = "claude-opus-4-5"
MAX_ITER  = 20      # hard ceiling on tool-use rounds to prevent infinite loops
MAX_TOKENS = 4096


Event = dict[str, Any]


def run_agent(
    df: pd.DataFrame,
    task_type: str = "full_audit",
    user_input: str = "",
    max_iter: int = MAX_ITER,
) -> Generator[Event, None, None]:
    """
    Run the agent on a trading DataFrame.

    Yields Event dicts:
        {"type": "thinking",    "text": "..."}
        {"type": "text_delta",  "text": "..."}
        {"type": "tool_call",   "tool": "...", "params": {...}}
        {"type": "tool_result", "tool": "...", "result": {...}}
        {"type": "done",        "final_text": "..."}
        {"type": "error",       "message": "..."}
    """
    client     = _get_client()
    tools_defs = get_tool_definitions()
    findings   = get_findings_summary()
    system     = build_task_prompt(task_type, user_input, findings)

    messages: list[dict] = [
        {"role": "user", "content": "Please begin the analysis."}
    ]

    final_text = ""
    iteration  = 0

    while iteration < max_iter:
        iteration += 1
        yield {"type": "thinking", "text": f"Agent iteration {iteration}…"}

        try:
            response = client.messages.create(
                model      = MODEL,
                max_tokens = MAX_TOKENS,
                system     = system,
                tools      = tools_defs,
                messages   = messages,
            )
        except Exception as e:
            yield {"type": "error", "message": str(e)}
            return

        # ── collect text and tool-use blocks ─────────────────────────────────
        text_blocks      = []
        tool_use_blocks  = []

        for block in response.content:
            if block.type == "text":
                text_blocks.append(block.text)
                yield {"type": "text_delta", "text": block.text}
            elif block.type == "tool_use":
                tool_use_blocks.append(block)

        if text_blocks:
            final_text += "\n".join(text_blocks)

        # ── stopping condition ────────────────────────────────────────────────
        if response.stop_reason == "end_turn" or not tool_use_blocks:
            yield {"type": "done", "final_text": final_text}
            return

        # ── execute tool calls ────────────────────────────────────────────────
        # Add assistant message with all content
        messages.append({
            "role":    "assistant",
            "content": response.content,
        })

        tool_results = []
        for block in tool_use_blocks:
            tool_name = block.name
            params    = block.input if isinstance(block.input, dict) else {}

            yield {"type": "tool_call", "tool": tool_name, "params": params}

            result = execute_tool(tool_name, params, df)

            yield {"type": "tool_result", "tool": tool_name, "result": result}

            tool_results.append({
                "type":        "tool_result",
                "tool_use_id": block.id,
                "content":     json.dumps(result, default=str)[:8000],  # token safety
            })

        # ── add tool results to conversation ──────────────────────────────────
        messages.append({
            "role":    "user",
            "content": tool_results,
        })

    yield {"type": "error", "message": f"Max iterations ({max_iter}) reached."}
