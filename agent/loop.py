"""
agent/loop.py
=============
Agent reasoning loop — Google Gemini backend.

Uses google-genai SDK with native function calling.
Model: gemini-2.0-flash (free tier, fast, supports function calling).

Falls back gracefully if the model changes — set GEMINI_MODEL env var to override.

Usage:
    for event in run_agent(df=df, task_type="full_audit"):
        if event["type"] == "text_delta":   print(event["text"])
        elif event["type"] == "tool_call":  print(event["tool"])
        elif event["type"] == "done":       break
        elif event["type"] == "error":      print(event["message"]); break
"""

from __future__ import annotations

import json
import os
from typing import Generator, Any

import pandas as pd

from agent.tools   import get_tool_definitions, execute_tool
from agent.prompts import build_task_prompt
from agent.memory  import get_findings_summary

MODEL     = os.environ.get("GEMINI_MODEL", "gemini-1.5-flash")
MAX_ITER  = 20
MAX_TOKENS = 4096

Event = dict[str, Any]


def _build_gemini_tools(tool_defs: list[dict]):
    """Convert Anthropic-style tool definitions to Gemini FunctionDeclaration list."""
    from google.genai import types

    declarations = []
    for t in tool_defs:
        schema = t.get("input_schema", {})
        props  = schema.get("properties", {})
        req    = schema.get("required", [])

        # Convert each property to Gemini Schema
        gemini_props = {}
        for pname, pdef in props.items():
            ptype = pdef.get("type", "string").upper()
            # Gemini type map
            type_map = {
                "STRING":  "STRING",
                "NUMBER":  "NUMBER",
                "INTEGER": "INTEGER",
                "BOOLEAN": "BOOLEAN",
                "OBJECT":  "OBJECT",
                "ARRAY":   "ARRAY",
            }
            gemini_type = type_map.get(ptype, "STRING")
            gemini_props[pname] = types.Schema(
                type=gemini_type,
                description=pdef.get("description", ""),
            )

        fn_schema = types.Schema(
            type="OBJECT",
            properties=gemini_props,
            required=req,
        )
        declarations.append(
            types.FunctionDeclaration(
                name=t["name"],
                description=t["description"],
                parameters=fn_schema,
            )
        )
    return [types.Tool(function_declarations=declarations)]


def run_agent(
    df: pd.DataFrame,
    task_type: str = "full_audit",
    user_input: str = "",
    max_iter: int = MAX_ITER,
) -> Generator[Event, None, None]:
    """
    Run the agent reasoning loop using Google Gemini.

    Yields Event dicts — same interface as the Anthropic version so the
    Streamlit page works without changes.
    """
    import google.genai as genai
    from google.genai import types

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        yield {"type": "error",
               "message": "GEMINI_API_KEY not set. Add it to Streamlit secrets."}
        return

    client = genai.Client(api_key=api_key)

    # Build system prompt + tool definitions
    findings  = get_findings_summary()
    system    = build_task_prompt(task_type, user_input, findings)
    tools     = _build_gemini_tools(get_tool_definitions())

    # Gemini conversation history
    contents: list = [
        types.UserContent(parts=[types.Part(text="Please begin the analysis.")])
    ]

    final_text = ""
    iteration  = 0

    while iteration < max_iter:
        iteration += 1
        yield {"type": "thinking", "text": f"Agent iteration {iteration}…"}

        # Retry up to 3 times on rate-limit (429) with backoff
        response = None
        last_err = None
        for attempt in range(3):
            try:
                response = client.models.generate_content(
                    model=MODEL,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction=system,
                        tools=tools,
                        max_output_tokens=MAX_TOKENS,
                        temperature=0.1,
                    ),
                )
                break
            except Exception as e:
                last_err = e
                msg = str(e)
                if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                    import time
                    wait = 15 * (attempt + 1)   # 15s, 30s, 45s
                    yield {"type": "thinking",
                           "text": f"Rate limit hit — waiting {wait}s before retry {attempt+2}/3…"}
                    time.sleep(wait)
                else:
                    yield {"type": "error", "message": msg}
                    return
        if response is None:
            yield {"type": "error",
                   "message": "Rate limit: " + str(last_err)}
            return

        # ── Parse response parts ──────────────────────────────────────────────
        candidate   = response.candidates[0]
        parts       = candidate.content.parts if candidate.content else []
        text_parts  = []
        fn_calls    = []

        for part in parts:
            if hasattr(part, "text") and part.text:
                text_parts.append(part.text)
                yield {"type": "text_delta", "text": part.text}
            if hasattr(part, "function_call") and part.function_call:
                fn_calls.append(part.function_call)

        if text_parts:
            final_text += "\n".join(text_parts)

        # ── Stopping condition ────────────────────────────────────────────────
        finish = candidate.finish_reason
        # STOP = natural end, MAX_TOKENS = ran out of tokens
        # If no function calls and model has stopped, we are done
        if not fn_calls:
            yield {"type": "done", "final_text": final_text}
            return

        # ── Add model turn to history ─────────────────────────────────────────
        contents.append(types.ModelContent(parts=parts))

        # ── Execute tool calls ────────────────────────────────────────────────
        fn_response_parts = []
        for fc in fn_calls:
            tool_name = fc.name
            # Gemini passes args as a dict-like object
            try:
                params = dict(fc.args) if fc.args else {}
            except Exception:
                params = {}

            yield {"type": "tool_call", "tool": tool_name, "params": params}

            result = execute_tool(tool_name, params, df)

            yield {"type": "tool_result", "tool": tool_name, "result": result}

            fn_response_parts.append(
                types.Part(
                    function_response=types.FunctionResponse(
                        name=tool_name,
                        response={"result": json.dumps(result, default=str)[:6000]},
                    )
                )
            )

        # ── Add tool results to conversation ──────────────────────────────────
        contents.append(
            types.UserContent(parts=fn_response_parts)
        )

    yield {"type": "error",
           "message": f"Max iterations ({max_iter}) reached without conclusion."}
