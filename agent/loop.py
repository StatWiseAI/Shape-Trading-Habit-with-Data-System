"""
agent/loop.py
=============
Agent reasoning loop — Google Gemini backend.

Compatible with google-genai >= 1.0.0 (tested on 2.7.0).
Uses gemini-1.5-flash-latest which is on the free tier.

Override model: set GEMINI_MODEL env var in Streamlit secrets, e.g.
  GEMINI_MODEL = "models/gemini-1.5-pro"
"""

from __future__ import annotations

import json
import os
import time
from typing import Generator, Any

import pandas as pd

from agent.tools   import get_tool_definitions, execute_tool
from agent.prompts import build_task_prompt
from agent.memory  import get_findings_summary

# gemini-1.5-flash-latest is the correct free-tier model name for v1beta API
MODEL      = os.environ.get("GEMINI_MODEL", "gemini-1.5-flash-latest")
MAX_ITER   = 20
MAX_TOKENS = 4096

Event = dict[str, Any]


def _tool_defs_to_gemini(tool_defs: list[dict]):
    """Convert Anthropic-style tool defs to Gemini Tool objects."""
    from google.genai import types

    declarations = []
    for t in tool_defs:
        schema = t.get("input_schema", {})
        props  = schema.get("properties", {})
        req    = schema.get("required", [])

        type_map = {
            "string":  "STRING",  "number":  "NUMBER",
            "integer": "INTEGER", "boolean": "BOOLEAN",
            "object":  "OBJECT",  "array":   "ARRAY",
        }

        gemini_props = {}
        for pname, pdef in props.items():
            raw_type   = pdef.get("type", "string").lower()
            gemini_type = type_map.get(raw_type, "STRING")
            gemini_props[pname] = types.Schema(
                type=gemini_type,
                description=pdef.get("description", ""),
            )

        declarations.append(
            types.FunctionDeclaration(
                name=t["name"],
                description=t["description"],
                parameters=types.Schema(
                    type="OBJECT",
                    properties=gemini_props,
                    required=req,
                ),
            )
        )
    return [types.Tool(function_declarations=declarations)]


def _call_with_retry(client, model, contents, config, max_attempts=3):
    """Call generate_content with exponential backoff on 429/503."""
    import google.genai as genai
    last_err = None
    for attempt in range(max_attempts):
        try:
            return client.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )
        except Exception as e:
            last_err = e
            msg = str(e)
            if "429" in msg or "RESOURCE_EXHAUSTED" in msg or "503" in msg:
                wait = 20 * (attempt + 1)   # 20s, 40s, 60s
                time.sleep(wait)
            else:
                raise
    raise last_err


def run_agent(
    df: pd.DataFrame,
    task_type: str = "full_audit",
    user_input: str = "",
    max_iter: int = MAX_ITER,
) -> Generator[Event, None, None]:
    """
    Run the agent reasoning loop.
    Yields Event dicts identical to the Anthropic version.
    """
    import google.genai as genai
    from google.genai import types

    api_key = (
        os.environ.get("GEMINI_API_KEY") or
        os.environ.get("GOOGLE_API_KEY")
    )
    if not api_key:
        yield {"type": "error",
               "message": "GEMINI_API_KEY not set. Add it to Streamlit secrets."}
        return

    client   = genai.Client(api_key=api_key)
    findings = get_findings_summary()
    system   = build_task_prompt(task_type, user_input, findings)
    tools    = _tool_defs_to_gemini(get_tool_definitions())

    config = types.GenerateContentConfig(
        system_instruction=system,
        tools=tools,
        max_output_tokens=MAX_TOKENS,
        temperature=0.1,
    )

    # Build conversation history
    # In google-genai v2.x use plain dicts for content turns
    contents = [{"role": "user", "parts": [{"text": "Please begin the analysis."}]}]

    final_text = ""
    iteration  = 0

    while iteration < max_iter:
        iteration += 1
        yield {"type": "thinking", "text": f"Agent iteration {iteration}…"}

        try:
            response = _call_with_retry(client, MODEL, contents, config)
        except Exception as e:
            msg = str(e)
            if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                yield {"type": "error",
                       "message": (
                           "Rate limit reached. Wait a minute then try again, "
                           "or reduce 'Max tool-use rounds' to 3-5. Error: " + msg[:200]
                       )}
            else:
                yield {"type": "error", "message": msg[:400]}
            return

        # ── Parse response ────────────────────────────────────────────────────
        candidate = response.candidates[0]
        parts     = candidate.content.parts if candidate.content else []

        text_parts = []
        fn_calls   = []

        for part in parts:
            # Text
            if hasattr(part, "text") and part.text:
                text_parts.append(part.text)
                yield {"type": "text_delta", "text": part.text}
            # Function call
            if hasattr(part, "function_call") and part.function_call:
                fn_calls.append(part.function_call)

        if text_parts:
            final_text += "\n".join(text_parts)

        # ── Stopping condition ────────────────────────────────────────────────
        if not fn_calls:
            yield {"type": "done", "final_text": final_text}
            return

        # ── Add model turn to history ─────────────────────────────────────────
        contents.append({
            "role": "model",
            "parts": [
                {"text": p.text} if (hasattr(p,"text") and p.text)
                else {"function_call": {
                    "name": p.function_call.name,
                    "args": dict(p.function_call.args) if p.function_call.args else {},
                }}
                for p in parts
            ]
        })

        # ── Execute tools ─────────────────────────────────────────────────────
        fn_response_parts = []
        for fc in fn_calls:
            name = fc.name
            try:
                params = dict(fc.args) if fc.args else {}
            except Exception:
                params = {}

            yield {"type": "tool_call", "tool": name, "params": params}

            result = execute_tool(name, params, df)

            yield {"type": "tool_result", "tool": name, "result": result}

            fn_response_parts.append({
                "function_response": {
                    "name": name,
                    "response": {
                        "result": json.dumps(result, default=str)[:6000]
                    },
                }
            })

        # ── Add tool results to conversation ──────────────────────────────────
        contents.append({
            "role": "user",
            "parts": fn_response_parts,
        })

    yield {"type": "error",
           "message": f"Max iterations ({max_iter}) reached without conclusion."}
