"""
agent/memory.py
===============
Persistent research log for the agent.

Storage
-------
Findings are stored in  agent_memory.json  in the repo root.
This file IS committed to git — it is your research log.
Each finding is immutable once written (append-only).

On Streamlit Cloud: findings persist within a session but reset on redeployment.
For true persistence across deployments, use st.session_state or a cloud store.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

MEMORY_FILE = Path(__file__).resolve().parent.parent / "agent_memory.json"


def _load() -> list[dict]:
    if MEMORY_FILE.exists():
        try:
            return json.loads(MEMORY_FILE.read_text())
        except Exception:
            return []
    return []


def _save(findings: list[dict]) -> None:
    MEMORY_FILE.write_text(json.dumps(findings, indent=2, default=str))


def append_finding(finding: dict) -> dict:
    """Add a finding to the research log and return it with metadata."""
    findings = _load()
    entry = {
        "id":         len(findings) + 1,
        "timestamp":  datetime.now(timezone.utc).isoformat(),
        **finding,
    }
    findings.append(entry)
    _save(findings)
    return entry


def get_all_findings() -> list[dict]:
    return _load()


def get_findings_summary() -> str:
    """Compact text summary of all findings for injection into agent context."""
    findings = _load()
    if not findings:
        return "No findings logged yet."
    lines = []
    for f in findings:
        conf = f.get("confidence", "?").upper()
        lines.append(
            f"[{f['id']}] [{conf}] {f.get('title','')}\n"
            f"  Finding: {f.get('finding','')}\n"
            f"  Action:  {f.get('action','')}"
        )
    return "\n\n".join(lines)


def clear_findings() -> None:
    """Clear all findings (use with care)."""
    _save([])
