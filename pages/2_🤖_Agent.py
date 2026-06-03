"""
pages/2_Agent.py  —  AI Trading Co-Pilot Agent
Works with Google Gemini (free) or Anthropic Claude (paid).
"""

import os
import sys
import json
from pathlib import Path

# ── Find repo root by walking up from pages/ until agent/__init__.py found ───
def _find_repo_root() -> str:
    p = Path(__file__).resolve().parent
    for _ in range(5):
        if (p / "agent" / "__init__.py").exists():
            return str(p)
        p = p.parent
    return str(Path.cwd())

_repo_root = _find_repo_root()
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import streamlit as st
import pandas as pd

# ── Inject API keys from Streamlit secrets ───────────────────────────────────
try:
    if "gemini" in st.secrets:
        os.environ["GEMINI_API_KEY"] = st.secrets["gemini"]["GEMINI_API_KEY"]
except Exception:
    pass
try:
    if "anthropic" in st.secrets:
        os.environ["ANTHROPIC_API_KEY"] = st.secrets["anthropic"]["ANTHROPIC_API_KEY"]
except Exception:
    pass

# ── Agent imports ─────────────────────────────────────────────────────────────
AGENT_AVAILABLE = False
_AGENT_ERR = ""
try:
    from agent.loop   import run_agent
    from agent.memory import get_all_findings, clear_findings
    AGENT_AVAILABLE = True
except ImportError as e:
    _AGENT_ERR = str(e)
except Exception as e:
    _AGENT_ERR = str(e)

# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Trading Co-Pilot",
    page_icon="🤖",
    layout="wide",
)

st.markdown("""
<style>
.agent-bubble {
    background:#1A2E45; border-radius:8px; padding:14px 18px;
    margin-bottom:10px; font-size:13px; line-height:1.7; color:#E0E8F0;
    white-space: pre-wrap;
}
.tool-call {
    background:#0D1B2A; border:1px solid #00C2CB; border-radius:6px;
    padding:8px 12px; margin:4px 0; font-size:11px; color:#00C2CB;
    font-family:monospace;
}
.tool-result {
    background:#0D1B2A; border:1px solid #607D8B; border-radius:6px;
    padding:8px 12px; margin:4px 0; font-size:11px; color:#B0BEC5;
    font-family:monospace; max-height:180px; overflow-y:auto;
}
.finding-high   { background:#0D2E1A; border-left:4px solid #27AE60;
                  border-radius:0 8px 8px 0; padding:12px 16px;
                  margin-bottom:10px; font-size:13px; line-height:1.6; }
.finding-medium { background:#2E1E00; border-left:4px solid #F5A623;
                  border-radius:0 8px 8px 0; padding:12px 16px;
                  margin-bottom:10px; font-size:13px; line-height:1.6; }
.finding-low    { background:#1A2E45; border-left:4px solid #607D8B;
                  border-radius:0 8px 8px 0; padding:12px 16px;
                  margin-bottom:10px; font-size:13px; line-height:1.6; }
</style>
""", unsafe_allow_html=True)


def get_df():
    return st.session_state.get("prev_df")

def api_key_ok() -> bool:
    return bool(
        os.environ.get("GEMINI_API_KEY") or
        os.environ.get("GOOGLE_API_KEY") or
        os.environ.get("ANTHROPIC_API_KEY")
    )

def active_provider() -> str:
    if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
        return "Gemini"
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "Anthropic"
    return "none"


def main():
    st.title("🤖 AI Trading Co-Pilot Agent")
    st.caption(
        "The agent autonomously analyses your trade data, runs statistical "
        "tests, and produces evidence-based trading rule recommendations."
    )

    # ── Agent module check ────────────────────────────────────────────────────
    if not AGENT_AVAILABLE:
        st.error("Could not import agent modules: " + _AGENT_ERR)
        with st.expander("Debug: file system view", expanded=True):
            st.code("__file__: " + str(Path(__file__).resolve()))
            st.code("repo root: " + _repo_root)
            st.code("sys.path[:5]:\n" + "\n".join(sys.path[:5]))
            repo = Path(_repo_root)
            try:
                st.code("repo contents:\n" + "\n".join(
                    sorted(p.name for p in repo.iterdir())
                ))
            except Exception as ex:
                st.code("error listing repo: " + str(ex))
            found = [
                str(Path(p) / "agent" / "__init__.py")
                for p in sys.path
                if (Path(p) / "agent" / "__init__.py").exists()
            ]
            st.code("agent/__init__.py found at:\n" + (
                "\n".join(found) if found else "NOT FOUND"
            ))
        return

    # ── Data check ────────────────────────────────────────────────────────────
    df = get_df()
    if df is None:
        st.warning(
            "No trade data loaded. Upload your trade export on the "
            "**📊 Main page** first, then return here."
        )
        return

    # ── API key check ─────────────────────────────────────────────────────────
    if not api_key_ok():
        st.error("No API key found.")
        st.markdown("Go to **Streamlit Cloud → your app → Settings → Secrets** and paste one of:")
        st.markdown("**Google Gemini (free tier — recommended):**")
        st.code("[gemini]\nGEMINI_API_KEY = \"your-key-from-aistudio.google.com\"",
                language="toml")
        st.markdown("**Anthropic Claude (paid):**")
        st.code("[anthropic]\nANTHROPIC_API_KEY = \"sk-ant-...\"",
                language="toml")
        st.markdown(
            "Get a free Gemini key at "
            "[aistudio.google.com/apikey](https://aistudio.google.com/apikey)"
        )
        return

    provider = active_provider()
    st.success(
        str(len(df)) + " trades loaded  ·  "
        + provider + " API key present"
    )

    col_left, col_right = st.columns([2, 1])

    # ── RIGHT: Research Log ───────────────────────────────────────────────────
    with col_right:
        st.markdown("### 📋 Research Log")
        findings = get_all_findings()
        if findings:
            for f in reversed(findings):
                conf  = f.get("confidence", "medium").lower()
                css   = "finding-" + (conf if conf in ["high","medium","low"] else "medium")
                st.markdown(
                    '<div class="' + css + '">'
                    '<strong>[' + str(f.get("id","")) + '] '
                    + f.get("title","") + '</strong> '
                    '<span style="color:#607D8B;font-size:11px">'
                    + conf.upper() + '</span><br>'
                    + f.get("finding","") + '<br>'
                    '<span style="color:#00C2CB">→ '
                    + f.get("action","") + '</span>'
                    '</div>',
                    unsafe_allow_html=True,
                )
            if st.button("🗑 Clear research log", type="secondary"):
                clear_findings()
                st.rerun()
        else:
            st.info("No findings yet. Run the agent to start building your research log.")

    # ── LEFT: Run Agent ───────────────────────────────────────────────────────
    with col_left:
        st.markdown("### 🎯 Run the Agent")

        task_map = {
            "Full statistical audit":   "full_audit",
            "Instrument deep dive":     "instrument_deep_dive",
            "Hold-time investigation":  "hold_time_investigation",
            "Stop-loss optimisation":   "stop_loss_optimisation",
            "Regime analysis":          "regime_analysis",
            "Custom question":          "custom",
        }
        task_label = st.selectbox("Task", list(task_map.keys()))
        task_type  = task_map[task_label]

        user_input = ""
        if task_type == "custom":
            user_input = st.text_area(
                "Your question",
                placeholder=(
                    "e.g. Is my MNQM6 edge concentrated in the 14-16 CET window? "
                    "Run a permutation test comparing that window to all other hours."
                ),
                height=100,
            )

        with st.expander("Advanced options"):
            max_iter   = st.slider("Max tool-use rounds", 3, 25, 12)
            show_tools = st.checkbox("Show tool calls in real time", value=True)

        run_btn = st.button("▶ Run Agent", type="primary", use_container_width=True)

        if run_btn:
            if task_type == "custom" and not user_input.strip():
                st.warning("Please enter your question first.")
                return

            output_container = st.container()
            tool_container   = st.container()
            full_text = ""
            tool_log  = []

            with st.spinner("Agent is working…"):
                for event in run_agent(
                    df=df,
                    task_type=task_type,
                    user_input=user_input,
                    max_iter=max_iter,
                ):
                    etype = event.get("type")

                    if etype == "thinking":
                        with output_container:
                            st.caption(event["text"])

                    elif etype == "text_delta":
                        full_text += event["text"]
                        with output_container:
                            st.markdown(
                                '<div class="agent-bubble">'
                                + full_text + '</div>',
                                unsafe_allow_html=True,
                            )

                    elif etype == "tool_call" and show_tools:
                        params_str = json.dumps(event["params"], indent=2)[:300]
                        tool_log.append(
                            '<div class="tool-call">🔧 '
                            + event["tool"] + "(" + params_str + ")</div>"
                        )
                        with tool_container:
                            st.markdown("".join(tool_log), unsafe_allow_html=True)

                    elif etype == "tool_result" and show_tools:
                        result_str = json.dumps(
                            event["result"], indent=2, default=str
                        )[:400]
                        tool_log.append(
                            '<div class="tool-result">✓ ' + result_str + "</div>"
                        )
                        with tool_container:
                            st.markdown("".join(tool_log), unsafe_allow_html=True)

                    elif etype == "done":
                        st.success("✅ Analysis complete — see Research Log →")
                        st.rerun()

                    elif etype == "error":
                        st.error("Agent error: " + event["message"])
                        break


if __name__ == "__main__":
    main()
