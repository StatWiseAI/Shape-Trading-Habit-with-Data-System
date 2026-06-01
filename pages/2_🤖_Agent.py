"""
pages/2_🤖_Agent.py
====================
Streamlit multi-page entry for the AI trading co-pilot agent.

This page is automatically picked up by Streamlit's multi-page app system
when placed in the  pages/  directory alongside  app.py.

Setup
-----
1. Set your Anthropic API key in Streamlit secrets:
   → .streamlit/secrets.toml (local)  OR  Streamlit Cloud secrets manager

   [anthropic]
   ANTHROPIC_API_KEY = "sk-ant-..."

2. Upload trade data on the main page (app.py) first — the agent reads
   the dataframe from st.session_state["prev_df"].
"""

import os, sys, json
from pathlib import Path

import streamlit as st
import pandas as pd

# ── path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── inject Anthropic API key from Streamlit secrets ───────────────────────────
if "anthropic" in st.secrets:
    os.environ["ANTHROPIC_API_KEY"] = st.secrets["anthropic"]["ANTHROPIC_API_KEY"]

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Trading Co-Pilot Agent",
    page_icon="🤖",
    layout="wide",
)

st.markdown("""
<style>
.agent-bubble {
    background:#1A2E45; border-radius:8px; padding:12px 16px;
    margin-bottom:8px; font-size:13px; line-height:1.6; color:#E0E8F0;
}
.tool-call {
    background:#0D1B2A; border:1px solid #00C2CB; border-radius:6px;
    padding:8px 12px; margin:4px 0; font-size:12px; color:#00C2CB;
    font-family:monospace;
}
.tool-result {
    background:#0D1B2A; border:1px solid #607D8B; border-radius:6px;
    padding:8px 12px; margin:4px 0; font-size:11px; color:#B0BEC5;
    font-family:monospace; max-height:200px; overflow-y:auto;
}
.finding-card {
    background:#0D2E1A; border-left:4px solid #27AE60;
    border-radius:0 8px 8px 0; padding:12px 16px;
    margin-bottom:10px; font-size:13px; line-height:1.6;
}
.finding-card.medium { border-left-color:#F5A623; background:#2E1E00; }
.finding-card.low    { border-left-color:#607D8B; background:#1A2E45; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def get_df() -> pd.DataFrame | None:
    return st.session_state.get("prev_df")

def api_key_ok() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    st.title("🤖 AI Trading Co-Pilot Agent")
    st.caption(
        "The agent autonomously analyses your trade data, runs statistical tests, "
        "and produces evidence-based trading rule recommendations."
    )

    # ── Pre-flight checks ─────────────────────────────────────────────────────
    df = get_df()
    if df is None:
        st.warning(
            "No trade data loaded. Upload your trade export on the **main page** first, "
            "then come back here."
        )
        return

    if not api_key_ok():
        st.error(
            "Anthropic API key not found.  \n"
            "Add it to `.streamlit/secrets.toml`:  \n"
            "```toml\n[anthropic]\nANTHROPIC_API_KEY = \"sk-ant-...\"\n```  \n"
            "Or set it in Streamlit Cloud → App settings → Secrets."
        )
        return

    st.success(f"✅ {len(df)} trades loaded · API key present")

    # ── Layout ────────────────────────────────────────────────────────────────
    col_left, col_right = st.columns([2, 1])

    with col_right:
        st.markdown("### 📋 Research Log")
        from agent.memory import get_all_findings, clear_findings
        findings = get_all_findings()
        if findings:
            for f in reversed(findings):
                conf  = f.get("confidence","medium").lower()
                conf_class = conf if conf in ["medium","low"] else ""
                st.markdown(
                    f'<div class="finding-card {conf_class}">'
                    f'<strong>[{f["id"]}] {f.get("title","")}</strong>  '
                    f'<span style="color:#607D8B;font-size:11px">{f.get("confidence","?").upper()}</span><br>'
                    f'{f.get("finding","")}<br>'
                    f'<span style="color:#00C2CB">→ {f.get("action","")}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            if st.button("🗑 Clear research log", type="secondary"):
                clear_findings()
                st.rerun()
        else:
            st.info("No findings yet. Run the agent to start building your research log.")

    with col_left:
        st.markdown("### 🎯 Run the Agent")

        # Task selection
        task_map = {
            "Full statistical audit":       "full_audit",
            "Instrument deep dive":         "instrument_deep_dive",
            "Hold-time investigation":      "hold_time_investigation",
            "Stop-loss optimisation":       "stop_loss_optimisation",
            "Regime analysis":              "regime_analysis",
            "Custom question":              "custom",
        }
        task_label = st.selectbox(
            "Task",
            list(task_map.keys()),
            help="Select a pre-defined research task or write your own question.",
        )
        task_type = task_map[task_label]

        user_input = ""
        if task_type == "custom":
            user_input = st.text_area(
                "Your question",
                placeholder="e.g. Is my MNQM6 edge concentrated in the 14-16 CET window? "
                            "Run a permutation test comparing that window to all other hours.",
                height=100,
            )

        # Advanced options
        with st.expander("Advanced options"):
            max_iter = st.slider("Max tool-use rounds", 3, 25, 12)
            show_tools = st.checkbox("Show tool calls in real time", value=True)

        run_btn = st.button("▶ Run Agent", type="primary", use_container_width=True)

        # ── Agent output area ─────────────────────────────────────────────────
        if run_btn:
            if task_type == "custom" and not user_input.strip():
                st.warning("Please enter your question first.")
                return

            from agent.loop import run_agent

            output_area  = st.empty()
            tool_area    = st.container()
            final_area   = st.empty()
            full_text    = ""
            tool_log     = []

            with st.spinner("Agent is working…"):
                for event in run_agent(
                    df         = df,
                    task_type  = task_type,
                    user_input = user_input,
                    max_iter   = max_iter,
                ):
                    etype = event.get("type")

                    if etype == "thinking":
                        output_area.caption(event["text"])

                    elif etype == "text_delta":
                        full_text += event["text"]
                        # Stream text progressively
                        output_area.markdown(
                            f'<div class="agent-bubble">{full_text}</div>',
                            unsafe_allow_html=True,
                        )

                    elif etype == "tool_call" and show_tools:
                        params_str = json.dumps(event["params"], indent=2)[:400]
                        tool_log.append(
                            f'<div class="tool-call">🔧 {event["tool"]}({params_str})</div>'
                        )
                        with tool_area:
                            st.markdown("".join(tool_log), unsafe_allow_html=True)

                    elif etype == "tool_result" and show_tools:
                        result_str = json.dumps(event["result"], indent=2,
                                                default=str)[:600]
                        tool_log.append(
                            f'<div class="tool-result">✓ {result_str}</div>'
                        )
                        with tool_area:
                            st.markdown("".join(tool_log), unsafe_allow_html=True)

                    elif etype == "done":
                        output_area.empty()
                        final_area.markdown(
                            f'<div class="agent-bubble">{full_text}</div>',
                            unsafe_allow_html=True,
                        )
                        st.success("✅ Analysis complete — see Research Log →")
                        st.rerun()   # refresh finding panel

                    elif etype == "error":
                        st.error(f"Agent error: {event['message']}")
                        break

        # ── Prior session output ──────────────────────────────────────────────
        st.markdown("---")
        st.markdown("### 📜 Session history")
        if "agent_history" not in st.session_state:
            st.session_state["agent_history"] = []

        # Save results to session history (populated in run_btn block above via rerun)
        history = st.session_state.get("agent_history", [])
        if not history:
            st.caption("Agent outputs from this session will appear here.")


if __name__ == "__main__":
    main()
