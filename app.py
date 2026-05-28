"""
app.py
======
Trading Analytics Platform — Streamlit entry point.

Upload any CSV or XLSX trade export → full descriptive + inferential analysis.
Supports 10+ brokers via auto-detection (NinjaTrader/Topstep is zero-config).

Run locally:
    streamlit run app.py

Deploy:
    Push to GitHub → connect to share.streamlit.io → set main file = app.py
"""

import io
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

warnings.filterwarnings("ignore")

# ── page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="Trading Analytics Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── ensure core package is importable when running from repo root ─────────────
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.ingest      import ingest, IngestResult
from core.etl         import enrich
from core.descriptive import build_summary, to_json
from core.inferential import run_all
from core.descriptive import _json_safe

# ─────────────────────────────────────────────────────────────────────────────
# THEME & CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
GREEN  = "#27AE60"
RED    = "#E74C3C"
GOLD   = "#F5A623"
TEAL   = "#00C2CB"
BLUE   = "#2980B9"
GREY   = "#607D8B"
BG     = "#0D1B2A"

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
code, .metric-value { font-family: 'DM Mono', monospace; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0D1B2A;
    border-right: 1px solid #1A2E45;
}
section[data-testid="stSidebar"] * { color: #B0BEC5 !important; }
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 { color: #00C2CB !important; }

/* Metric cards */
[data-testid="metric-container"] {
    background: #1A2E45;
    border: 1px solid #223554;
    border-radius: 8px;
    padding: 12px 16px;
}
[data-testid="metric-container"] label { color: #607D8B !important; font-size: 11px !important; }
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-family: 'DM Mono', monospace;
    font-size: 22px !important;
}

/* Tabs */
button[data-baseweb="tab"] {
    font-family: 'DM Sans', sans-serif;
    font-weight: 500;
    font-size: 13px;
}

/* Dataframes */
[data-testid="stDataFrame"] { border-radius: 8px; overflow: hidden; }

/* Headers */
h1 { color: #FFFFFF !important; font-weight: 600; letter-spacing: -0.5px; }
h2 { color: #E0E8F0 !important; font-weight: 500; }
h3 { color: #B0BEC5 !important; font-weight: 500; font-size: 14px !important;
     text-transform: uppercase; letter-spacing: 1px; }

/* Dividers */
hr { border-color: #1A2E45; }

/* Badges */
.badge-green { background:#0D2E1A; color:#27AE60; border:1px solid #27AE60;
               padding:2px 10px; border-radius:20px; font-size:12px; font-weight:500; }
.badge-red   { background:#2E0D0D; color:#E74C3C; border:1px solid #E74C3C;
               padding:2px 10px; border-radius:20px; font-size:12px; font-weight:500; }
.badge-amber { background:#2E1E00; color:#F5A623; border:1px solid #F5A623;
               padding:2px 10px; border-radius:20px; font-size:12px; font-weight:500; }
.badge-teal  { background:#002E2E; color:#00C2CB; border:1px solid #00C2CB;
               padding:2px 10px; border-radius:20px; font-size:12px; font-weight:500; }
.finding-card {
    background: #1A2E45; border-left: 3px solid #E74C3C;
    border-radius: 0 8px 8px 0; padding: 10px 14px;
    margin-bottom: 8px; font-size: 13px; line-height: 1.6;
}
.finding-card.ok  { border-left-color: #27AE60; }
.finding-card.warn{ border-left-color: #F5A623; }
.finding-card.info{ border-left-color: #00C2CB; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def fmt(v, prefix="$", suffix="", dec=2):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{prefix}{v:,.{dec}f}{suffix}"

def pct(v, dec=1):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{v*100:.{dec}f}%"

def sig_badge(p):
    if p is None: return ""
    if p < 0.001: return "🔴 ***"
    if p < 0.01:  return "🟠 **"
    if p < 0.05:  return "🟡 *"
    if p < 0.10:  return "⚪ ."
    return "✅ ns"

def color_val(v):
    if v is None: return "—"
    color = GREEN if v > 0 else (RED if v < 0 else GREY)
    return f"<span style='color:{color};font-weight:500'>{fmt(v)}</span>"

def plotly_dark():
    return dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#0D1B2A",
        font=dict(color="#B0BEC5", family="DM Sans"),
        xaxis=dict(gridcolor="#1A2E45", zerolinecolor="#1A2E45"),
        yaxis=dict(gridcolor="#1A2E45", zerolinecolor="#1A2E45"),
        margin=dict(l=40, r=40, t=40, b=40),
    )


# ─────────────────────────────────────────────────────────────────────────────
# CHARTS
# ─────────────────────────────────────────────────────────────────────────────

def equity_curve_chart(equity, daily_pnl=None):
    x = list(range(1, len(equity) + 1))
    colors = [GREEN if v >= 0 else RED for v in
              [equity[0]] + [equity[i]-equity[i-1] for i in range(1, len(equity))]]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=equity, mode="lines",
        line=dict(color=TEAL, width=2),
        fill="tozeroy",
        fillcolor="rgba(0,194,203,0.08)",
        name="Equity curve",
        hovertemplate="Trade %{x}<br>Net PnL: $%{y:,.0f}<extra></extra>",
    ))
    # Zero line
    fig.add_hline(y=0, line_dash="dash", line_color=GREY, line_width=1)
    fig.update_layout(**plotly_dark(), height=280,
                      title=dict(text="Equity Curve", font=dict(size=13)))
    return fig


def bar_chart(labels, values, title, height=280, color_by_sign=True):
    colors = [GREEN if v >= 0 else RED for v in values] if color_by_sign \
             else [TEAL] * len(values)
    fig = go.Figure(go.Bar(
        x=labels, y=values,
        marker_color=colors,
        hovertemplate="%{x}<br>$%{y:,.0f}<extra></extra>",
    ))
    fig.add_hline(y=0, line_dash="dash", line_color=GREY, line_width=1)
    fig.update_layout(**plotly_dark(), height=height,
                      title=dict(text=title, font=dict(size=13)),
                      showlegend=False)
    return fig


def hold_time_chart(da):
    wins_mean   = (da.get("wins") or {}).get("mean")
    losses_mean = (da.get("losses") or {}).get("mean")
    wins_med    = (da.get("wins") or {}).get("median")
    losses_med  = (da.get("losses") or {}).get("median")
    if not any([wins_mean, losses_mean]): return None
    cats = ["Wins — Mean", "Wins — Median", "Losses — Mean", "Losses — Median"]
    vals = [wins_mean or 0, wins_med or 0, losses_mean or 0, losses_med or 0]
    cols = [GREEN, GREEN, RED, RED]
    fig = go.Figure(go.Bar(
        x=vals, y=cats, orientation="h",
        marker_color=cols,
        hovertemplate="%{y}: %{x:.1f} min<extra></extra>",
    ))
    fig.update_layout(**plotly_dark(), height=240,
                      title=dict(text="Hold Time: Wins vs Losses (min)", font=dict(size=13)),
                      showlegend=False)
    return fig


def cusum_chart(cusum_data):
    pos = cusum_data.get("cusum_pos", [])
    neg = cusum_data.get("cusum_neg", [])
    thr = cusum_data.get("threshold_used", 0)
    if not pos: return None
    x = list(range(1, len(pos) + 1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=pos, mode="lines",
        line=dict(color=GREEN, width=1.5), name="CUSUM+",
        hovertemplate="Trade %{x}<br>%{y:.0f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=x, y=neg, mode="lines",
        line=dict(color=RED, width=1.5), name="CUSUM−",
        fill="tozeroy", fillcolor="rgba(231,76,60,0.08)",
        hovertemplate="Trade %{x}<br>%{y:.0f}<extra></extra>"))
    if thr:
        fig.add_hline(y=thr,  line_dash="dash", line_color=GOLD, line_width=1)
        fig.add_hline(y=-thr, line_dash="dash", line_color=GOLD, line_width=1)
    fig.update_layout(**plotly_dark(), height=260,
                      title=dict(text="CUSUM Regime Detection", font=dict(size=13)))
    return fig


def scatter_instruments(by_inst):
    rows = []
    for inst, b in by_inst.items():
        if not b or b.get("n_trades", 0) < 3: continue
        rows.append({
            "Instrument": inst,
            "Win Rate %":  round((b.get("win_rate") or 0) * 100, 1),
            "Profit Factor": b.get("profit_factor") or 0,
            "n_trades":    b.get("n_trades", 0),
            "Total PnL":   b.get("total_net_pnl", 0),
        })
    if not rows: return None
    df_s = pd.DataFrame(rows)
    fig = px.scatter(df_s, x="Win Rate %", y="Profit Factor",
                     size="n_trades", color="Total PnL",
                     text="Instrument",
                     color_continuous_scale=[[0,"#E74C3C"],[0.5,"#607D8B"],[1,"#27AE60"]],
                     size_max=40, height=320)
    fig.add_hline(y=1, line_dash="dash", line_color=GREY, line_width=1)
    fig.add_vline(x=50, line_dash="dash", line_color=GREY, line_width=1)
    fig.update_traces(textposition="top center",
                      textfont=dict(color="#FFFFFF", size=11))
    fig.update_layout(**plotly_dark(),
                      title=dict(text="Win Rate vs Profit Factor by Instrument",
                                 font=dict(size=13)),
                      coloraxis_colorbar=dict(title="Net PnL"))
    return fig


def daily_pnl_chart(daily_pnl):
    dates  = list(daily_pnl.keys())
    values = list(daily_pnl.values())
    colors = [GREEN if v >= 0 else RED for v in values]
    fig = go.Figure(go.Bar(
        x=dates, y=values, marker_color=colors,
        hovertemplate="%{x}<br>$%{y:,.0f}<extra></extra>",
    ))
    fig.add_hline(y=0, line_dash="dash", line_color=GREY, line_width=1)
    fig.update_layout(**plotly_dark(), height=260,
                      title=dict(text="Daily Net PnL", font=dict(size=13)),
                      showlegend=False,
                      xaxis=dict(tickangle=-45))
    return fig


def bootstrap_ci_chart(boot_by_inst):
    rows = []
    for inst, b in boot_by_inst.items():
        if not b or "estimate" not in b or b["estimate"] is None: continue
        rows.append({
            "inst":  inst,
            "est":   b["estimate"],
            "lo":    b.get("ci_lo") or 0,
            "hi":    b.get("ci_hi") or 0,
            "sig":   b.get("significant", False),
        })
    if not rows: return None
    rows.sort(key=lambda r: r["est"])
    fig = go.Figure()
    for r in rows:
        color = GREEN if r["est"] > 0 else RED
        fig.add_trace(go.Scatter(
            x=[r["lo"], r["hi"]], y=[r["inst"], r["inst"]],
            mode="lines", line=dict(color=color, width=6),
            showlegend=False,
            hovertemplate=f"{r['inst']}: CI [{r['lo']:.1f}, {r['hi']:.1f}]<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=[r["est"]], y=[r["inst"]], mode="markers",
            marker=dict(color=color, size=12, symbol="diamond"),
            showlegend=False,
            hovertemplate=f"{r['inst']}: estimate ${r['est']:.1f}<extra></extra>",
        ))
    fig.add_vline(x=0, line_dash="dash", line_color=GREY, line_width=1.5)
    fig.update_layout(**plotly_dark(), height=300,
                      title=dict(text="Bootstrap 95% CI — Expectancy by Instrument",
                                 font=dict(size=13)),
                      xaxis_title="Expected PnL per trade ($)")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

def render_sidebar():
    with st.sidebar:
        st.markdown("## 📊 Trading Analytics")
        st.markdown("---")

        uploaded = st.file_uploader(
            "Upload trade export",
            type=["csv", "xlsx", "xls"],
            help="Supports NinjaTrader/Topstep, Interactive Brokers, Tradovate, "
                 "TradeStation, Tastytrade, MetaTrader, Rithmic/Apex, Binance, Kraken, Webull.",
        )

        st.markdown("---")
        st.markdown("### ⚙️ Options")
        run_inf = st.checkbox(
            "Run inferential tests",
            value=True,
            help="Bootstrap CIs, MWU, KS, Ljung-Box, CUSUM, runs test, permutation tests. "
                 "Adds ~5 seconds for large datasets.",
        )
        merge_files = st.checkbox(
            "Merge with previous upload",
            value=False,
            help="Combine this file with the previously uploaded file before analysis.",
        )

        st.markdown("---")
        st.markdown("### 📁 Multiple files")
        st.caption("To combine files: upload file 1, tick 'Merge with previous upload', then upload file 2.")

        st.markdown("---")
        st.markdown(
            "<div style='font-size:11px;color:#3D5A73'>"
            "Trading Analytics Platform<br>"
            "Steps 1 · 2a · 2b complete<br>"
            "Step 3 (Streamlit app) ✓<br><br>"
            "Built on 199 live Topstep trades<br>"
            "Apr–May 2026"
            "</div>",
            unsafe_allow_html=True,
        )

    return uploaded, run_inf, merge_files


# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────

def tab_overview(summary, df):
    ov = summary["overall"]
    me = summary["meta"]
    da = summary["duration_analysis"]
    fa = summary["fee_analysis"]
    ra = summary["run_analysis"]

    # ── key metrics row ───────────────────────────────────────────────────────
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    total = ov.get("total_net_pnl") or 0
    c1.metric("Total Net PnL",  fmt(total),
              delta=None, delta_color="off")
    c2.metric("Win Rate",        pct(ov.get("win_rate")))
    c3.metric("Expectancy",      fmt(ov.get("expectancy_usd")),
              delta=None, delta_color="off")
    c4.metric("Profit Factor",   f"{ov.get('profit_factor') or 0:.3f}")
    c5.metric("Sharpe Ratio",    f"{ov.get('sharpe') or 0:.3f}")
    c6.metric("Max Drawdown",    fmt(ov.get("max_dd_usd")),
              delta=None, delta_color="off")

    st.markdown("---")

    col_l, col_r = st.columns([3, 2])

    with col_l:
        eq = summary.get("equity_curve", [])
        if eq:
            st.plotly_chart(equity_curve_chart(eq), use_container_width=True)

    with col_r:
        daily = summary.get("daily_pnl", {})
        if daily:
            st.plotly_chart(daily_pnl_chart(daily), use_container_width=True)

    # ── secondary metrics ─────────────────────────────────────────────────────
    st.markdown("---")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Trades",         ov.get("n_trades"))
    c2.metric("Trading Days",   me.get("trading_days"))
    c3.metric("Avg Win",        fmt(ov.get("avg_win")))
    c4.metric("Avg Loss",       fmt(ov.get("avg_loss")))
    c5.metric("Fee Drag",       fmt(fa.get("total_cost")))
    c6.metric("Max Win Streak", ra.get("max_win_streak"))

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Gross Profit",    fmt(ov.get("gross_profit")))
    c2.metric("Gross Loss",      fmt(ov.get("gross_loss")))
    c3.metric("Skewness",        f"{ov.get('skewness') or 0:.3f}")
    c4.metric("Kurtosis",        f"{ov.get('kurtosis') or 0:.3f}")
    c5.metric("Max Loss Streak", ra.get("max_loss_streak"))
    ratio = da.get("loss_to_win_hold_ratio")
    c6.metric("Hold Asymmetry",
              f"{ratio:.2f}×" if ratio else "—",
              delta="losses held longer" if ratio and ratio > 1.2 else None,
              delta_color="inverse")

    # ── period info ───────────────────────────────────────────────────────────
    st.markdown("---")
    st.caption(
        f"Period: **{me.get('data_from')}** → **{me.get('data_to')}**  ·  "
        f"Instruments: **{', '.join(me.get('instruments', []))}**  ·  "
        f"Directions: **{', '.join(me.get('directions', []))}**"
    )


def tab_instruments(summary):
    by_inst = summary.get("by_instrument", {})
    if not by_inst:
        st.info("No instrument breakdown available.")
        return

    # Scatter
    fig = scatter_instruments(by_inst)
    if fig:
        st.plotly_chart(fig, use_container_width=True)

    # Bar chart net PnL
    insts  = list(by_inst.keys())
    totals = [by_inst[i].get("total_net_pnl", 0) or 0 for i in insts]
    order  = sorted(range(len(totals)), key=lambda x: totals[x])
    st.plotly_chart(
        bar_chart([insts[i] for i in order], [totals[i] for i in order],
                  "Net PnL by Instrument"),
        use_container_width=True,
    )

    # Table
    rows = []
    for inst, b in by_inst.items():
        if not b: continue
        rows.append({
            "Instrument":    inst,
            "Trades":        b.get("n_trades"),
            "Net PnL":       b.get("total_net_pnl"),
            "Win Rate":      pct(b.get("win_rate")),
            "Profit Factor": b.get("profit_factor"),
            "Avg Win":       b.get("avg_win"),
            "Avg Loss":      b.get("avg_loss"),
            "Sharpe":        b.get("sharpe"),
            "Max DD ($)":    b.get("max_dd_usd"),
            "Skewness":      b.get("skewness"),
        })
    df_t = pd.DataFrame(rows).sort_values("Net PnL", ascending=False)
    st.dataframe(
        df_t.style.format({
            "Net PnL": "${:,.2f}",
            "Profit Factor": "{:.3f}",
            "Avg Win": "${:,.2f}",
            "Avg Loss": "${:,.2f}",
            "Sharpe": "{:.3f}",
            "Max DD ($)": "${:,.2f}",
            "Skewness": "{:.3f}",
        }).background_gradient(subset=["Net PnL"], cmap="RdYlGn"),
        use_container_width=True, hide_index=True,
    )


def tab_timing(summary):
    col_l, col_r = st.columns(2)

    # By day
    with col_l:
        by_day = summary.get("by_day_of_week", {})
        day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday"]
        days   = [d for d in day_order if d in by_day]
        d_vals = [by_day[d].get("total_net_pnl", 0) or 0 for d in days]
        d_wr   = [pct(by_day[d].get("win_rate")) for d in days]
        st.plotly_chart(bar_chart(days, d_vals, "Net PnL by Day of Week"),
                        use_container_width=True)

        df_day = pd.DataFrame({
            "Day": days, "Net PnL": d_vals,
            "Win Rate": d_wr,
            "Trades": [by_day[d].get("n_trades") for d in days],
        })
        st.dataframe(df_day.style.format({"Net PnL": "${:,.2f}"}),
                     use_container_width=True, hide_index=True)

    # By hour
    with col_r:
        by_hr = summary.get("by_hour_bin", {})
        hours  = sorted(by_hr.keys())
        h_vals = [by_hr[h].get("total_net_pnl", 0) or 0 for h in hours]
        st.plotly_chart(bar_chart(hours, h_vals, "Net PnL by Hour Bin (entry time)"),
                        use_container_width=True)

        df_hr = pd.DataFrame({
            "Hour (local)": hours, "Net PnL": h_vals,
            "Win Rate": [pct(by_hr[h].get("win_rate")) for h in hours],
            "Trades": [by_hr[h].get("n_trades") for h in hours],
            "PF": [by_hr[h].get("profit_factor") for h in hours],
        })
        st.dataframe(
            df_hr.style.format({"Net PnL": "${:,.2f}", "PF": "{:.3f}"}),
            use_container_width=True, hide_index=True,
        )

    # Hold times
    st.markdown("---")
    da = summary.get("duration_analysis", {})
    fig_hold = hold_time_chart(da)
    if fig_hold:
        col_h, col_s = st.columns([2, 1])
        with col_h:
            st.plotly_chart(fig_hold, use_container_width=True)
        with col_s:
            ratio = da.get("loss_to_win_hold_ratio")
            st.metric("Loss/Win Hold Ratio", f"{ratio:.2f}×" if ratio else "—")
            st.metric("Avg Win Hold",   f"{(da.get('wins') or {}).get('mean') or 0:.1f} min")
            st.metric("Avg Loss Hold",  f"{(da.get('losses') or {}).get('mean') or 0:.1f} min")
            st.metric("PnL/min (mean)", fmt(da.get("pnl_per_min_mean"), dec=3))


def tab_direction(summary):
    by_dir = summary.get("by_direction", {})
    if not by_dir:
        st.info("No direction breakdown available.")
        return

    dirs   = list(by_dir.keys())
    totals = [by_dir[d].get("total_net_pnl", 0) or 0 for d in dirs]
    st.plotly_chart(bar_chart(dirs, totals, "Net PnL by Direction"),
                    use_container_width=True)

    rows = []
    for d, b in by_dir.items():
        rows.append({
            "Direction": d, "Trades": b.get("n_trades"),
            "Net PnL": b.get("total_net_pnl"),
            "Win Rate": pct(b.get("win_rate")),
            "Profit Factor": b.get("profit_factor"),
            "Avg Win": b.get("avg_win"), "Avg Loss": b.get("avg_loss"),
            "Sharpe": b.get("sharpe"),
        })
    st.dataframe(
        pd.DataFrame(rows).style.format({
            "Net PnL": "${:,.2f}", "Profit Factor": "{:.3f}",
            "Avg Win": "${:,.2f}", "Avg Loss": "${:,.2f}", "Sharpe": "{:.3f}",
        }),
        use_container_width=True, hide_index=True,
    )


def tab_inferential(inf):
    st.markdown("### Bootstrap Confidence Intervals")

    # Overall
    bc = inf.get("bootstrap", {})
    exp = bc.get("expectancy", {})
    shr = bc.get("sharpe", {})

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Expectancy estimate", fmt(exp.get("estimate")))
    c2.metric("95% CI lower",        fmt(exp.get("ci_lo")))
    c3.metric("95% CI upper",        fmt(exp.get("ci_hi")))
    c4.metric("CI excludes zero",    "✅ Yes" if exp.get("significant") else "❌ No")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Sharpe estimate",  f"{shr.get('estimate') or 0:.3f}")
    c2.metric("Sharpe CI lower",  f"{shr.get('ci_lo') or 0:.3f}")
    c3.metric("Sharpe CI upper",  f"{shr.get('ci_hi') or 0:.3f}")
    c4.metric("Sharpe significant", "✅ Yes" if shr.get("significant") else "❌ No")

    # By instrument CI chart
    boot_inst = bc.get("by_instrument", {})
    fig_boot  = bootstrap_ci_chart(boot_inst)
    if fig_boot:
        st.plotly_chart(fig_boot, use_container_width=True)

    st.markdown("---")
    st.markdown("### All 10 Statistical Tests")

    def test_row(name, result, p_key="p_value", interp_key="interpretation"):
        if not result or "note" in result:
            st.markdown(f"**{name}** — ⚠️ {result.get('note', 'skipped') if result else 'no data'}")
            return
        p = result.get(p_key)
        interp = result.get(interp_key, "")
        stat_str = ""
        for k in ["W_stat","U_stat","H_stat","KS_stat","LB_stat","DW_stat","z_stat","observed_diff","observed_runs"]:
            if k in result and result[k] is not None:
                stat_str = f" | {k}={result[k]:.3f}"
                break
        col1, col2 = st.columns([1, 4])
        with col1:
            if p is not None:
                st.markdown(f"**p = {p:.4f}** {sig_badge(p)}")
            else:
                st.markdown(f"**{name}**")
        with col2:
            st.markdown(f"*{interp}*{stat_str}")

    # Normality
    nm = inf.get("normality", {})
    st.markdown("**1. Normality (Shapiro-Wilk + D'Agostino-Pearson)**")
    sw = nm.get("shapiro_wilk", {})
    dp = nm.get("dagostino_pearson", {})
    if sw:
        test_row("Shapiro-Wilk", sw, p_key="p_value",
                 interp_key="p_value")
        st.caption(f"SW: W={sw.get('W_stat',0):.4f}, p={sw.get('p_value',0):.2e} {sig_badge(sw.get('p_value'))}")
    if dp:
        st.caption(f"D'AP: K²={dp.get('K2_stat',0):.3f}, p={dp.get('p_value',0):.2e} {sig_badge(dp.get('p_value'))}")
    st.caption(nm.get("interpretation", ""))

    st.markdown("---")

    # Hold time
    ht = inf.get("hold_time", {})
    st.markdown("**2. Hold-Time Asymmetry (Mann-Whitney U)**")
    mwu_ht = ht.get("mann_whitney", {})
    if mwu_ht and "p_value" in mwu_ht:
        c1, c2, c3 = st.columns(3)
        c1.metric("p-value", f"{mwu_ht['p_value']:.4f} {sig_badge(mwu_ht['p_value'])}")
        c2.metric("Effect size r", f"{mwu_ht.get('effect_size_r',0):.3f} ({mwu_ht.get('effect_label','')})")
        c3.metric("Reject H₀", "Yes" if mwu_ht.get("reject_h0") else "No")
        st.caption(mwu_ht.get("interpretation", ""))

    st.markdown("---")

    # MWU by instrument
    mwu_i = inf.get("mwu_by_instrument", {})
    st.markdown("**3. Mann-Whitney U — Per Instrument vs Rest**")
    rows_mwu = []
    for inst, b in mwu_i.items():
        if not b or "p_value" not in b: continue
        rows_mwu.append({
            "Instrument": inst,
            "n": b.get("n"),
            "Median PnL": b.get("median_pnl"),
            "p-value": b.get("p_value"),
            "Significance": sig_badge(b.get("p_value")),
            "Effect r": b.get("effect_size_r"),
            "Reject H₀": "Yes" if b.get("reject_h0") else "No",
        })
    if rows_mwu:
        st.dataframe(
            pd.DataFrame(rows_mwu).sort_values("p-value").style.format({
                "Median PnL": "${:,.2f}", "p-value": "{:.4f}", "Effect r": "{:.3f}",
            }),
            use_container_width=True, hide_index=True,
        )

    st.markdown("---")

    # Kruskal-Wallis
    kw = inf.get("kruskal_wallis", {})
    st.markdown("**4. Kruskal-Wallis Omnibus Tests**")
    for label, key in [("By Instrument","by_instrument"),("By Day","by_day"),("By Hour","by_hour")]:
        r = kw.get(key, {})
        if not r: continue
        p = r.get("p_value")
        st.caption(f"{label}: H={r.get('H_stat',0):.2f}, df={r.get('df',0)}, p={p:.4f} {sig_badge(p)} — {r.get('interpretation','')}")

    st.markdown("---")

    # Permutation tests
    pm = inf.get("permutation", {})
    st.markdown("**5. Permutation Tests**")
    for label, key in [("Long vs Short","long_vs_short"),("MNQM6 vs ZFM6","mnqm6_vs_zfm6")]:
        r = pm.get(key, {})
        if not r or "note" in r: continue
        p = r.get("p_value")
        st.caption(f"{label}: diff=${r.get('observed_diff',0):.2f}, p={p:.4f} {sig_badge(p)} — {r.get('interpretation','')}")

    st.markdown("---")

    # Ljung-Box
    lb = inf.get("ljung_box", {})
    st.markdown("**6. Ljung-Box Autocorrelation**")
    if "results" in lb:
        for lag_key, lr in lb["results"].items():
            p = lr.get("p_value")
            st.caption(f"Lag {lr.get('lag')}: LB={lr.get('LB_stat',0):.2f}, p={p:.4f} {sig_badge(p)}")
        st.caption(lb.get("interpretation",""))
    elif "note" in lb:
        st.caption(lb["note"])

    st.markdown("---")

    # DW
    dw = inf.get("durbin_watson", {})
    st.markdown("**7. Durbin-Watson (First-Order Autocorrelation)**")
    st.caption(f"DW = {dw.get('DW_stat',0):.4f} — {dw.get('interpretation','')}")

    st.markdown("---")

    # Runs test
    ru = inf.get("runs_test", {})
    st.markdown("**8. Wald-Wolfowitz Runs Test**")
    if "p_value" in ru:
        p = ru.get("p_value")
        c1, c2, c3 = st.columns(3)
        c1.metric("Observed runs", ru.get("observed_runs"))
        c2.metric("Expected runs", f"{ru.get('expected_runs',0):.1f}")
        c3.metric("p-value", f"{p:.4f} {sig_badge(p)}")
        st.caption(ru.get("interpretation",""))

    st.markdown("---")

    # CUSUM
    cusum = inf.get("cusum", {})
    st.markdown("**9. CUSUM Regime Detection**")
    c1, c2, c3 = st.columns(3)
    c1.metric("Current regime",     cusum.get("current_regime","—").upper())
    c2.metric("First shift at trade", cusum.get("first_regime_shift_trade","—"))
    c3.metric("Downward signals",   len(cusum.get("change_points_down",[])))
    st.caption(cusum.get("interpretation",""))
    fig_cs = cusum_chart(cusum)
    if fig_cs:
        st.plotly_chart(fig_cs, use_container_width=True)


def tab_raw(df):
    st.markdown(f"**{len(df)} trades** · {len(df.columns)} columns")

    # Filters
    c1, c2, c3 = st.columns(3)
    insts = ["All"] + sorted(df["instrument"].dropna().unique().tolist())
    dirs  = ["All"] + sorted(df["direction"].dropna().unique().tolist())
    outs  = ["All"] + sorted(df["outcome"].dropna().unique().tolist())
    sel_inst = c1.selectbox("Instrument", insts)
    sel_dir  = c2.selectbox("Direction",  dirs)
    sel_out  = c3.selectbox("Outcome",    outs)

    fdf = df.copy()
    if sel_inst != "All": fdf = fdf[fdf["instrument"] == sel_inst]
    if sel_dir  != "All": fdf = fdf[fdf["direction"]  == sel_dir]
    if sel_out  != "All": fdf = fdf[fdf["outcome"]    == sel_out]

    cols = ["trade_index","trade_date","instrument","direction","entry_price",
            "exit_price","net_pnl","outcome","duration_minutes","hour_bin","day_name"]
    show_cols = [c for c in cols if c in fdf.columns]
    st.dataframe(
        fdf[show_cols].style.format({
            "net_pnl": "${:,.2f}",
            "entry_price": "{:,.2f}",
            "exit_price": "{:,.2f}",
            "duration_minutes": "{:.1f}",
        }).applymap(
            lambda v: f"color:{GREEN}" if v == "win" else
                      (f"color:{RED}" if v == "loss" else ""),
            subset=["outcome"],
        ),
        use_container_width=True, hide_index=True,
    )

    # Download
    csv = fdf.to_csv(index=False).encode()
    st.download_button("⬇ Download filtered CSV", csv,
                       "filtered_trades.csv", "text/csv")


def tab_json(summary, inf):
    st.markdown("### JSON Summary Payload")
    st.markdown("Copy this and paste directly into a Claude conversation for deep statistical dialogue.")
    combined = {"descriptive": summary, "inferential": _json_safe(inf)}
    js = json.dumps(combined, indent=2, default=str)
    st.code(js[:8000] + ("\n... (truncated for display)" if len(js) > 8000 else ""),
            language="json")
    st.download_button("⬇ Download full summary.json",
                       js.encode(), "summary.json", "application/json")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    uploaded, run_inf, merge_files = render_sidebar()

    # ── landing page ──────────────────────────────────────────────────────────
    if uploaded is None:
        st.markdown("""
# 📊 Trading Analytics Platform

Upload your broker export in the sidebar to get started.

**Supported brokers** — auto-detected, zero configuration:
- NinjaTrader / Topstep (CSV)
- Interactive Brokers (Flex Query)
- Tradovate · TradeStation · Tastytrade
- MetaTrader 4/5 · Rithmic / Apex
- Binance · Kraken · Webull

**What you get:**
- Full descriptive statistics — equity curve, by instrument, by direction, by day, by hour, hold-time analysis, fee drag
- 10 inferential tests — bootstrap CIs, Mann-Whitney U, Kruskal-Wallis, permutation tests, Ljung-Box, CUSUM, runs test
- Downloadable JSON payload to use with Claude for deeper analysis
        """)
        return

    # ── load & process ────────────────────────────────────────────────────────
    with st.spinner("Detecting broker and loading file…"):
        try:
            # Save upload to temp file (ingest needs a path)
            suffix = Path(uploaded.name).suffix
            tmp    = Path(f"/tmp/upload{suffix}")
            tmp.write_bytes(uploaded.read())

            result = ingest(tmp)
            df_new = enrich(result.df)

        except Exception as e:
            st.error(f"Failed to load file: {e}")
            return

    # Merge if requested
    if merge_files and "prev_df" in st.session_state:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = pd.concat([st.session_state["prev_df"], df_new],
                           ignore_index=True).sort_values("entered_at_dt").reset_index(drop=True)
            df["trade_index"] = range(1, len(df)+1)
        st.success(f"Merged {len(st.session_state['prev_df'])} + {len(df_new)} = **{len(df)} trades**")
    else:
        df = df_new

    st.session_state["prev_df"] = df

    # Broker info banner
    if result.warnings:
        for w in result.warnings:
            st.warning(w)
    st.caption(
        f"🔍 Detected broker: **{result.broker}** "
        f"({result.confidence:.0%} confidence)  ·  "
        f"**{len(df)} trades** loaded"
    )

    # ── compute ───────────────────────────────────────────────────────────────
    with st.spinner("Computing descriptive statistics…"):
        summary = build_summary(df)

    inf = {}
    if run_inf:
        with st.spinner("Running 10 inferential tests (bootstrap takes ~5 sec)…"):
            inf = run_all(df)

    # ── tabs ──────────────────────────────────────────────────────────────────
    tabs = st.tabs([
        "📈 Overview",
        "🎯 Instruments",
        "⏰ Timing",
        "↔️ Direction",
        "🔬 Inferential",
        "📋 Raw Data",
        "📦 JSON Export",
    ])

    with tabs[0]: tab_overview(summary, df)
    with tabs[1]: tab_instruments(summary)
    with tabs[2]: tab_timing(summary)
    with tabs[3]: tab_direction(summary)
    with tabs[4]:
        if inf:
            tab_inferential(inf)
        else:
            st.info("Enable 'Run inferential tests' in the sidebar and reload.")
    with tabs[5]: tab_raw(df)
    with tabs[6]: tab_json(summary, inf)


if __name__ == "__main__":
    main()
