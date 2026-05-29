"""
app.py  —  Trading Analytics Platform
======================================
Streamlit entry point. Upload any CSV/XLSX broker export to get full
descriptive + inferential analysis.

Compatible with:
  - Python 3.10 – 3.14
  - pandas 2.x (no background_gradient / applymap)
  - plotly 5.x  (no xaxis= in update_layout kwargs)
  - Streamlit 1.35+

Run locally:   streamlit run app.py
Deploy:        share.streamlit.io  →  main file = app.py
"""

# ── std-lib first ─────────────────────────────────────────────────────────────
import io, json, sys, tempfile, warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# ── third-party ───────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# ── page config  (MUST be the very first Streamlit call) ─────────────────────
st.set_page_config(
    page_title="Trading Analytics Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── make core importable when running from repo root ─────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.ingest      import ingest
from core.etl         import enrich
from core.descriptive import build_summary, to_json, _json_safe
from core.inferential import run_all

# ═════════════════════════════════════════════════════════════════════════════
# THEME
# ═════════════════════════════════════════════════════════════════════════════
G   = "#27AE60"   # green / win
R   = "#E74C3C"   # red   / loss
GLD = "#F5A623"   # gold  / warning
TEL = "#00C2CB"   # teal  / accent
GRY = "#607D8B"   # grey  / neutral

st.markdown("""
<style>
[data-testid="metric-container"] {
    background:#1A2E45; border:1px solid #223554;
    border-radius:8px; padding:12px 16px;
}
[data-testid="metric-container"] label {
    color:#607D8B !important; font-size:11px !important;
}
section[data-testid="stSidebar"] { background:#0D1B2A; }
h3 { color:#B0BEC5 !important; font-size:13px !important;
     text-transform:uppercase; letter-spacing:1px; }
hr { border-color:#1A2E45; }
</style>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# SAFE FORMATTERS
# ═════════════════════════════════════════════════════════════════════════════

def _f(v, pre="$", dec=2):
    """Format a numeric value safely — returns '—' for None/NaN."""
    try:
        if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
            return "—"
        return f"{pre}{v:,.{dec}f}"
    except Exception:
        return "—"

def _pct(v, dec=1):
    try:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "—"
        return f"{v*100:.{dec}f}%"
    except Exception:
        return "—"

def _sig(p):
    """Significance stars."""
    if p is None: return ""
    try:
        if p < 0.001: return "★★★"
        if p < 0.01:  return "★★"
        if p < 0.05:  return "★"
        if p < 0.10:  return "·"
        return "ns"
    except Exception:
        return ""

def _safe_float(v):
    try:
        f = float(v)
        return None if (np.isnan(f) or np.isinf(f)) else f
    except Exception:
        return None


# ═════════════════════════════════════════════════════════════════════════════
# PLOTLY BASE  —  no xaxis/yaxis keys (avoids kwarg collision in plotly 5.x)
# ═════════════════════════════════════════════════════════════════════════════

def _base_layout(height=300, title=""):
    return dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#0D1B2A",
        font=dict(color="#B0BEC5", family="sans-serif", size=12),
        margin=dict(l=50, r=120, t=45, b=45),
        height=height,
        title=dict(text=title, font=dict(size=13), x=0),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
    )

def _style_axes(fig, x_tickangle=0):
    """Apply grid/zero-line style via update_xaxes/update_yaxes — never via layout."""
    fig.update_xaxes(
        gridcolor="#1A2E45", zerolinecolor="#1A2E45",
        tickangle=x_tickangle, tickfont=dict(size=11),
    )
    fig.update_yaxes(
        gridcolor="#1A2E45", zerolinecolor="#1A2E45",
        tickfont=dict(size=11),
    )
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# CHART BUILDERS
# ═════════════════════════════════════════════════════════════════════════════

def chart_equity(equity: list) -> go.Figure:
    x = list(range(1, len(equity) + 1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=equity, mode="lines", name="Equity",
        line=dict(color=TEL, width=2),
        fill="tozeroy", fillcolor="rgba(0,194,203,0.07)",
        hovertemplate="Trade %{x}<br>Net PnL: $%{y:,.0f}<extra></extra>",
    ))
    fig.add_hline(y=0, line_dash="dash", line_color=GRY, line_width=1)
    fig.update_layout(**_base_layout(280, "Equity Curve"), showlegend=False)
    return _style_axes(fig)


def chart_bars(labels: list, values: list, title: str,
               height=280, color_sign=True, x_tickangle=0) -> go.Figure:
    colors = [G if v >= 0 else R for v in values] if color_sign else [TEL]*len(values)
    fig = go.Figure(go.Bar(
        x=labels, y=values, marker_color=colors,
        hovertemplate="%{x}<br>$%{y:,.0f}<extra></extra>",
    ))
    fig.add_hline(y=0, line_dash="dash", line_color=GRY, line_width=1)
    fig.update_layout(**_base_layout(height, title), showlegend=False)
    return _style_axes(fig, x_tickangle=x_tickangle)


def chart_daily(daily_pnl: dict) -> go.Figure:
    dates  = list(daily_pnl.keys())
    values = []
    for v in daily_pnl.values():
        try:
            f = float(v)
            values.append(0.0 if (np.isnan(f) or np.isinf(f)) else f)
        except Exception:
            values.append(0.0)
    return chart_bars(dates, values, "Daily Net PnL", height=260, x_tickangle=-45)


def chart_hold_time(da: dict) -> go.Figure | None:
    wins_m  = _safe_float((da.get("wins")   or {}).get("mean"))
    wins_md = _safe_float((da.get("wins")   or {}).get("median"))
    loss_m  = _safe_float((da.get("losses") or {}).get("mean"))
    loss_md = _safe_float((da.get("losses") or {}).get("median"))
    if not any([wins_m, loss_m]):
        return None
    cats = ["Wins · Mean", "Wins · Median", "Losses · Mean", "Losses · Median"]
    vals = [wins_m or 0, wins_md or 0, loss_m or 0, loss_md or 0]
    cols = [G, G, R, R]
    fig = go.Figure(go.Bar(
        x=vals, y=cats, orientation="h", marker_color=cols,
        hovertemplate="%{y}: %{x:.1f} min<extra></extra>",
    ))
    fig.update_layout(**_base_layout(240, "Hold Time — Wins vs Losses (min)"),
                      showlegend=False)
    return _style_axes(fig)


def chart_scatter_inst(by_inst: dict) -> go.Figure | None:
    rows = []
    for inst, b in by_inst.items():
        if not b or (b.get("n_trades") or 0) < 3:
            continue
        rows.append({
            "Instrument": inst,
            "Win Rate %": round((_safe_float(b.get("win_rate")) or 0) * 100, 1),
            "Profit Factor": _safe_float(b.get("profit_factor")) or 0,
            "n": b.get("n_trades", 1),
            "PnL": _safe_float(b.get("total_net_pnl")) or 0,
        })
    if not rows:
        return None
    df_s = pd.DataFrame(rows)
    fig = px.scatter(
        df_s, x="Win Rate %", y="Profit Factor",
        size="n", color="PnL", text="Instrument",
        color_continuous_scale=[[0, R], [0.5, GRY], [1, G]],
        size_max=40, height=320,
    )
    fig.add_hline(y=1,  line_dash="dash", line_color=GRY, line_width=1)
    fig.add_vline(x=50, line_dash="dash", line_color=GRY, line_width=1)
    fig.update_traces(textposition="top center",
                      textfont=dict(color="#FFFFFF", size=11))
    fig.update_layout(
        **_base_layout(320, "Win Rate vs Profit Factor by Instrument"),
        coloraxis_colorbar=dict(title="Net PnL", tickfont=dict(size=10)),
    )
    return _style_axes(fig)


def chart_cusum(cusum: dict) -> go.Figure | None:
    pos = cusum.get("cusum_pos", [])
    neg = cusum.get("cusum_neg", [])
    if not pos:
        return None
    thr = _safe_float(cusum.get("threshold_used")) or 0
    x = list(range(1, len(pos) + 1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=pos, mode="lines", name="CUSUM+",
        line=dict(color=G, width=1.5),
        hovertemplate="Trade %{x}<br>%{y:.0f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=x, y=neg, mode="lines", name="CUSUM−",
        line=dict(color=R, width=1.5),
        fill="tozeroy", fillcolor="rgba(231,76,60,0.08)",
        hovertemplate="Trade %{x}<br>%{y:.0f}<extra></extra>",
    ))
    if thr:
        fig.add_hline(y= thr, line_dash="dash", line_color=GLD, line_width=1)
        fig.add_hline(y=-thr, line_dash="dash", line_color=GLD, line_width=1)
    fig.update_layout(**_base_layout(260, "CUSUM Regime Detection"))
    return _style_axes(fig)


def chart_bootstrap_ci(boot_by_inst: dict) -> go.Figure | None:
    rows = []
    for inst, b in boot_by_inst.items():
        if not b or b.get("estimate") is None:
            continue
        est = _safe_float(b.get("estimate"))
        lo  = _safe_float(b.get("ci_lo"))
        hi  = _safe_float(b.get("ci_hi"))
        if est is None or lo is None or hi is None:
            continue
        rows.append({"inst": inst, "est": est, "lo": lo, "hi": hi})
    if not rows:
        return None
    rows.sort(key=lambda r: r["est"])
    fig = go.Figure()
    for r in rows:
        col = G if r["est"] > 0 else R
        fig.add_trace(go.Scatter(
            x=[r["lo"], r["hi"]], y=[r["inst"], r["inst"]],
            mode="lines", line=dict(color=col, width=7),
            showlegend=False,
            hovertemplate=f"{r['inst']}: CI [{r['lo']:.1f}, {r['hi']:.1f}]<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=[r["est"]], y=[r["inst"]], mode="markers",
            marker=dict(color=col, size=12, symbol="diamond"),
            showlegend=False,
            hovertemplate=f"{r['inst']}: ${r['est']:.1f}<extra></extra>",
        ))
    fig.add_vline(x=0, line_dash="dash", line_color=GRY, line_width=1.5)
    fig.update_layout(**_base_layout(300, "Bootstrap 95% CI — Expectancy by Instrument"))
    fig.update_xaxes(gridcolor="#1A2E45", zerolinecolor="#1A2E45",
                     title_text="Expected PnL per trade ($)", tickfont=dict(size=11))
    fig.update_yaxes(gridcolor="#1A2E45", zerolinecolor="#1A2E45", tickfont=dict(size=11))
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# SAFE DATAFRAME STYLING  —  no background_gradient, no applymap
# ═════════════════════════════════════════════════════════════════════════════

def _colour_pnl_col(col: pd.Series) -> list:
    """Return list of CSS strings colouring each cell by sign. No matplotlib."""
    out = []
    for v in col:
        try:
            fv = float(v)
            if np.isnan(fv) or np.isinf(fv):
                out.append("")
            elif fv > 0:
                out.append("background-color:#0D2E1A; color:#27AE60")
            elif fv < 0:
                out.append("background-color:#2E0D0D; color:#E74C3C")
            else:
                out.append("")
        except Exception:
            out.append("")
    return out

def _colour_outcome_col(col: pd.Series) -> list:
    """Colour the outcome column strings."""
    out = []
    for v in col:
        s = str(v).lower()
        if s == "win":
            out.append("color:#27AE60; font-weight:500")
        elif s == "loss":
            out.append("color:#E74C3C; font-weight:500")
        else:
            out.append("")
    return out


# ═════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═════════════════════════════════════════════════════════════════════════════

def render_sidebar():
    with st.sidebar:
        st.markdown("## 📊 Trading Analytics")
        st.markdown("---")
        uploaded = st.file_uploader(
            "Upload trade export",
            type=["csv", "xlsx", "xls"],
            help=(
                "Auto-detects: NinjaTrader/Topstep, Interactive Brokers, "
                "Tradovate, TradeStation, Tastytrade, MetaTrader 4/5, "
                "Rithmic/Apex, Binance, Kraken, Webull."
            ),
        )
        st.markdown("---")
        st.markdown("### ⚙️ Options")
        run_inf = st.checkbox(
            "Run inferential tests",
            value=True,
            help="10 tests: bootstrap CIs, MWU, KS, Ljung-Box, CUSUM, runs. ~5 sec.",
        )
        merge = st.checkbox(
            "Merge with previous upload",
            value=False,
            help="Combines this file with the previously uploaded file.",
        )
        st.markdown("---")
        st.caption(
            "Trading Analytics Platform  \n"
            "Steps 1 · 2a · 2b · 3 complete  \n"
            "Built on 199 live Topstep trades"
        )
    return uploaded, run_inf, merge


# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ═════════════════════════════════════════════════════════════════════════════

def tab_overview(summary: dict, df: pd.DataFrame):
    ov = summary.get("overall", {})
    me = summary.get("meta", {})
    da = summary.get("duration_analysis", {})
    fa = summary.get("fee_analysis", {})
    ra = summary.get("run_analysis", {})

    # Row 1 — primary KPIs
    c = st.columns(6)
    c[0].metric("Total Net PnL",   _f(ov.get("total_net_pnl")))
    c[1].metric("Win Rate",        _pct(ov.get("win_rate")))
    c[2].metric("Expectancy",      _f(ov.get("expectancy_usd")))
    c[3].metric("Profit Factor",   _f(ov.get("profit_factor"), pre="", dec=3))
    c[4].metric("Sharpe Ratio",    _f(ov.get("sharpe"), pre="", dec=3))
    c[5].metric("Max Drawdown",    _f(ov.get("max_dd_usd")))

    st.markdown("---")

    # Charts row
    col_l, col_r = st.columns([3, 2])
    with col_l:
        eq = summary.get("equity_curve", [])
        if eq:
            st.plotly_chart(chart_equity(eq), use_container_width=True)
    with col_r:
        dp = summary.get("daily_pnl", {})
        if dp:
            st.plotly_chart(chart_daily(dp), use_container_width=True)

    st.markdown("---")

    # Row 2 — secondary KPIs
    c = st.columns(6)
    c[0].metric("Trades",          ov.get("n_trades", "—"))
    c[1].metric("Trading Days",    me.get("trading_days", "—"))
    c[2].metric("Avg Win",         _f(ov.get("avg_win")))
    c[3].metric("Avg Loss",        _f(ov.get("avg_loss")))
    c[4].metric("Fee Drag",        _f(fa.get("total_cost")))
    c[5].metric("Best Trade",      _f(ov.get("best_trade")))

    c = st.columns(6)
    c[0].metric("Gross Profit",    _f(ov.get("gross_profit")))
    c[1].metric("Gross Loss",      _f(ov.get("gross_loss")))
    c[2].metric("Skewness",        _f(ov.get("skewness"), pre="", dec=3))
    c[3].metric("Kurtosis",        _f(ov.get("kurtosis"), pre="", dec=3))
    c[4].metric("Max Win Streak",  ra.get("max_win_streak", "—"))
    ratio = _safe_float(da.get("loss_to_win_hold_ratio"))
    c[5].metric(
        "Hold Asymmetry",
        f"{ratio:.2f}×" if ratio else "—",
        delta="losses held longer" if ratio and ratio > 1.2 else None,
        delta_color="inverse",
    )

    st.markdown("---")
    st.caption(
        f"Period: **{me.get('data_from')}** → **{me.get('data_to')}**  ·  "
        f"Instruments: **{', '.join(me.get('instruments', []))}**  ·  "
        f"Directions: **{', '.join(me.get('directions', []))}**"
    )


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — INSTRUMENTS
# ═════════════════════════════════════════════════════════════════════════════

def tab_instruments(summary: dict):
    by_inst = summary.get("by_instrument", {})
    if not by_inst:
        st.info("No instrument data available.")
        return

    # Scatter
    fig_sc = chart_scatter_inst(by_inst)
    if fig_sc:
        st.plotly_chart(fig_sc, use_container_width=True)

    # Bar — sorted by PnL
    insts  = list(by_inst.keys())
    totals = [_safe_float((by_inst[i] or {}).get("total_net_pnl")) or 0 for i in insts]
    order  = sorted(range(len(totals)), key=lambda x: totals[x])
    st.plotly_chart(
        chart_bars([insts[i] for i in order], [totals[i] for i in order],
                   "Net PnL by Instrument"),
        use_container_width=True,
    )

    # Table — pure styling, no matplotlib
    rows = []
    for inst, b in by_inst.items():
        if not b:
            continue
        rows.append({
            "Instrument":    inst,
            "Trades":        b.get("n_trades"),
            "Net PnL":       _safe_float(b.get("total_net_pnl")),
            "Win Rate":      _pct(b.get("win_rate")),
            "Profit Factor": _safe_float(b.get("profit_factor")),
            "Avg Win ($)":   _safe_float(b.get("avg_win")),
            "Avg Loss ($)":  _safe_float(b.get("avg_loss")),
            "Sharpe":        _safe_float(b.get("sharpe")),
            "Max DD ($)":    _safe_float(b.get("max_dd_usd")),
            "Skewness":      _safe_float(b.get("skewness")),
        })
    if not rows:
        return

    df_t = pd.DataFrame(rows).sort_values("Net PnL", ascending=False)

    # Safe number formatting — convert None to NaN first
    fmt_cols = {
        "Net PnL": "${:,.2f}", "Profit Factor": "{:.3f}",
        "Avg Win ($)": "${:,.2f}", "Avg Loss ($)": "${:,.2f}",
        "Sharpe": "{:.3f}", "Max DD ($)": "${:,.2f}", "Skewness": "{:.3f}",
    }
    # Fill None → NaN for formatting
    for col in fmt_cols:
        if col in df_t.columns:
            df_t[col] = pd.to_numeric(df_t[col], errors="coerce")

    styled = (
        df_t.style
        .format(fmt_cols, na_rep="—")
        .apply(_colour_pnl_col, subset=["Net PnL"])
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — TIMING
# ═════════════════════════════════════════════════════════════════════════════

def tab_timing(summary: dict):
    col_l, col_r = st.columns(2)

    # Day of week
    with col_l:
        by_day = summary.get("by_day_of_week", {})
        day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday"]
        days   = [d for d in day_order if d in by_day]
        d_vals = [_safe_float((by_day[d] or {}).get("total_net_pnl")) or 0 for d in days]
        st.plotly_chart(chart_bars(days, d_vals, "Net PnL by Day of Week"),
                        use_container_width=True)
        df_day = pd.DataFrame({
            "Day":     days,
            "Net PnL": d_vals,
            "Win Rate": [_pct((by_day[d] or {}).get("win_rate")) for d in days],
            "Trades":  [(by_day[d] or {}).get("n_trades") for d in days],
            "PF":      [_safe_float((by_day[d] or {}).get("profit_factor")) for d in days],
        })
        df_day["Net PnL"] = pd.to_numeric(df_day["Net PnL"], errors="coerce")
        df_day["PF"]      = pd.to_numeric(df_day["PF"],      errors="coerce")
        st.dataframe(
            df_day.style
            .format({"Net PnL": "${:,.2f}", "PF": "{:.3f}"}, na_rep="—")
            .apply(_colour_pnl_col, subset=["Net PnL"]),
            use_container_width=True, hide_index=True,
        )

    # Hour bin
    with col_r:
        by_hr = summary.get("by_hour_bin", {})
        hours  = sorted(by_hr.keys())
        h_vals = [_safe_float((by_hr[h] or {}).get("total_net_pnl")) or 0 for h in hours]
        st.plotly_chart(chart_bars(hours, h_vals, "Net PnL by Hour Bin"),
                        use_container_width=True)
        df_hr = pd.DataFrame({
            "Hour":    hours,
            "Net PnL": h_vals,
            "Win Rate":[_pct((by_hr[h] or {}).get("win_rate")) for h in hours],
            "Trades":  [(by_hr[h] or {}).get("n_trades") for h in hours],
            "PF":      [_safe_float((by_hr[h] or {}).get("profit_factor")) for h in hours],
        })
        df_hr["Net PnL"] = pd.to_numeric(df_hr["Net PnL"], errors="coerce")
        df_hr["PF"]      = pd.to_numeric(df_hr["PF"],      errors="coerce")
        st.dataframe(
            df_hr.style
            .format({"Net PnL": "${:,.2f}", "PF": "{:.3f}"}, na_rep="—")
            .apply(_colour_pnl_col, subset=["Net PnL"]),
            use_container_width=True, hide_index=True,
        )

    # Hold times
    st.markdown("---")
    da = summary.get("duration_analysis", {})
    fig_h = chart_hold_time(da)
    if fig_h:
        col_h, col_s = st.columns([2, 1])
        with col_h:
            st.plotly_chart(fig_h, use_container_width=True)
        with col_s:
            ratio = _safe_float(da.get("loss_to_win_hold_ratio"))
            st.metric("Loss/Win Hold Ratio",
                      f"{ratio:.2f}×" if ratio else "—")
            w = (da.get("wins") or {})
            l = (da.get("losses") or {})
            st.metric("Avg Win Hold",
                      f"{_safe_float(w.get('mean')) or 0:.1f} min")
            st.metric("Avg Loss Hold",
                      f"{_safe_float(l.get('mean')) or 0:.1f} min")
            st.metric("PnL / min",
                      _f(da.get("pnl_per_min_mean"), pre="$", dec=3))


# ═════════════════════════════════════════════════════════════════════════════
# TAB 4 — DIRECTION
# ═════════════════════════════════════════════════════════════════════════════

def tab_direction(summary: dict):
    by_dir = summary.get("by_direction", {})
    if not by_dir:
        st.info("No direction data available.")
        return

    dirs   = list(by_dir.keys())
    totals = [_safe_float((by_dir[d] or {}).get("total_net_pnl")) or 0 for d in dirs]
    st.plotly_chart(chart_bars(dirs, totals, "Net PnL by Direction"),
                    use_container_width=True)

    rows = []
    for d, b in by_dir.items():
        if not b:
            continue
        rows.append({
            "Direction":     d,
            "Trades":        b.get("n_trades"),
            "Net PnL":       _safe_float(b.get("total_net_pnl")),
            "Win Rate":      _pct(b.get("win_rate")),
            "Profit Factor": _safe_float(b.get("profit_factor")),
            "Avg Win ($)":   _safe_float(b.get("avg_win")),
            "Avg Loss ($)":  _safe_float(b.get("avg_loss")),
            "Sharpe":        _safe_float(b.get("sharpe")),
        })
    df_d = pd.DataFrame(rows)
    for col in ["Net PnL","Profit Factor","Avg Win ($)","Avg Loss ($)","Sharpe"]:
        df_d[col] = pd.to_numeric(df_d[col], errors="coerce")
    st.dataframe(
        df_d.style
        .format({
            "Net PnL": "${:,.2f}", "Profit Factor": "{:.3f}",
            "Avg Win ($)": "${:,.2f}", "Avg Loss ($)": "${:,.2f}",
            "Sharpe": "{:.3f}",
        }, na_rep="—")
        .apply(_colour_pnl_col, subset=["Net PnL"]),
        use_container_width=True, hide_index=True,
    )


# ═════════════════════════════════════════════════════════════════════════════
# TAB 5 — INFERENTIAL
# ═════════════════════════════════════════════════════════════════════════════

def tab_inferential(inf: dict):
    if not inf:
        st.info("Enable 'Run inferential tests' in the sidebar.")
        return

    # ── Bootstrap ────────────────────────────────────────────────────────────
    st.markdown("### Bootstrap Confidence Intervals (10 000 resamples)")
    bc  = inf.get("bootstrap", {})
    exp = bc.get("expectancy", {}) or {}
    shr = bc.get("sharpe",     {}) or {}

    c = st.columns(4)
    c[0].metric("Expectancy estimate",  _f(exp.get("estimate")))
    c[1].metric("95% CI lower",         _f(exp.get("ci_lo")))
    c[2].metric("95% CI upper",         _f(exp.get("ci_hi")))
    c[3].metric("CI excludes zero",
                "✅ Yes" if exp.get("significant") else "❌ No")

    c = st.columns(4)
    c[0].metric("Sharpe estimate",  _f(shr.get("estimate"), pre="", dec=3))
    c[1].metric("Sharpe CI lower",  _f(shr.get("ci_lo"),    pre="", dec=3))
    c[2].metric("Sharpe CI upper",  _f(shr.get("ci_hi"),    pre="", dec=3))
    c[3].metric("Sharpe significant",
                "✅ Yes" if shr.get("significant") else "❌ No")

    fig_b = chart_bootstrap_ci(bc.get("by_instrument", {}))
    if fig_b:
        st.plotly_chart(fig_b, use_container_width=True)

    st.markdown("---")

    # ── Results table ─────────────────────────────────────────────────────────
    st.markdown("### All 10 Tests — Summary Table")

    def _p_row(label, p, stat_label="", stat_val=None, note=""):
        p_f    = _safe_float(p)
        sig    = _sig(p_f) if p_f is not None else "—"
        stat_s = f"{stat_label}={stat_val:.4f}" if stat_val is not None else ""
        p_s    = f"{p_f:.4f}" if p_f is not None else "—"
        st.markdown(
            f"**{label}** — p = `{p_s}` {sig}"
            + (f"  ·  {stat_s}" if stat_s else "")
            + (f"  \n*{note}*" if note else "")
        )

    # 1. Normality
    nm = inf.get("normality", {}) or {}
    sw = nm.get("shapiro_wilk", {}) or {}
    dp_nm = nm.get("dagostino_pearson", {}) or {}
    st.markdown("**1. Normality**")
    _p_row("Shapiro-Wilk", sw.get("p_value"),
           "W", _safe_float(sw.get("W_stat")),
           nm.get("interpretation",""))
    _p_row("D'Agostino-Pearson", dp_nm.get("p_value"),
           "K²", _safe_float(dp_nm.get("K2_stat")))

    st.markdown("---")

    # 2. Hold-time MWU
    ht  = inf.get("hold_time", {}) or {}
    mwu_ht = ht.get("mann_whitney", {}) or {}
    st.markdown("**2. Hold-Time Asymmetry (Mann-Whitney U)**")
    if "p_value" in mwu_ht:
        c = st.columns(3)
        p = _safe_float(mwu_ht.get("p_value"))
        c[0].metric("p-value", f"{p:.4f} {_sig(p)}" if p else "—")
        c[1].metric("Effect size r",
                    f"{_safe_float(mwu_ht.get('effect_size_r')) or 0:.3f}"
                    f" ({mwu_ht.get('effect_label','')})")
        c[2].metric("Reject H₀", "Yes" if mwu_ht.get("reject_h0") else "No")
        st.caption(mwu_ht.get("interpretation",""))

    st.markdown("---")

    # 3. MWU per instrument
    mwu_i = inf.get("mwu_by_instrument", {}) or {}
    st.markdown("**3. Mann-Whitney U — Per Instrument vs Rest**")
    rows_mwu = []
    for inst, b in mwu_i.items():
        if not b or "p_value" not in b:
            continue
        p = _safe_float(b.get("p_value"))
        rows_mwu.append({
            "Instrument":   inst,
            "n":            b.get("n"),
            "Median PnL":   _safe_float(b.get("median_pnl")),
            "p-value":      p,
            "Significance": _sig(p),
            "Effect r":     _safe_float(b.get("effect_size_r")),
            "Reject H₀":   "Yes" if b.get("reject_h0") else "No",
        })
    if rows_mwu:
        df_mwu = pd.DataFrame(rows_mwu).sort_values("p-value")
        for col in ["Median PnL","p-value","Effect r"]:
            df_mwu[col] = pd.to_numeric(df_mwu[col], errors="coerce")
        st.dataframe(
            df_mwu.style.format(
                {"Median PnL": "${:,.2f}", "p-value": "{:.4f}", "Effect r": "{:.3f}"},
                na_rep="—",
            ),
            use_container_width=True, hide_index=True,
        )

    st.markdown("---")

    # 4. Kruskal-Wallis
    kw = inf.get("kruskal_wallis", {}) or {}
    st.markdown("**4. Kruskal-Wallis Omnibus Tests**")
    for label, key in [("By Instrument","by_instrument"),
                       ("By Day","by_day"), ("By Hour","by_hour")]:
        r = kw.get(key) or {}
        if not r:
            continue
        p = _safe_float(r.get("p_value"))
        h = _safe_float(r.get("H_stat"))
        st.caption(
            f"{label}: H={h:.2f}, df={r.get('df','?')}, "
            f"p={p:.4f} {_sig(p)} — {r.get('interpretation','')}"
        )

    st.markdown("---")

    # 5. Permutation tests
    pm = inf.get("permutation", {}) or {}
    st.markdown("**5. Permutation Tests**")
    for label, key in [("Long vs Short","long_vs_short"),
                       ("MNQM6 vs ZFM6","mnqm6_vs_zfm6")]:
        r = pm.get(key) or {}
        if not r or "note" in r:
            continue
        p   = _safe_float(r.get("p_value"))
        dif = _safe_float(r.get("observed_diff"))
        st.caption(
            f"{label}: diff=${dif:.2f}, p={p:.4f} {_sig(p)}"
            f" — {r.get('interpretation','')}"
        )

    st.markdown("---")

    # 6. Ljung-Box
    lb = inf.get("ljung_box", {}) or {}
    st.markdown("**6. Ljung-Box Autocorrelation**")
    if "results" in lb:
        for lag_key, lr in (lb.get("results") or {}).items():
            p = _safe_float((lr or {}).get("p_value"))
            lbv = _safe_float((lr or {}).get("LB_stat"))
            st.caption(
                f"Lag {(lr or {}).get('lag','?')}: "
                f"LB={lbv:.2f}, p={p:.4f} {_sig(p)}"
            )
        st.caption(lb.get("interpretation",""))
    elif "note" in lb:
        st.caption(lb["note"])

    st.markdown("---")

    # 7. Durbin-Watson
    dw = inf.get("durbin_watson", {}) or {}
    st.markdown("**7. Durbin-Watson**")
    st.caption(
        f"DW = {_safe_float(dw.get('DW_stat')) or 0:.4f}"
        f" — {dw.get('interpretation','')}"
    )

    st.markdown("---")

    # 8. Runs test
    ru = inf.get("runs_test", {}) or {}
    st.markdown("**8. Wald-Wolfowitz Runs Test**")
    if "p_value" in ru:
        c = st.columns(3)
        p = _safe_float(ru.get("p_value"))
        c[0].metric("Observed runs",  ru.get("observed_runs","—"))
        c[1].metric("Expected runs",
                    f"{_safe_float(ru.get('expected_runs')) or 0:.1f}")
        c[2].metric("p-value",
                    f"{p:.4f} {_sig(p)}" if p is not None else "—")
        st.caption(ru.get("interpretation",""))
    elif "note" in ru:
        st.caption(ru["note"])

    st.markdown("---")

    # 9. CUSUM
    cusum = inf.get("cusum", {}) or {}
    st.markdown("**9. CUSUM Regime Detection**")
    c = st.columns(3)
    c[0].metric("Current regime",
                str(cusum.get("current_regime","—")).upper())
    c[1].metric("First shift at trade",
                cusum.get("first_regime_shift_trade","—"))
    c[2].metric("Downward signals",
                len(cusum.get("change_points_down",[])))
    st.caption(cusum.get("interpretation",""))
    fig_cs = chart_cusum(cusum)
    if fig_cs:
        st.plotly_chart(fig_cs, use_container_width=True)

    st.markdown("---")

    # 10. Bootstrap CI by instrument table
    st.markdown("**10. Bootstrap CI — Per Instrument**")
    ci_rows = []
    for inst, b in (bc.get("by_instrument") or {}).items():
        if not b or b.get("estimate") is None:
            continue
        ci_rows.append({
            "Instrument": inst,
            "Estimate ($)": _safe_float(b.get("estimate")),
            "CI Low ($)":   _safe_float(b.get("ci_lo")),
            "CI High ($)":  _safe_float(b.get("ci_hi")),
            "Excl. Zero":   "✅" if b.get("significant") else "❌",
        })
    if ci_rows:
        df_ci = pd.DataFrame(ci_rows)
        for col in ["Estimate ($)","CI Low ($)","CI High ($)"]:
            df_ci[col] = pd.to_numeric(df_ci[col], errors="coerce")
        st.dataframe(
            df_ci.style.format(
                {"Estimate ($)": "${:,.2f}",
                 "CI Low ($)":   "${:,.2f}",
                 "CI High ($)":  "${:,.2f}"},
                na_rep="—",
            ).apply(_colour_pnl_col, subset=["Estimate ($)"]),
            use_container_width=True, hide_index=True,
        )


# ═════════════════════════════════════════════════════════════════════════════
# TAB 6 — RAW DATA
# ═════════════════════════════════════════════════════════════════════════════

def tab_raw(df: pd.DataFrame):
    st.markdown(f"**{len(df)} trades** · {len(df.columns)} columns")

    c = st.columns(3)
    insts = ["All"] + sorted(df["instrument"].dropna().unique().tolist())
    dirs  = ["All"] + sorted(df["direction"].dropna().unique().tolist())
    outs  = ["All"] + sorted(df["outcome"].dropna().unique().tolist())
    sel_inst = c[0].selectbox("Instrument", insts)
    sel_dir  = c[1].selectbox("Direction",  dirs)
    sel_out  = c[2].selectbox("Outcome",    outs)

    fdf = df.copy()
    if sel_inst != "All": fdf = fdf[fdf["instrument"] == sel_inst]
    if sel_dir  != "All": fdf = fdf[fdf["direction"]  == sel_dir]
    if sel_out  != "All": fdf = fdf[fdf["outcome"]    == sel_out]

    st.caption(f"Showing {len(fdf)} trades")

    display_cols = [
        "trade_index","trade_date","instrument","direction",
        "entry_price","exit_price","net_pnl","outcome",
        "duration_minutes","hour_bin","day_name",
    ]
    show_cols = [c for c in display_cols if c in fdf.columns]
    dsp = fdf[show_cols].copy()

    # Ensure numeric columns are numeric
    for col in ["net_pnl","entry_price","exit_price","duration_minutes"]:
        if col in dsp.columns:
            dsp[col] = pd.to_numeric(dsp[col], errors="coerce")

    fmt_map = {}
    if "net_pnl"           in dsp.columns: fmt_map["net_pnl"]           = "${:,.2f}"
    if "entry_price"       in dsp.columns: fmt_map["entry_price"]       = "{:,.2f}"
    if "exit_price"        in dsp.columns: fmt_map["exit_price"]        = "{:,.2f}"
    if "duration_minutes"  in dsp.columns: fmt_map["duration_minutes"]  = "{:.1f}"

    apply_subsets = {}
    if "net_pnl" in dsp.columns:   apply_subsets["net_pnl"] = _colour_pnl_col
    if "outcome" in dsp.columns:   apply_subsets["outcome"] = _colour_outcome_col

    styled = dsp.style.format(fmt_map, na_rep="—")
    for col, fn in apply_subsets.items():
        styled = styled.apply(fn, subset=[col])

    st.dataframe(styled, use_container_width=True, hide_index=True)

    csv = fdf.to_csv(index=False).encode()
    st.download_button("⬇ Download filtered CSV", csv,
                       "filtered_trades.csv", "text/csv")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 7 — JSON EXPORT
# ═════════════════════════════════════════════════════════════════════════════

def tab_json(summary: dict, inf: dict):
    st.markdown(
        "Paste this payload into Claude for deeper statistical dialogue "
        "or archive it alongside your trade data."
    )
    combined = _json_safe({"descriptive": summary, "inferential": inf})
    js = json.dumps(combined, indent=2, default=str)
    PREVIEW_LIMIT = 8000
    st.code(
        js[:PREVIEW_LIMIT] + ("\n\n# ... truncated — download for full payload"
                              if len(js) > PREVIEW_LIMIT else ""),
        language="json",
    )
    st.download_button(
        "⬇ Download summary.json",
        js.encode(), "summary.json", "application/json",
    )


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    uploaded, run_inf, merge = render_sidebar()

    # ── Landing page ──────────────────────────────────────────────────────────
    if uploaded is None:
        st.markdown("""
# 📊 Trading Analytics Platform

**Upload your broker export in the sidebar to start.**

### Supported brokers (auto-detected)
NinjaTrader / Topstep · Interactive Brokers · Tradovate · TradeStation ·
Tastytrade · MetaTrader 4/5 · Rithmic / Apex · Binance · Kraken · Webull

### What you get
- **Overview** — equity curve, daily PnL, all key metrics
- **Instruments** — per-instrument stats, scatter plot, full table
- **Timing** — by day of week, by hour bin, hold-time asymmetry
- **Direction** — Long vs Short breakdown
- **Inferential** — 10 tests: bootstrap CIs, Mann-Whitney U, Kruskal-Wallis,
  permutation tests, Ljung-Box, Durbin-Watson, runs test, CUSUM
- **Raw Data** — filterable table + CSV download
- **JSON Export** — full analytical payload for Claude
        """)
        return

    # ── Ingest ────────────────────────────────────────────────────────────────
    with st.spinner("Detecting broker and loading file…"):
        try:
            suffix = Path(uploaded.name).suffix or ".csv"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tf:
                tf.write(uploaded.read())
                tmp_path = Path(tf.name)

            result  = ingest(tmp_path)
            df_new  = enrich(result.df)
            tmp_path.unlink(missing_ok=True)

        except Exception as e:
            st.error(f"Could not load file: {e}")
            return

    # ── Merge ─────────────────────────────────────────────────────────────────
    if merge and "prev_df" in st.session_state:
        prev = st.session_state["prev_df"]
        df = (
            pd.concat([prev, df_new], ignore_index=True)
            .sort_values("entered_at_dt")
            .reset_index(drop=True)
        )
        df["trade_index"] = range(1, len(df) + 1)
        st.success(f"Merged {len(prev)} + {len(df_new)} = **{len(df)} trades**")
    else:
        df = df_new

    st.session_state["prev_df"] = df

    # ── Broker banner ─────────────────────────────────────────────────────────
    for w in (result.warnings or []):
        st.warning(w)
    st.caption(
        f"🔍 Broker: **{result.broker}** ({result.confidence:.0%} confidence)  ·  "
        f"**{len(df)} trades** loaded"
    )

    # ── Compute ───────────────────────────────────────────────────────────────
    with st.spinner("Computing statistics…"):
        summary = build_summary(df)

    inf = {}
    if run_inf:
        with st.spinner("Running inferential tests (~5 sec)…"):
            inf = run_all(df)

    # ── Render tabs ───────────────────────────────────────────────────────────
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
    with tabs[4]: tab_inferential(inf)
    with tabs[5]: tab_raw(df)
    with tabs[6]: tab_json(summary, inf)


if __name__ == "__main__":
    main()
