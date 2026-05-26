"""
core/descriptive.py
===================
Step 2a — Descriptive statistics.

All functions operate on a net_pnl Series or an enriched DataFrame.
Nothing here is instrument-specific.

Public API
----------
    summary = build_summary(df)   # → dict  (JSON-serialisable)
    summary = full_report(df)     # alias kept for Streamlit compatibility
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# LOW-LEVEL STATISTICAL HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def profit_factor(pnl: pd.Series) -> float | None:
    gross_profit = float(pnl[pnl > 0].sum())
    gross_loss   = float(abs(pnl[pnl < 0].sum()))
    if gross_loss == 0:
        return None          # undefined (all-wins edge case)
    return round(gross_profit / gross_loss, 4)


def sharpe_ratio(pnl: pd.Series, periods_per_year: int = 252) -> float | None:
    """
    Annualised Sharpe on the raw PnL series (each row = one trade).
    Uses sample std (ddof=1).  Returns None if fewer than 2 observations.
    """
    if len(pnl) < 2 or pnl.std(ddof=1) == 0:
        return None
    return round(
        float(pnl.mean() / pnl.std(ddof=1)) * np.sqrt(periods_per_year), 4
    )


def max_drawdown(pnl: pd.Series) -> dict:
    """
    Compute max drawdown on the cumulative equity curve.

    Returns
    -------
    dict with keys:
        max_dd_usd    : float  (negative number)
        max_dd_pct    : float  (percentage, negative)
        max_dd_trades : int    (number of trades in the worst drawdown period)
    """
    equity    = pnl.cumsum().reset_index(drop=True)
    roll_max  = equity.cummax()
    drawdown  = equity - roll_max

    max_dd_usd = float(drawdown.min())

    # Percentage relative to the running peak (avoid division by zero at t=0)
    pct_dd = drawdown / roll_max.replace(0, np.nan)
    max_dd_pct = float(pct_dd.min() * 100) if not pct_dd.isna().all() else 0.0

    # Duration: longest consecutive streak below the previous peak
    below = (drawdown < 0).astype(int)
    streak_id = (below != below.shift()).cumsum()
    max_dur = int(
        below.groupby(streak_id).sum().max()
    ) if below.any() else 0

    return {
        "max_dd_usd":    round(max_dd_usd, 4),
        "max_dd_pct":    round(max_dd_pct, 4),
        "max_dd_trades": max_dur,
    }


def consecutive_runs(pnl: pd.Series) -> dict:
    """Longest and average winning / losing streaks."""
    win_streaks: list[int]  = []
    loss_streaks: list[int] = []
    current: str | None     = None
    count                   = 0

    for v in pnl:
        label = "win" if v > 1e-9 else "loss"
        if label == current:
            count += 1
        else:
            if current == "win":
                win_streaks.append(count)
            elif current == "loss":
                loss_streaks.append(count)
            current = label
            count   = 1
    # flush last run
    if current == "win":
        win_streaks.append(count)
    elif current == "loss":
        loss_streaks.append(count)

    return {
        "max_win_streak":  max(win_streaks,  default=0),
        "max_loss_streak": max(loss_streaks, default=0),
        "avg_win_streak":  round(float(np.mean(win_streaks)),  2) if win_streaks  else 0.0,
        "avg_loss_streak": round(float(np.mean(loss_streaks)), 2) if loss_streaks else 0.0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CORE DESCRIPTIVE BLOCK
# ─────────────────────────────────────────────────────────────────────────────

def descriptive_block(pnl: pd.Series, label: str = "") -> dict:
    """
    Full set of descriptive statistics for any net_pnl slice.
    Safe on small samples (n < 2) and handles NaN gracefully.
    """
    pnl  = pnl.dropna()
    n    = len(pnl)
    if n == 0:
        return {"label": label, "n_trades": 0}

    wins   = pnl[pnl > 1e-9]
    losses = pnl[pnl < -1e-9]

    win_loss_ratio = (
        round(float(abs(wins.mean() / losses.mean())), 4)
        if len(wins) > 0 and len(losses) > 0
        else None
    )

    return {
        "label":          label,
        "n_trades":       n,
        "win_rate":       round(len(wins) / n, 4),
        "expectancy_usd": round(float(pnl.mean()), 4),
        "total_net_pnl":  round(float(pnl.sum()),  4),
        "gross_profit":   round(float(wins.sum()),  4),
        "gross_loss":     round(float(losses.sum()),4),
        "profit_factor":  profit_factor(pnl),
        "avg_win":        round(float(wins.mean()),   4) if len(wins)   else 0.0,
        "avg_loss":       round(float(losses.mean()), 4) if len(losses) else 0.0,
        "win_loss_ratio": win_loss_ratio,
        "std_pnl":        round(float(pnl.std(ddof=1)), 4) if n > 1 else 0.0,
        "skewness":       round(float(pnl.skew()), 4)      if n > 2 else None,
        "kurtosis":       round(float(pnl.kurt()), 4)      if n > 3 else None,
        "best_trade":     round(float(pnl.max()), 4),
        "worst_trade":    round(float(pnl.min()), 4),
        "sharpe":         sharpe_ratio(pnl),
        **max_drawdown(pnl),
    }


# ─────────────────────────────────────────────────────────────────────────────
# DURATION ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def duration_analysis(df: pd.DataFrame) -> dict:
    """
    Compare hold times for winning vs losing trades.
    The win/loss hold-time asymmetry is one of the most actionable signals.
    """
    if "duration_minutes" not in df.columns:
        return {}

    wins   = df.loc[df["outcome"] == "win",  "duration_minutes"].dropna()
    losses = df.loc[df["outcome"] == "loss", "duration_minutes"].dropna()
    all_   = df["duration_minutes"].dropna()

    def _stats(s: pd.Series) -> dict:
        if len(s) == 0:
            return {"mean": None, "median": None, "std": None}
        return {
            "mean":   round(float(s.mean()),   2),
            "median": round(float(s.median()), 2),
            "std":    round(float(s.std()),    2) if len(s) > 1 else 0.0,
        }

    hold_ratio = (
        round(float(losses.mean() / wins.mean()), 4)
        if len(wins) > 0 and len(losses) > 0 and wins.mean() != 0
        else None
    )

    return {
        "all":        _stats(all_),
        "wins":       _stats(wins),
        "losses":     _stats(losses),
        "loss_to_win_hold_ratio": hold_ratio,   # > 1 → losses held longer
        "pnl_per_min_mean": round(float(df["pnl_per_minute"].mean()), 4)
            if "pnl_per_minute" in df.columns else None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# FEE ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def fee_analysis(df: pd.DataFrame) -> dict:
    total_fees   = float(df["fees"].fillna(0).sum())
    total_comm   = float(df["commissions"].fillna(0).sum())
    total_cost   = total_fees + total_comm
    gross_abs    = float(df["pnl_gross"].abs().sum())
    cost_pct     = round(total_cost / gross_abs * 100, 2) if gross_abs > 0 else None

    return {
        "total_fees":        round(total_fees, 4),
        "total_commissions": round(total_comm, 4),
        "total_cost":        round(total_cost, 4),
        "cost_pct_of_gross": cost_pct,
    }


# ─────────────────────────────────────────────────────────────────────────────
# AGGREGATION HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _slice_by(df: pd.DataFrame, col: str) -> dict:
    """Return a descriptive_block dict keyed by each unique value in col."""
    if col not in df.columns:
        return {}
    return {
        str(val): descriptive_block(grp["net_pnl"], str(val))
        for val, grp in df.groupby(col, sort=True)
        if len(grp) > 0
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN SUMMARY BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_summary(df: pd.DataFrame) -> dict:
    """
    Build the full analytical payload from an enriched DataFrame.

    The returned dict is:
    - JSON-serialisable (no numpy types, no NaN — replaced with None)
    - Structured for direct use in Streamlit widgets AND as a Claude prompt payload

    Parameters
    ----------
    df : pd.DataFrame
        Output of core.etl.enrich()

    Returns
    -------
    dict
    """
    pnl = df["net_pnl"]

    summary: dict[str, Any] = {

        "meta": {
            "data_from":      str(df["trade_date"].min()) if "trade_date" in df.columns else None,
            "data_to":        str(df["trade_date"].max()) if "trade_date" in df.columns else None,
            "trading_days":   int(df["trade_date"].nunique()) if "trade_date" in df.columns else None,
            "instruments":    sorted(df["instrument"].dropna().unique().tolist()),
            "n_instruments":  int(df["instrument"].nunique()),
            "directions":     sorted(df["direction"].dropna().unique().tolist()),
        },

        # ── Overall ────────────────────────────────────────────────────────
        "overall": descriptive_block(pnl, "all_trades"),

        # ── By instrument ──────────────────────────────────────────────────
        "by_instrument": _slice_by(df, "instrument"),

        # ── By direction (Long / Short) ────────────────────────────────────
        "by_direction": _slice_by(df, "direction"),

        # ── By day of week ─────────────────────────────────────────────────
        "by_day_of_week": _slice_by(df, "day_name"),

        # ── By hour bin (universal session proxy) ──────────────────────────
        "by_hour_bin": _slice_by(df, "hour_bin"),

        # ── Duration analysis ──────────────────────────────────────────────
        "duration_analysis": duration_analysis(df),

        # ── Consecutive runs ───────────────────────────────────────────────
        "run_analysis": consecutive_runs(pnl),

        # ── Fee drag ───────────────────────────────────────────────────────
        "fee_analysis": fee_analysis(df),

        # ── Daily PnL (for calendar heatmap) ──────────────────────────────
        "daily_pnl": (
            {
                str(k): round(float(v), 4)
                for k, v in df.groupby(df["trade_date"].astype(str))["net_pnl"].sum().items()
            }
            if "trade_date" in df.columns else {}
        ),

        # ── Equity curve (trade-level, for chart) ─────────────────────────
        "equity_curve": pnl.cumsum().round(4).tolist(),
    }

    return _json_safe(summary)


# Alias for Streamlit compatibility
full_report = build_summary


# ─────────────────────────────────────────────────────────────────────────────
# JSON SERIALISATION HELPER
# ─────────────────────────────────────────────────────────────────────────────

def _json_safe(obj: Any) -> Any:
    """Recursively replace numpy scalars, NaN, and inf with JSON-safe types."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if (np.isnan(v) or np.isinf(v)) else v
    if isinstance(obj, np.ndarray):
        return _json_safe(obj.tolist())
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    return obj


def to_json(summary: dict, indent: int = 2) -> str:
    """Serialise the summary dict to a JSON string."""
    return json.dumps(summary, indent=indent, default=str)
