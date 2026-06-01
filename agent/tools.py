"""
agent/tools.py
==============
Every capability the agent can invoke as a tool call.

Design principles
-----------------
- Each tool is a pure function: (df, params) → structured result dict
- Tools never call the LLM — they are called BY the LLM
- Results are always JSON-serialisable
- Failures return {"error": "..."} not exceptions

Tools available
---------------
1.  get_summary            — full descriptive summary of the dataset
2.  get_inferential        — run all 10 statistical tests
3.  filter_trades          — filter by instrument / direction / date / hour
4.  compare_segments       — compare two filtered slices (e.g. MNQM6 vs ZFM6)
5.  run_bootstrap          — bootstrap CI for any segment
6.  run_permutation_test   — permutation test between two segments
7.  detect_regime          — CUSUM on any PnL slice
8.  worst_trades_analysis  — profile the N worst trades
9.  best_trades_analysis   — profile the N best trades
10. holding_time_profile   — full hold-time distribution stats
11. calendar_analysis      — aggregate PnL by any time dimension
12. consecutive_run_stats  — streak analysis on any slice
13. fee_impact_simulation  — "what if fees were X% lower?"
14. stop_loss_simulation   — "what if I had a hard stop at $Y?"
15. write_finding          — persist a validated finding to the research log
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

# ── re-use existing core modules ─────────────────────────────────────────────
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.descriptive import (
    build_summary, descriptive_block, profit_factor,
    sharpe_ratio, max_drawdown, consecutive_runs, _json_safe,
)
from core.inferential import (
    bootstrap_ci, _sharpe_fn, permutation_pair,
    cusum_regime, runs_test, wilcoxon_hold_time,
)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _safe(v: Any) -> Any:
    if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
        return None
    if isinstance(v, (np.integer,)):  return int(v)
    if isinstance(v, (np.floating,)): return float(v)
    return v

def _safe_dict(d: dict) -> dict:
    return {k: _safe(v) for k, v in d.items()}

def _filter(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Apply filter params to df. All params are optional."""
    fdf = df.copy()
    if params.get("instrument"):
        fdf = fdf[fdf["instrument"] == params["instrument"]]
    if params.get("direction"):
        fdf = fdf[fdf["direction"] == params["direction"]]
    if params.get("outcome"):
        fdf = fdf[fdf["outcome"] == params["outcome"]]
    if params.get("hour_bin"):
        fdf = fdf[fdf["hour_bin"] == params["hour_bin"]]
    if params.get("day_name"):
        fdf = fdf[fdf["day_name"] == params["day_name"]]
    if params.get("date_from"):
        fdf = fdf[fdf["trade_date"].astype(str) >= params["date_from"]]
    if params.get("date_to"):
        fdf = fdf[fdf["trade_date"].astype(str) <= params["date_to"]]
    if params.get("min_duration_min"):
        fdf = fdf[fdf["duration_minutes"] >= float(params["min_duration_min"])]
    if params.get("max_duration_min"):
        fdf = fdf[fdf["duration_minutes"] <= float(params["max_duration_min"])]
    return fdf.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# TOOL IMPLEMENTATIONS
# ─────────────────────────────────────────────────────────────────────────────

def get_summary(df: pd.DataFrame, params: dict) -> dict:
    """Full descriptive summary of the entire dataset (or a filtered slice)."""
    fdf = _filter(df, params)
    if len(fdf) == 0:
        return {"error": "No trades match the filter criteria."}
    return _json_safe(build_summary(fdf))


def get_inferential(df: pd.DataFrame, params: dict) -> dict:
    """Run all 10 inferential tests on the dataset (or a filtered slice)."""
    import warnings
    warnings.filterwarnings("ignore")
    fdf = _filter(df, params)
    if len(fdf) < 10:
        return {"error": f"Too few trades ({len(fdf)}) for inferential tests (need ≥10)."}
    from core.inferential import run_all
    return _json_safe(run_all(fdf))


def filter_trades(df: pd.DataFrame, params: dict) -> dict:
    """
    Return a filtered slice of trades with basic stats.
    Useful for the agent to explore subsets before running heavier tests.
    """
    fdf = _filter(df, params)
    if len(fdf) == 0:
        return {"error": "No trades match the filter criteria."}
    pnl  = fdf["net_pnl"]
    wins = pnl[pnl > 0]; losses = pnl[pnl < 0]
    return _json_safe({
        "n_trades":      len(fdf),
        "filter_applied": params,
        "net_pnl":       float(pnl.sum()),
        "win_rate":      float((pnl > 0).mean()),
        "expectancy":    float(pnl.mean()),
        "profit_factor": profit_factor(pnl),
        "avg_win":       float(wins.mean()) if len(wins) else None,
        "avg_loss":      float(losses.mean()) if len(losses) else None,
        "instruments":   fdf["instrument"].unique().tolist(),
        "date_range":    [str(fdf["trade_date"].min()),
                          str(fdf["trade_date"].max())],
    })


def compare_segments(df: pd.DataFrame, params: dict) -> dict:
    """
    Compare two filtered segments side by side.
    params: {"segment_a": {...filter}, "segment_b": {...filter},
             "label_a": "...", "label_b": "..."}
    """
    a_params = params.get("segment_a", {})
    b_params = params.get("segment_b", {})
    label_a  = params.get("label_a", "Segment A")
    label_b  = params.get("label_b", "Segment B")

    df_a = _filter(df, a_params)
    df_b = _filter(df, b_params)

    if len(df_a) == 0 or len(df_b) == 0:
        return {"error": "One or both segments are empty."}

    block_a = descriptive_block(df_a["net_pnl"], label_a)
    block_b = descriptive_block(df_b["net_pnl"], label_b)

    # Run permutation test if both segments have enough data
    perm = {}
    if len(df_a) >= 5 and len(df_b) >= 5:
        from scipy.stats import mannwhitneyu
        stat, p = mannwhitneyu(
            df_a["net_pnl"].dropna().values,
            df_b["net_pnl"].dropna().values,
            alternative="two-sided",
        )
        perm = {
            "test":      "Mann-Whitney U",
            "U_stat":    float(stat),
            "p_value":   float(p),
            "significant_at_5pct": bool(p < 0.05),
        }

    return _json_safe({
        label_a: block_a,
        label_b: block_b,
        "statistical_comparison": perm,
        "winner": label_a if (block_a.get("total_net_pnl") or 0)
                           > (block_b.get("total_net_pnl") or 0) else label_b,
    })


def run_bootstrap(df: pd.DataFrame, params: dict) -> dict:
    """
    Bootstrap CI for mean or Sharpe on a filtered slice.
    params: {"filter": {...}, "stat": "mean"|"sharpe", "n_boot": 10000}
    """
    fdf  = _filter(df, params.get("filter", {}))
    stat = params.get("stat", "mean")
    n    = int(params.get("n_boot", 10000))

    if len(fdf) < 5:
        return {"error": f"Too few trades ({len(fdf)}) for bootstrap."}

    fn = np.mean if stat == "mean" else _sharpe_fn
    return _json_safe(bootstrap_ci(fdf["net_pnl"], fn,
                                   n_boot=n, label=stat))


def run_permutation_test(df: pd.DataFrame, params: dict) -> dict:
    """
    Permutation test comparing two instruments or two segments.
    params: {"instrument_a": "MNQM6", "instrument_b": "ZFM6"}
         OR params: {"segment_a": {...filter}, "segment_b": {...filter}}
    """
    if params.get("instrument_a") and params.get("instrument_b"):
        return _json_safe(permutation_pair(
            df, params["instrument_a"], params["instrument_b"]
        ))

    df_a = _filter(df, params.get("segment_a", {}))
    df_b = _filter(df, params.get("segment_b", {}))
    if len(df_a) < 5 or len(df_b) < 5:
        return {"error": "Segments too small for permutation test."}

    from core.inferential import RNG, N_BOOT
    obs    = float(np.median(df_a["net_pnl"].values) -
                   np.median(df_b["net_pnl"].values))
    pooled = np.concatenate([df_a["net_pnl"].values,
                              df_b["net_pnl"].values])
    na     = len(df_a)
    diffs  = np.array([
        np.median(RNG.permutation(pooled)[:na]) -
        np.median(RNG.permutation(pooled)[na:])
        for _ in range(N_BOOT)
    ])
    p = float(np.mean(np.abs(diffs) >= np.abs(obs)))
    return _json_safe({
        "test":            "Permutation test (median difference)",
        "observed_diff":   obs,
        "p_value":         p,
        "significant":     bool(p < 0.05),
        "n_permutations":  N_BOOT,
    })


def detect_regime(df: pd.DataFrame, params: dict) -> dict:
    """
    CUSUM regime detection on a filtered slice.
    params: {"filter": {...}, "threshold_multiplier": 1.0}
    """
    fdf = _filter(df, params.get("filter", {}))
    if len(fdf) < 20:
        return {"error": f"Too few trades ({len(fdf)}) for CUSUM (need ≥20)."}
    thr = float(params.get("threshold_multiplier", 1.0))
    result = cusum_regime(fdf["net_pnl"], threshold_multiplier=thr)
    # Strip long series for token efficiency
    result.pop("cusum_pos", None)
    result.pop("cusum_neg", None)
    return _json_safe(result)


def worst_trades_analysis(df: pd.DataFrame, params: dict) -> dict:
    """
    Profile the N worst trades.
    params: {"n": 10, "filter": {...}}
    """
    n   = int(params.get("n", 10))
    fdf = _filter(df, params.get("filter", {}))
    worst = fdf.nsmallest(n, "net_pnl")
    cols  = ["trade_date","instrument","direction","net_pnl",
             "duration_minutes","hour_bin","day_name"]
    cols  = [c for c in cols if c in worst.columns]
    return _json_safe({
        "n_worst":           n,
        "total_damage":      float(worst["net_pnl"].sum()),
        "avg_loss":          float(worst["net_pnl"].mean()),
        "most_common_instrument": worst["instrument"].mode().iloc[0]
                                  if "instrument" in worst.columns else None,
        "most_common_hour":  worst["hour_bin"].mode().iloc[0]
                             if "hour_bin" in worst.columns else None,
        "most_common_day":   worst["day_name"].mode().iloc[0]
                             if "day_name" in worst.columns else None,
        "avg_hold_min":      float(worst["duration_minutes"].mean())
                             if "duration_minutes" in worst.columns else None,
        "trades": worst[cols].to_dict(orient="records"),
    })


def best_trades_analysis(df: pd.DataFrame, params: dict) -> dict:
    """
    Profile the N best trades to understand what winning looks like.
    params: {"n": 10, "filter": {...}}
    """
    n    = int(params.get("n", 10))
    fdf  = _filter(df, params.get("filter", {}))
    best = fdf.nlargest(n, "net_pnl")
    cols = ["trade_date","instrument","direction","net_pnl",
            "duration_minutes","hour_bin","day_name"]
    cols = [c for c in cols if c in best.columns]
    return _json_safe({
        "n_best":            n,
        "total_contribution":float(best["net_pnl"].sum()),
        "avg_win":           float(best["net_pnl"].mean()),
        "most_common_instrument": best["instrument"].mode().iloc[0]
                                  if "instrument" in best.columns else None,
        "most_common_hour":  best["hour_bin"].mode().iloc[0]
                             if "hour_bin" in best.columns else None,
        "most_common_day":   best["day_name"].mode().iloc[0]
                             if "day_name" in best.columns else None,
        "avg_hold_min":      float(best["duration_minutes"].mean())
                             if "duration_minutes" in best.columns else None,
        "trades": best[cols].to_dict(orient="records"),
    })


def holding_time_profile(df: pd.DataFrame, params: dict) -> dict:
    """
    Full hold-time distribution: percentiles, by outcome, by instrument.
    params: {"filter": {...}}
    """
    fdf = _filter(df, params.get("filter", {}))
    if "duration_minutes" not in fdf.columns:
        return {"error": "duration_minutes column not available."}
    dur = fdf["duration_minutes"].dropna()
    if len(dur) == 0:
        return {"error": "No valid duration data in this slice."}

    result: dict = {
        "overall": {
            "mean":   round(float(dur.mean()), 2),
            "median": round(float(dur.median()), 2),
            "std":    round(float(dur.std()), 2),
            "p10":    round(float(np.percentile(dur, 10)), 2),
            "p25":    round(float(np.percentile(dur, 25)), 2),
            "p75":    round(float(np.percentile(dur, 75)), 2),
            "p90":    round(float(np.percentile(dur, 90)), 2),
            "max":    round(float(dur.max()), 2),
        },
    }

    for outcome in ["win", "loss", "breakeven"]:
        sub = fdf.loc[fdf["outcome"] == outcome, "duration_minutes"].dropna()
        if len(sub) > 0:
            result[outcome] = {
                "n":      len(sub),
                "mean":   round(float(sub.mean()), 2),
                "median": round(float(sub.median()), 2),
                "p25":    round(float(np.percentile(sub, 25)), 2),
                "p75":    round(float(np.percentile(sub, 75)), 2),
            }
    return _json_safe(result)


def calendar_analysis(df: pd.DataFrame, params: dict) -> dict:
    """
    Aggregate net_pnl by a time dimension.
    params: {"group_by": "day_name"|"hour_bin"|"trade_date"|"month",
             "filter": {...}}
    """
    fdf      = _filter(df, params.get("filter", {}))
    group_by = params.get("group_by", "day_name")

    if group_by not in fdf.columns:
        return {"error": f"Column '{group_by}' not available."}

    agg = (
        fdf.groupby(group_by)["net_pnl"]
        .agg(["sum","mean","count",
              lambda s: (s > 0).mean()])
        .rename(columns={"sum":"total","mean":"expectancy",
                         "count":"n_trades","<lambda_0>":"win_rate"})
        .round(4)
    )
    return _json_safe({
        "group_by": group_by,
        "data": agg.reset_index().to_dict(orient="records"),
    })


def consecutive_run_stats(df: pd.DataFrame, params: dict) -> dict:
    """
    Streak / consecutive run analysis on a filtered slice.
    params: {"filter": {...}}
    """
    fdf = _filter(df, params.get("filter", {}))
    if len(fdf) < 5:
        return {"error": "Too few trades for streak analysis."}
    return _json_safe(consecutive_runs(fdf["net_pnl"]))


def fee_impact_simulation(df: pd.DataFrame, params: dict) -> dict:
    """
    What if total fees+commissions were reduced by X%?
    params: {"reduction_pct": 50}  → simulates 50% fee reduction
    """
    reduction = float(params.get("reduction_pct", 50)) / 100
    cost_col  = df.get("fees", pd.Series(0, index=df.index)).fillna(0) + \
                df.get("commissions", pd.Series(0, index=df.index)).fillna(0) \
                if "fees" in df.columns else pd.Series(0, index=df.index)

    if "fees" not in df.columns:
        return {"error": "fees column not found in dataset."}

    current_cost = float((df["fees"].fillna(0) +
                          df["commissions"].fillna(0)).sum())
    saving       = current_cost * reduction
    sim_pnl      = df["net_pnl"] + (df["fees"].fillna(0) +
                                    df["commissions"].fillna(0)) * reduction
    return _json_safe({
        "reduction_applied_pct": reduction * 100,
        "current_total_cost":    round(current_cost, 2),
        "simulated_saving":      round(saving, 2),
        "current_net_pnl":       round(float(df["net_pnl"].sum()), 2),
        "simulated_net_pnl":     round(float(sim_pnl.sum()), 2),
        "simulated_win_rate":    round(float((sim_pnl > 0).mean()), 4),
        "simulated_expectancy":  round(float(sim_pnl.mean()), 4),
    })


def stop_loss_simulation(df: pd.DataFrame, params: dict) -> dict:
    """
    What if I had applied a hard stop at $Y per trade?
    params: {"stop_usd": 150, "filter": {...}}
    """
    stop  = float(params.get("stop_usd", 150))
    fdf   = _filter(df, params.get("filter", {}))
    pnl   = fdf["net_pnl"].copy()
    capped = pnl.clip(lower=-abs(stop))

    trades_stopped = int((pnl < -abs(stop)).sum())
    damage_saved   = float(pnl[pnl < -abs(stop)].sum() -
                           capped[pnl < -abs(stop)].sum())

    return _json_safe({
        "stop_applied_usd":     stop,
        "trades_stopped_out":   trades_stopped,
        "original_net_pnl":     round(float(pnl.sum()), 2),
        "simulated_net_pnl":    round(float(capped.sum()), 2),
        "damage_saved":         round(damage_saved, 2),
        "simulated_win_rate":   round(float((capped > 0).mean()), 4),
        "simulated_profit_factor": profit_factor(capped),
        "original_worst_trade": round(float(pnl.min()), 2),
        "simulated_worst_trade":round(float(capped.min()), 2),
    })


def write_finding(df: pd.DataFrame, params: dict) -> dict:
    """
    Persist a validated finding to the research log.
    params: {"title": "...", "finding": "...", "evidence": {...},
             "action": "...", "confidence": "high|medium|low"}
    This is the agent's way of committing a conclusion to memory.
    """
    from agent.memory import append_finding
    finding = {
        "title":      params.get("title", "Untitled finding"),
        "finding":    params.get("finding", ""),
        "evidence":   params.get("evidence", {}),
        "action":     params.get("action", ""),
        "confidence": params.get("confidence", "medium"),
    }
    append_finding(finding)
    return {"status": "saved", "finding": finding}


# ─────────────────────────────────────────────────────────────────────────────
# TOOL REGISTRY
# ─────────────────────────────────────────────────────────────────────────────

TOOLS: dict[str, dict] = {
    "get_summary": {
        "fn":          get_summary,
        "description": "Get full descriptive statistics for the dataset or a filtered slice.",
        "parameters": {
            "type": "object",
            "properties": {
                "instrument":       {"type":"string","description":"Filter to one instrument, e.g. MNQM6"},
                "direction":        {"type":"string","enum":["Long","Short"]},
                "hour_bin":         {"type":"string","description":"e.g. '14-16'"},
                "day_name":         {"type":"string","description":"e.g. 'Monday'"},
                "date_from":        {"type":"string","description":"YYYY-MM-DD"},
                "date_to":          {"type":"string","description":"YYYY-MM-DD"},
            },
        },
    },
    "get_inferential": {
        "fn":          get_inferential,
        "description": "Run all 10 inferential statistical tests.",
        "parameters": {"type":"object","properties":{}},
    },
    "filter_trades": {
        "fn":          filter_trades,
        "description": "Get a quick stats summary of a filtered subset of trades.",
        "parameters": {
            "type": "object",
            "properties": {
                "instrument":       {"type":"string"},
                "direction":        {"type":"string","enum":["Long","Short"]},
                "outcome":          {"type":"string","enum":["win","loss","breakeven"]},
                "hour_bin":         {"type":"string"},
                "day_name":         {"type":"string"},
                "date_from":        {"type":"string"},
                "date_to":          {"type":"string"},
                "min_duration_min": {"type":"number"},
                "max_duration_min": {"type":"number"},
            },
        },
    },
    "compare_segments": {
        "fn":          compare_segments,
        "description": "Compare two trade subsets side by side with a statistical test.",
        "parameters": {
            "type": "object",
            "properties": {
                "segment_a": {"type":"object","description":"Filter params for segment A"},
                "segment_b": {"type":"object","description":"Filter params for segment B"},
                "label_a":   {"type":"string"},
                "label_b":   {"type":"string"},
            },
            "required": ["segment_a","segment_b"],
        },
    },
    "run_bootstrap": {
        "fn":          run_bootstrap,
        "description": "Bootstrap confidence interval for mean PnL or Sharpe ratio.",
        "parameters": {
            "type": "object",
            "properties": {
                "filter": {"type":"object","description":"Filter params"},
                "stat":   {"type":"string","enum":["mean","sharpe"],"default":"mean"},
                "n_boot": {"type":"integer","default":10000},
            },
        },
    },
    "run_permutation_test": {
        "fn":          run_permutation_test,
        "description": "Permutation test: are two segments drawn from the same distribution?",
        "parameters": {
            "type": "object",
            "properties": {
                "instrument_a": {"type":"string"},
                "instrument_b": {"type":"string"},
                "segment_a":    {"type":"object"},
                "segment_b":    {"type":"object"},
            },
        },
    },
    "detect_regime": {
        "fn":          detect_regime,
        "description": "CUSUM regime-change detection on a trade slice.",
        "parameters": {
            "type": "object",
            "properties": {
                "filter":               {"type":"object"},
                "threshold_multiplier": {"type":"number","default":1.0},
            },
        },
    },
    "worst_trades_analysis": {
        "fn":          worst_trades_analysis,
        "description": "Profile the N worst trades to find patterns in losses.",
        "parameters": {
            "type": "object",
            "properties": {
                "n":      {"type":"integer","default":10},
                "filter": {"type":"object"},
            },
        },
    },
    "best_trades_analysis": {
        "fn":          best_trades_analysis,
        "description": "Profile the N best trades to understand what winning looks like.",
        "parameters": {
            "type": "object",
            "properties": {
                "n":      {"type":"integer","default":10},
                "filter": {"type":"object"},
            },
        },
    },
    "holding_time_profile": {
        "fn":          holding_time_profile,
        "description": "Full hold-time distribution broken down by outcome.",
        "parameters": {
            "type": "object",
            "properties": {
                "filter": {"type":"object"},
            },
        },
    },
    "calendar_analysis": {
        "fn":          calendar_analysis,
        "description": "Aggregate PnL by day, hour, date, or any time dimension.",
        "parameters": {
            "type": "object",
            "properties": {
                "group_by": {"type":"string",
                             "enum":["day_name","hour_bin","trade_date"],
                             "default":"day_name"},
                "filter":   {"type":"object"},
            },
        },
    },
    "consecutive_run_stats": {
        "fn":          consecutive_run_stats,
        "description": "Win/loss streak statistics for a trade slice.",
        "parameters": {
            "type": "object",
            "properties": {
                "filter": {"type":"object"},
            },
        },
    },
    "fee_impact_simulation": {
        "fn":          fee_impact_simulation,
        "description": "Simulate the impact of reducing fees by a given percentage.",
        "parameters": {
            "type": "object",
            "properties": {
                "reduction_pct": {"type":"number","description":"0-100","default":50},
            },
        },
    },
    "stop_loss_simulation": {
        "fn":          stop_loss_simulation,
        "description": "Simulate applying a hard dollar stop-loss to every trade.",
        "parameters": {
            "type": "object",
            "properties": {
                "stop_usd": {"type":"number","description":"Hard stop in dollars, e.g. 150"},
                "filter":   {"type":"object"},
            },
            "required": ["stop_usd"],
        },
    },
    "write_finding": {
        "fn":          write_finding,
        "description": "Save a validated finding to the persistent research log.",
        "parameters": {
            "type": "object",
            "properties": {
                "title":      {"type":"string"},
                "finding":    {"type":"string","description":"The statistical finding in plain language"},
                "evidence":   {"type":"object","description":"Key numbers supporting the finding"},
                "action":     {"type":"string","description":"Concrete trading rule implied by this finding"},
                "confidence": {"type":"string","enum":["high","medium","low"]},
            },
            "required": ["title","finding","action"],
        },
    },
}


def get_tool_definitions() -> list[dict]:
    """Return tool definitions in Anthropic API format."""
    return [
        {
            "name":         name,
            "description":  spec["description"],
            "input_schema": spec["parameters"],
        }
        for name, spec in TOOLS.items()
    ]


def execute_tool(name: str, params: dict, df: pd.DataFrame) -> dict:
    """Dispatch a tool call by name."""
    if name not in TOOLS:
        return {"error": f"Unknown tool: {name}"}
    try:
        return TOOLS[name]["fn"](df, params)
    except Exception as e:
        return {"error": str(e)}
