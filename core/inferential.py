"""
core/inferential.py
====================
Step 2b — Inferential statistics.

All tests are non-parametric or distribution-free where possible, given
the consistently negative skewness and fat tails in this dataset.

Tests implemented
-----------------
1.  Bootstrap CI          — expectancy and Sharpe (10 000 resamples)
2.  Wilcoxon signed-rank  — hold-time asymmetry: winner vs loser durations
3.  Mann-Whitney U        — per-instrument expectancy vs rest-of-book
4.  Kruskal-Wallis        — omnibus test: are any instruments / days / hours different?
5.  Permutation test      — Long vs Short expectancy
6.  Ljung-Box             — serial autocorrelation in PnL series (tilt / revenge trading)
7.  Runs test             — non-randomness in win/loss sequence
8.  CUSUM                 — regime-change detection on rolling PnL
9.  Kolmogorov-Smirnov    — winner vs loser hold-time distributions
10. Normality tests        — Shapiro-Wilk + D'Agostino-Pearson on net_pnl

Public API
----------
    results = run_all(df)   # → dict  (JSON-serialisable)
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import (
    mannwhitneyu, wilcoxon, kruskal,
    ks_2samp, shapiro, normaltest,
    chi2,
)
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import acorr_ljungbox

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

RNG = np.random.default_rng(42)
N_BOOT = 10_000
ALPHA  = 0.05


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _pval_label(p: float) -> str:
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    if p < 0.10:  return "."
    return "ns"


def _ci_label(lo: float, hi: float, unit: str = "$") -> str:
    return f"[{unit}{lo:.2f}, {unit}{hi:.2f}]"


def _safe(v: Any) -> Any:
    """Convert numpy scalars / nan / inf to JSON-safe types."""
    if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
        return None
    if isinstance(v, (np.integer,)):  return int(v)
    if isinstance(v, (np.floating,)): return float(v)
    if isinstance(v, np.bool_):       return bool(v)
    return v


def _safe_dict(d: dict) -> dict:
    return {k: _safe(v) for k, v in d.items()}


# ─────────────────────────────────────────────────────────────────────────────
# 1. BOOTSTRAP CONFIDENCE INTERVALS
# ─────────────────────────────────────────────────────────────────────────────

def bootstrap_ci(
    pnl: pd.Series,
    stat_fn,
    n_boot: int = N_BOOT,
    ci: float = 0.95,
    label: str = "",
) -> dict:
    """
    Percentile bootstrap CI for any scalar statistic.
    stat_fn must accept a 1-D numpy array and return a scalar.
    """
    arr = pnl.dropna().values
    if len(arr) < 5:
        return {"label": label, "estimate": None, "ci_lo": None, "ci_hi": None,
                "ci_level": ci, "n_boot": n_boot, "note": "insufficient data"}

    obs = stat_fn(arr)
    boots = np.array([
        stat_fn(RNG.choice(arr, size=len(arr), replace=True))
        for _ in range(n_boot)
    ])
    lo = float(np.nanpercentile(boots, (1 - ci) / 2 * 100))
    hi = float(np.nanpercentile(boots, (1 + ci) / 2 * 100))

    return _safe_dict({
        "label":        label,
        "estimate":     float(obs),
        "ci_lo":        lo,
        "ci_hi":        hi,
        "ci_level":     ci,
        "n_boot":       n_boot,
        "significant":  lo > 0 or hi < 0,          # CI excludes zero
        "interpretation": (
            f"95% CI {_ci_label(lo, hi, '' if label=='sharpe' else '$')}"
            f" {'excludes' if (lo > 0 or hi < 0) else 'includes'} zero"
        ),
    })


def _sharpe_fn(arr: np.ndarray) -> float:
    s = arr.std(ddof=1)
    return float(arr.mean() / s * np.sqrt(252)) if s > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 2. WILCOXON SIGNED-RANK: HOLD-TIME ASYMMETRY
# ─────────────────────────────────────────────────────────────────────────────

def wilcoxon_hold_time(df: pd.DataFrame) -> dict:
    """
    H0: winners and losers have the same hold-time distribution.
    Because paired comparison requires equal-length groups, we use
    Mann-Whitney U (unpaired) on the two independent groups.
    """
    wins   = df.loc[df["outcome"] == "win",  "duration_minutes"].dropna().values
    losses = df.loc[df["outcome"] == "loss", "duration_minutes"].dropna().values

    if len(wins) < 5 or len(losses) < 5:
        return {"note": "insufficient data"}

    stat, p = mannwhitneyu(wins, losses, alternative="two-sided")
    # Effect size r = Z / sqrt(N)
    n = len(wins) + len(losses)
    z = stats.norm.ppf(p / 2)
    r = abs(z) / np.sqrt(n)

    return _safe_dict({
        "test":           "Mann-Whitney U (hold time: wins vs losses)",
        "n_wins":         len(wins),
        "n_losses":       len(losses),
        "median_wins_min":   float(np.median(wins)),
        "median_losses_min": float(np.median(losses)),
        "U_stat":         float(stat),
        "p_value":        float(p),
        "significance":   _pval_label(p),
        "effect_size_r":  round(float(r), 4),
        "effect_label":   "large" if r > 0.5 else ("medium" if r > 0.3 else "small"),
        "reject_h0":      bool(p < ALPHA),
        "interpretation": (
            f"Losses are held significantly longer (median {np.median(losses):.0f} min) "
            f"than wins (median {np.median(wins):.0f} min). "
            f"p={p:.4f} {_pval_label(p)}, effect size r={r:.2f} ({('large' if r>0.5 else ('medium' if r>0.3 else 'small'))})."
        ) if p < ALPHA else (
            f"No statistically significant hold-time asymmetry at α={ALPHA}. p={p:.4f}."
        ),
    })


# ─────────────────────────────────────────────────────────────────────────────
# 3. KOLMOGOROV-SMIRNOV: HOLD-TIME DISTRIBUTIONS
# ─────────────────────────────────────────────────────────────────────────────

def ks_hold_time(df: pd.DataFrame) -> dict:
    wins   = df.loc[df["outcome"] == "win",  "duration_minutes"].dropna().values
    losses = df.loc[df["outcome"] == "loss", "duration_minutes"].dropna().values
    if len(wins) < 5 or len(losses) < 5:
        return {"note": "insufficient data"}

    stat, p = ks_2samp(wins, losses)
    return _safe_dict({
        "test":          "Kolmogorov-Smirnov (hold-time distributions)",
        "KS_stat":       float(stat),
        "p_value":       float(p),
        "significance":  _pval_label(p),
        "reject_h0":     bool(p < ALPHA),
        "interpretation": (
            f"The full hold-time distributions of wins and losses are "
            f"{'significantly' if p < ALPHA else 'not significantly'} different "
            f"(KS={stat:.3f}, p={p:.4f})."
        ),
    })


# ─────────────────────────────────────────────────────────────────────────────
# 4. MANN-WHITNEY U: PER-INSTRUMENT VS REST
# ─────────────────────────────────────────────────────────────────────────────

def mwu_by_instrument(df: pd.DataFrame) -> dict:
    """For each instrument, test if its PnL distribution differs from all others."""
    results = {}
    instruments = df["instrument"].dropna().unique()

    for inst in instruments:
        group = df.loc[df["instrument"] == inst,   "net_pnl"].dropna().values
        rest  = df.loc[df["instrument"] != inst,   "net_pnl"].dropna().values
        if len(group) < 5 or len(rest) < 5:
            results[inst] = {"note": "insufficient data"}
            continue

        stat, p = mannwhitneyu(group, rest, alternative="two-sided")
        n = len(group) + len(rest)
        z = stats.norm.ppf(max(p / 2, 1e-10))
        r = abs(z) / np.sqrt(n)

        results[inst] = _safe_dict({
            "n":              len(group),
            "median_pnl":     float(np.median(group)),
            "U_stat":         float(stat),
            "p_value":        float(p),
            "significance":   _pval_label(p),
            "effect_size_r":  round(float(r), 4),
            "reject_h0":      bool(p < ALPHA),
            "interpretation": (
                f"{inst}: PnL distribution is "
                f"{'significantly' if p < ALPHA else 'not significantly'} "
                f"different from all other instruments "
                f"(p={p:.4f} {_pval_label(p)})."
            ),
        })

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 5. KRUSKAL-WALLIS: OMNIBUS ACROSS GROUPS
# ─────────────────────────────────────────────────────────────────────────────

def kruskal_omnibus(df: pd.DataFrame, group_col: str, label: str) -> dict:
    """
    H0: all groups have the same PnL distribution (location equality).
    Non-parametric alternative to one-way ANOVA.
    """
    groups = [
        grp["net_pnl"].dropna().values
        for _, grp in df.groupby(group_col)
        if len(grp) >= 5
    ]
    if len(groups) < 2:
        return {"note": "fewer than 2 groups with n≥5"}

    stat, p = kruskal(*groups)
    return _safe_dict({
        "test":          f"Kruskal-Wallis ({label})",
        "H_stat":        float(stat),
        "df":            len(groups) - 1,
        "p_value":       float(p),
        "significance":  _pval_label(p),
        "n_groups":      len(groups),
        "reject_h0":     bool(p < ALPHA),
        "interpretation": (
            f"{'At least one' if p < ALPHA else 'No'} {label} group has a "
            f"significantly different PnL distribution "
            f"(H={stat:.2f}, df={len(groups)-1}, p={p:.4f} {_pval_label(p)})."
        ),
    })


# ─────────────────────────────────────────────────────────────────────────────
# 6. PERMUTATION TEST: LONG VS SHORT EXPECTANCY
# ─────────────────────────────────────────────────────────────────────────────

def permutation_long_vs_short(df: pd.DataFrame, n_perm: int = N_BOOT) -> dict:
    """
    H0: Long and Short trades are drawn from the same distribution.
    Test statistic: difference in medians (robust to outliers).
    """
    long_  = df.loc[df["direction"] == "Long",  "net_pnl"].dropna().values
    short_ = df.loc[df["direction"] == "Short", "net_pnl"].dropna().values

    if len(long_) < 5 or len(short_) < 5:
        return {"note": "insufficient data"}

    obs_diff = float(np.median(long_) - np.median(short_))
    pooled   = np.concatenate([long_, short_])
    n_long   = len(long_)

    diffs = np.array([
        np.median(RNG.permutation(pooled)[:n_long]) -
        np.median(RNG.permutation(pooled)[n_long:])
        for _ in range(n_perm)
    ])
    p = float(np.mean(np.abs(diffs) >= np.abs(obs_diff)))

    return _safe_dict({
        "test":                "Permutation test (Long vs Short median PnL)",
        "n_long":              int(n_long),
        "n_short":             int(len(short_)),
        "median_long":         float(np.median(long_)),
        "median_short":        float(np.median(short_)),
        "observed_diff":       obs_diff,
        "p_value":             p,
        "significance":        _pval_label(p),
        "n_permutations":      n_perm,
        "reject_h0":           bool(p < ALPHA),
        "interpretation": (
            f"Long trades have a {'significantly' if p < ALPHA else 'not significantly'} "
            f"different median PnL from Short trades "
            f"(diff={obs_diff:.2f}, p={p:.4f} {_pval_label(p)})."
        ),
    })


# ─────────────────────────────────────────────────────────────────────────────
# 7. LJUNG-BOX: SERIAL AUTOCORRELATION IN PNL
# ─────────────────────────────────────────────────────────────────────────────

def ljung_box_pnl(pnl: pd.Series, lags: list[int] = [5, 10, 20]) -> dict:
    """
    H0: PnL series has no serial autocorrelation up to lag k.
    Significant autocorrelation = non-random patterns (streaks, tilt, revenge).
    """
    arr = pnl.dropna().values
    if len(arr) < 30:
        return {"note": "insufficient data (need n≥30)"}

    try:
        lb = acorr_ljungbox(arr, lags=lags, return_df=True)
        results = {}
        for lag in lags:
            row = lb.loc[lag]
            p   = float(row["lb_pvalue"])
            results[f"lag_{lag}"] = _safe_dict({
                "lag":          lag,
                "LB_stat":      float(row["lb_stat"]),
                "p_value":      p,
                "significance": _pval_label(p),
                "reject_h0":    bool(p < ALPHA),
            })

        any_sig = any(v["reject_h0"] for v in results.values())
        return {
            "test":          "Ljung-Box (serial autocorrelation in PnL)",
            "lags_tested":   lags,
            "results":       results,
            "any_significant": any_sig,
            "interpretation": (
                "Serial autocorrelation detected — PnL is NOT random. "
                "Streak/tilt/revenge trading patterns are present."
            ) if any_sig else (
                "No significant serial autocorrelation — PnL sequence is "
                "consistent with random ordering at tested lags."
            ),
        }
    except Exception as e:
        return {"note": f"Ljung-Box failed: {e}"}


# ─────────────────────────────────────────────────────────────────────────────
# 8. RUNS TEST: NON-RANDOMNESS IN WIN/LOSS SEQUENCE
# ─────────────────────────────────────────────────────────────────────────────

def runs_test(pnl: pd.Series) -> dict:
    """
    Wald-Wolfowitz runs test.
    H0: the win/loss sequence is random (no clustering or alternation).
    Too few runs → clustering (streaks).  Too many → alternation.
    """
    arr     = pnl.dropna().values
    binary  = (arr > 0).astype(int)
    n       = len(binary)
    if n < 10:
        return {"note": "insufficient data"}

    n1 = int(binary.sum())
    n0 = n - n1

    # Count runs
    runs = 1 + int(np.sum(binary[1:] != binary[:-1]))

    # Expected runs and variance under H0
    mu_r  = (2 * n0 * n1) / n + 1
    var_r = (2 * n0 * n1 * (2 * n0 * n1 - n)) / (n**2 * (n - 1))
    if var_r <= 0:
        return {"note": "degenerate sequence (all wins or all losses)"}

    z = (runs - mu_r) / np.sqrt(var_r)
    p = float(2 * stats.norm.sf(abs(z)))

    return _safe_dict({
        "test":          "Wald-Wolfowitz runs test",
        "n_wins":        n1,
        "n_losses":      n0,
        "observed_runs": runs,
        "expected_runs": round(float(mu_r), 2),
        "z_stat":        round(float(z), 4),
        "p_value":       float(p),
        "significance":  _pval_label(p),
        "reject_h0":     bool(p < ALPHA),
        "direction":     "clustering (streaks)" if z < 0 else "alternation",
        "interpretation": (
            f"{'Non-random' if p < ALPHA else 'Random'} win/loss sequence "
            f"(runs={runs}, expected={mu_r:.1f}, z={z:.2f}, p={p:.4f}). "
            + (f"Pattern: {'streaks (wins cluster together)' if z < 0 else 'alternation (wins/losses alternate)'}."
               if p < ALPHA else "")
        ),
    })


# ─────────────────────────────────────────────────────────────────────────────
# 9. CUSUM: REGIME CHANGE DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def cusum_regime(pnl: pd.Series, threshold_multiplier: float = 1.0) -> dict:
    """
    CUSUM control chart on net PnL.
    Detects the point at which the process mean shifted.
    threshold = threshold_multiplier × std(pnl)  (conventional: 1×–2×)

    Returns
    -------
    dict with:
        cusum_pos / cusum_neg  — cumulative sum series
        change_points          — list of trade indices where CUSUM exceeded threshold
        regime_labels          — 'normal' | 'degrading' | 'improving'
        current_regime         — regime at end of series
    """
    arr = pnl.dropna().values
    if len(arr) < 20:
        return {"note": "insufficient data for CUSUM (need n≥20)"}

    mu    = float(arr.mean())
    sigma = float(arr.std(ddof=1))
    k     = sigma * 0.5          # reference value (slack)
    h     = sigma * threshold_multiplier * 4  # decision interval

    cusum_pos = np.zeros(len(arr))
    cusum_neg = np.zeros(len(arr))
    change_pts_up   = []
    change_pts_down = []

    for i in range(1, len(arr)):
        cusum_pos[i] = max(0, cusum_pos[i-1] + (arr[i] - mu - k))
        cusum_neg[i] = min(0, cusum_neg[i-1] + (arr[i] - mu + k))
        if cusum_pos[i] >  h: change_pts_up.append(i + 1)
        if cusum_neg[i] < -h: change_pts_down.append(i + 1)

    # Regime at end of series
    final_cusum = float(cusum_pos[-1] + cusum_neg[-1])
    current_regime = (
        "degrading"  if final_cusum < -sigma else
        "improving"  if final_cusum >  sigma else
        "neutral"
    )

    # Identify primary regime shift (first signal from neg CUSUM, more relevant for a losing system)
    first_shift = None
    if change_pts_down:
        first_shift = int(change_pts_down[0])
    elif change_pts_up:
        first_shift = int(change_pts_up[0])

    return {
        "test":              "CUSUM regime-change detection",
        "threshold_used":    round(float(h), 4),
        "process_mean":      round(float(mu), 4),
        "process_std":       round(float(sigma), 4),
        "cusum_pos":         [round(float(v), 4) for v in cusum_pos],
        "cusum_neg":         [round(float(v), 4) for v in cusum_neg],
        "change_points_up":  change_pts_up[:10],    # first 10 only for JSON compactness
        "change_points_down":change_pts_down[:10],
        "first_regime_shift_trade": first_shift,
        "current_regime":    current_regime,
        "interpretation": (
            f"CUSUM signals a {current_regime} regime. "
            + (f"First downward shift detected at trade #{first_shift}. " if first_shift else "")
            + f"{len(change_pts_down)} downward and {len(change_pts_up)} upward signals total."
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 10. NORMALITY TESTS
# ─────────────────────────────────────────────────────────────────────────────

def normality_tests(pnl: pd.Series) -> dict:
    arr = pnl.dropna().values
    if len(arr) < 8:
        return {"note": "insufficient data"}

    results: dict = {}

    # Shapiro-Wilk (accurate for n < 2000)
    if len(arr) <= 2000:
        sw_stat, sw_p = shapiro(arr)
        results["shapiro_wilk"] = _safe_dict({
            "W_stat":       float(sw_stat),
            "p_value":      float(sw_p),
            "significance": _pval_label(sw_p),
            "reject_h0":    bool(sw_p < ALPHA),
        })

    # D'Agostino-Pearson (skewness + kurtosis omnibus)
    k2, dp_p = normaltest(arr)
    results["dagostino_pearson"] = _safe_dict({
        "K2_stat":      float(k2),
        "p_value":      float(dp_p),
        "significance": _pval_label(dp_p),
        "reject_h0":    bool(dp_p < ALPHA),
    })

    non_normal = any(v.get("reject_h0", False) for v in results.values())
    results["interpretation"] = (
        "PnL distribution is significantly non-normal. "
        "Parametric tests (t-test, standard Sharpe CIs) are invalid — "
        "use bootstrap CIs and non-parametric tests throughout."
    ) if non_normal else (
        "PnL distribution is consistent with normality at α=0.05."
    )

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 11. DURBIN-WATSON
# ─────────────────────────────────────────────────────────────────────────────

def durbin_watson_test(pnl: pd.Series) -> dict:
    """
    First-order autocorrelation.  DW ≈ 2 → no autocorrelation.
    DW < 1.5 → positive autocorrelation (streaks).
    DW > 2.5 → negative autocorrelation (alternation).
    """
    arr = pnl.dropna().values
    if len(arr) < 10:
        return {"note": "insufficient data"}

    dw = float(durbin_watson(arr))
    if dw < 1.5:
        interp = f"Positive first-order autocorrelation (DW={dw:.3f} < 1.5) — wins and losses cluster in streaks."
    elif dw > 2.5:
        interp = f"Negative first-order autocorrelation (DW={dw:.3f} > 2.5) — wins and losses tend to alternate."
    else:
        interp = f"No meaningful first-order autocorrelation (DW={dw:.3f} ≈ 2)."

    return {
        "test":          "Durbin-Watson (first-order autocorrelation)",
        "DW_stat":       round(dw, 4),
        "interpretation": interp,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 12. INSTRUMENT-PAIR PERMUTATION: MNQM6 vs ZFM6
# ─────────────────────────────────────────────────────────────────────────────

def permutation_pair(
    df: pd.DataFrame, inst_a: str, inst_b: str, n_perm: int = N_BOOT
) -> dict:
    """
    Permutation test on median PnL difference between two specific instruments.
    """
    a = df.loc[df["instrument"] == inst_a, "net_pnl"].dropna().values
    b = df.loc[df["instrument"] == inst_b, "net_pnl"].dropna().values

    if len(a) < 5 or len(b) < 5:
        return {"note": f"insufficient data for {inst_a} or {inst_b}"}

    obs_diff = float(np.median(a) - np.median(b))
    pooled   = np.concatenate([a, b])
    na       = len(a)

    diffs = np.array([
        np.median(RNG.permutation(pooled)[:na]) -
        np.median(RNG.permutation(pooled)[na:])
        for _ in range(n_perm)
    ])
    p = float(np.mean(np.abs(diffs) >= np.abs(obs_diff)))

    return _safe_dict({
        "test":            f"Permutation test ({inst_a} vs {inst_b} median PnL)",
        "inst_a":          inst_a,
        "inst_b":          inst_b,
        "n_a":             int(na),
        "n_b":             int(len(b)),
        "median_a":        float(np.median(a)),
        "median_b":        float(np.median(b)),
        "observed_diff":   obs_diff,
        "p_value":         p,
        "significance":    _pval_label(p),
        "n_permutations":  n_perm,
        "reject_h0":       bool(p < ALPHA),
        "interpretation": (
            f"{inst_a} and {inst_b} have "
            f"{'significantly' if p < ALPHA else 'not significantly'} "
            f"different median PnL (diff={obs_diff:.2f}, p={p:.4f} {_pval_label(p)})."
        ),
    })


# ─────────────────────────────────────────────────────────────────────────────
# MAIN: RUN ALL TESTS
# ─────────────────────────────────────────────────────────────────────────────

def run_all(df: pd.DataFrame) -> dict:
    """
    Execute all inferential tests and return a structured, JSON-serialisable dict.

    Parameters
    ----------
    df : pd.DataFrame
        Output of core.etl.enrich()

    Returns
    -------
    dict  — keyed by test name, values are result dicts
    """
    pnl = df["net_pnl"]

    print("  [1/10] Bootstrap CIs on expectancy and Sharpe …")
    boot_expectancy = bootstrap_ci(pnl, np.mean,      label="expectancy_usd")
    boot_sharpe     = bootstrap_ci(pnl, _sharpe_fn,   label="sharpe")

    # Per-instrument bootstrap CIs
    boot_by_inst = {}
    for inst, grp in df.groupby("instrument"):
        boot_by_inst[inst] = bootstrap_ci(grp["net_pnl"], np.mean, label=inst)

    print("  [2/10] Normality tests …")
    normality = normality_tests(pnl)

    print("  [3/10] Wilcoxon / MWU hold-time …")
    hold_mwu = wilcoxon_hold_time(df)
    hold_ks  = ks_hold_time(df)

    print("  [4/10] Mann-Whitney U per instrument …")
    mwu_inst = mwu_by_instrument(df)

    print("  [5/10] Kruskal-Wallis omnibus …")
    kw_instrument = kruskal_omnibus(df, "instrument", "instrument")
    kw_dow        = kruskal_omnibus(df, "day_name",   "day of week")
    kw_hour       = kruskal_omnibus(df, "hour_bin",   "hour bin")

    print("  [6/10] Permutation tests …")
    perm_direction = permutation_long_vs_short(df)
    perm_mnq_zf    = permutation_pair(df, "MNQM6", "ZFM6")

    print("  [7/10] Ljung-Box autocorrelation …")
    lb = ljung_box_pnl(pnl)

    print("  [8/10] Durbin-Watson …")
    dw = durbin_watson_test(pnl)

    print("  [9/10] Runs test …")
    runs = runs_test(pnl)

    print("  [10/10] CUSUM regime detection …")
    cusum = cusum_regime(pnl)

    return {
        "bootstrap": {
            "expectancy": boot_expectancy,
            "sharpe":     boot_sharpe,
            "by_instrument": boot_by_inst,
        },
        "normality":         normality,
        "hold_time": {
            "mann_whitney":   hold_mwu,
            "ks_test":        hold_ks,
        },
        "mwu_by_instrument": mwu_inst,
        "kruskal_wallis": {
            "by_instrument": kw_instrument,
            "by_day":        kw_dow,
            "by_hour":       kw_hour,
        },
        "permutation": {
            "long_vs_short":  perm_direction,
            "mnqm6_vs_zfm6":  perm_mnq_zf,
        },
        "ljung_box":         lb,
        "durbin_watson":     dw,
        "runs_test":         runs,
        "cusum":             cusum,
    }
