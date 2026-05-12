"""
Parity verification: run the ORIGINAL Streamlit engine and the REFACTORED engine
side-by-side with identical random inputs, and assert byte-equal outputs.

How we achieve byte-equal output despite the RNG API migration:
    Both engines are called with `np.random.seed(SEED)` *before* the run, and we
    monkey-patch the refactored engine's RNG-using methods so it draws from
    `np.random.*` (legacy) instead of `default_rng`. With identical RNG state,
    every downstream draw is identical and the math must be byte-equal.

Run with:
    python scripts/parity_check.py
"""

from __future__ import annotations

import sys
import importlib.util
from pathlib import Path

import numpy as np
from arch.univariate import SkewStudent

REPO_ROOT = Path(__file__).resolve().parent.parent
ORIG_PATH = Path("/home/user/portfoliosimskewedt/portfoliosimskewedt.py")

sys.path.insert(0, str(REPO_ROOT / "api" / "py"))
import engine as refactored  # noqa: E402


# ---------- Load the original engine without importing Streamlit ----------
def load_original_engine_functions():
    """Read the original file and exec just the engine bits in an isolated ns."""
    src = ORIG_PATH.read_text()
    # Strip Streamlit dependency and UI code: keep only the engine functions.
    # Cut at the "# --- UI helpers ---" marker.
    cutoff = src.index("# --- UI helpers ---")
    engine_src = src[:cutoff]
    # Replace the streamlit import (still in the truncated head) with a stub.
    engine_src = engine_src.replace("import streamlit as st", "st = None  # stubbed")
    engine_src = engine_src.replace("from scipy.stats import norm, skew, kurtosis", "")
    ns = {
        "np": np,
        "SkewStudent": SkewStudent,
        "pd": __import__("pandas"),
    }
    # Imports the original expects:
    from scipy.stats import norm, skew, kurtosis
    ns["norm"] = norm
    ns["skew"] = skew
    ns["kurtosis"] = kurtosis
    ns["px"] = None  # plotly not needed for engine
    exec(engine_src, ns)
    return ns


# ---------- Refactored engine forced onto legacy RNG ----------
class LegacyRNGAdapter:
    """Quacks like np.random.Generator but uses the legacy np.random.* API,
    so a single np.random.seed(...) controls both engines identically."""

    def uniform(self, low=0.0, high=1.0, size=None):
        return np.random.uniform(low, high, size)

    def normal(self, loc=0.0, scale=1.0, size=None):
        return np.random.normal(loc, scale, size)


def run_refactored_with_legacy_rng(seed: int, **kwargs):
    """Call the refactored engine but inject a legacy-RNG adapter so its draws
    come from np.random.* (the same source as the original)."""
    # Monkey-patch _as_rng to return our adapter while preserving the seed semantics
    np.random.seed(seed)
    orig_as_rng = refactored._as_rng
    refactored._as_rng = lambda _seed: LegacyRNGAdapter()
    try:
        return refactored.run_monte_carlo_simulation(seed=seed, **kwargs)
    finally:
        refactored._as_rng = orig_as_rng


# ---------- Scenarios ----------
BASE_PARAMS = dict(
    start_value=5_000_000.0,
    real_spending=150_000.0,
    cash_return=0.025,
    cash_vol=0.01,
    inflation_rate=0.017,
    inflation_vol=0.02,
    stock_log_loc=float(np.log(1.07)),
    stock_log_scale=0.20,
    skewt_nu=5.0,
    skewt_lambda=-0.3,
    simulation_years=30,
)

SCENARIOS = [
    ("rebalance + mid-year",       dict(stock_prop=0.70, cash_prop=0.30, withdrawal_timing="Mid-year",     rebalance_each_year=True)),
    ("rebalance + start-of-year",  dict(stock_prop=0.70, cash_prop=0.30, withdrawal_timing="Start of year", rebalance_each_year=True)),
    ("no rebalance + mid-year",    dict(stock_prop=0.70, cash_prop=0.30, withdrawal_timing="Mid-year",     rebalance_each_year=False)),
    ("no rebalance + start-of-yr", dict(stock_prop=0.70, cash_prop=0.30, withdrawal_timing="Start of year", rebalance_each_year=False)),
]

NUM_RUNS = 2000  # Smaller than 10k to keep parity script fast; math is identical.
SEED = 42


def run_original(orig_ns, seed: int, scenario_kwargs: dict):
    """Run the original engine.

    Original uses module-level NUM_RUNS=10_000; we temporarily override it.
    """
    np.random.seed(seed)
    orig_ns["NUM_RUNS"] = NUM_RUNS  # override module constant
    final_real, stock_returns, port_returns, pvot = orig_ns["run_monte_carlo_simulation"](
        **BASE_PARAMS, **scenario_kwargs
    )
    return final_real, stock_returns, port_returns, pvot


def compare(scenario_name: str, orig_ns):
    print(f"\n=== {scenario_name} ===")
    o_final, o_sr, o_pr, o_pvot = run_original(orig_ns, SEED, SCENARIOS_DICT[scenario_name])
    r = run_refactored_with_legacy_rng(
        seed=SEED, num_runs=NUM_RUNS, **BASE_PARAMS, **SCENARIOS_DICT[scenario_name]
    )

    checks = [
        ("final_real",         o_final, r.final_real),
        ("stock_returns",      o_sr,    r.stock_returns),
        ("portfolio_returns",  o_pr,    r.portfolio_returns),
        ("pvot",               o_pvot,  r.portfolio_values_over_time),
    ]
    all_ok = True
    for name, a, b in checks:
        # First check exact equality
        if np.array_equal(a, b):
            print(f"  {name:25s}  EXACT MATCH  shape={a.shape}")
            continue
        # Then numerical near-equality (1e-12 ulp-class tolerance)
        if a.shape != b.shape:
            print(f"  {name:25s}  SHAPE MISMATCH {a.shape} vs {b.shape}")
            all_ok = False
            continue
        max_abs = float(np.max(np.abs(a - b)))
        denom = np.maximum(np.abs(a), 1e-12)
        max_rel = float(np.max(np.abs(a - b) / denom))
        ok = np.allclose(a, b, rtol=1e-12, atol=1e-9)
        print(f"  {name:25s}  {'CLOSE' if ok else 'DIVERGED'}   max|Δ|={max_abs:.3e}  max rel={max_rel:.3e}")
        if not ok:
            # Show first few diverging indices
            diff_idx = np.argwhere(np.abs(a - b) > 1e-9)[:5]
            for idx in diff_idx:
                idx_t = tuple(idx)
                print(f"     idx={idx_t}: orig={a[idx_t]} refactored={b[idx_t]}")
            all_ok = False
    return all_ok


SCENARIOS_DICT = dict(SCENARIOS)


def main():
    print(f"Loading original engine from {ORIG_PATH}")
    orig_ns = load_original_engine_functions()
    print(f"NUM_RUNS={NUM_RUNS}, seed={SEED}")

    results = {name: compare(name, orig_ns) for name, _ in SCENARIOS}

    print("\n========== SUMMARY ==========")
    overall = True
    for name, ok in results.items():
        flag = "PASS" if ok else "FAIL"
        print(f"  [{flag}] {name}")
        overall &= ok
    print("\n" + ("ALL SCENARIOS MATCH ORIGINAL" if overall else "DIVERGENCE DETECTED"))
    sys.exit(0 if overall else 1)


if __name__ == "__main__":
    main()
