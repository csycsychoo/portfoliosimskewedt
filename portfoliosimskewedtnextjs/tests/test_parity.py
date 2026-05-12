"""
Parity test against the original Streamlit engine.

Verifies that the refactored engine produces byte-identical output to the
original `portfoliosimskewedt.py` when given identical random inputs. This is
the canonical proof of numerical equivalence.

The original file is expected at $ORIGINAL_ENGINE_PATH (default
/home/user/portfoliosimskewedt/portfoliosimskewedt.py); if absent, the test is
skipped so the suite still passes when the original isn't checked out alongside.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from arch.univariate import SkewStudent

ORIG_PATH = Path(os.environ.get(
    "ORIGINAL_ENGINE_PATH",
    "/home/user/portfoliosimskewedt/portfoliosimskewedt.py",
))

if not ORIG_PATH.exists():
    pytest.skip(
        f"Original engine not found at {ORIG_PATH}; "
        "set ORIGINAL_ENGINE_PATH to enable parity tests.",
        allow_module_level=True,
    )

import engine as refactored  # noqa: E402


def _load_original_ns() -> dict:
    src = ORIG_PATH.read_text()
    cutoff = src.index("# --- UI helpers ---")
    engine_src = src[:cutoff]
    engine_src = engine_src.replace("import streamlit as st", "st = None")
    engine_src = engine_src.replace("import plotly.express as px", "px = None")
    engine_src = engine_src.replace("from scipy.stats import norm, skew, kurtosis", "")
    ns: dict = {"np": np, "SkewStudent": SkewStudent, "pd": __import__("pandas")}
    from scipy.stats import norm, skew, kurtosis
    ns.update(norm=norm, skew=skew, kurtosis=kurtosis, px=None)
    exec(engine_src, ns)
    return ns


class _LegacyRNG:
    """Force the refactored engine to draw from np.random.* legacy API."""

    def uniform(self, low=0.0, high=1.0, size=None):
        return np.random.uniform(low, high, size)

    def normal(self, loc=0.0, scale=1.0, size=None):
        return np.random.normal(loc, scale, size)


BASE = dict(
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
    simulation_years=20,
    stock_prop=0.70,
    cash_prop=0.30,
)

SEED = 42
NUM_RUNS = 1000


@pytest.fixture(scope="module")
def orig_ns():
    return _load_original_ns()


@pytest.mark.parametrize(
    "withdrawal_timing,rebalance",
    [
        ("Mid-year", True),
        ("Start of year", True),
        ("Mid-year", False),
        ("Start of year", False),
    ],
)
def test_byte_identical_to_original(orig_ns, withdrawal_timing, rebalance):
    params = dict(BASE, withdrawal_timing=withdrawal_timing, rebalance_each_year=rebalance)

    # Original
    np.random.seed(SEED)
    orig_ns["NUM_RUNS"] = NUM_RUNS
    o_final, o_sr, o_pr, o_pvot = orig_ns["run_monte_carlo_simulation"](**params)

    # Refactored, forced onto legacy RNG so streams match
    np.random.seed(SEED)
    saved = refactored._as_rng
    refactored._as_rng = lambda _s: _LegacyRNG()
    try:
        r = refactored.run_monte_carlo_simulation(seed=SEED, num_runs=NUM_RUNS, **params)
    finally:
        refactored._as_rng = saved

    assert np.array_equal(o_final, r.final_real), "final_real diverged"
    assert np.array_equal(o_sr, r.stock_returns), "stock_returns diverged"
    assert np.array_equal(o_pr, r.portfolio_returns), "portfolio_returns diverged"
    assert np.array_equal(o_pvot, r.portfolio_values_over_time), "portfolio_values_over_time diverged"
