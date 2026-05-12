"""
Golden-master snapshot test.

Locks the engine's output against a JSON snapshot for a frozen parameter set and
seed. Any change in numpy/scipy/arch versions, code, or RNG sequencing that
would silently shift results will fail this test.

To regenerate the snapshot intentionally:
    REGEN_GOLDEN=1 pytest tests/test_golden.py
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest

from engine import run_monte_carlo_simulation, summarize

SNAPSHOT = Path(__file__).parent / "golden" / "snapshot.json"

# Frozen scenario: Singapore preset, 30 years, seed 42, 2000 runs (smaller than
# production NUM_RUNS so the test runs fast; engine math is identical).
PARAMS = dict(
    start_value=5_000_000.0,
    real_spending=150_000.0,
    stock_prop=0.70,
    cash_prop=0.30,
    cash_return=0.025,
    cash_vol=0.01,
    inflation_rate=0.017,
    inflation_vol=0.02,
    stock_log_loc=float(np.log(1.07)),  # 7% geometric
    stock_log_scale=0.20,
    skewt_nu=5.0,
    skewt_lambda=-0.3,
    simulation_years=30,
    withdrawal_timing="Mid-year",
    rebalance_each_year=True,
    num_runs=2_000,
    seed=42,
)


def _compute() -> dict:
    res = run_monte_carlo_simulation(**PARAMS)
    s = summarize(res, start_value=PARAMS["start_value"], simulation_years=PARAMS["simulation_years"])
    # Trim heavy fields not needed for parity (keep edges + a counts checksum)
    return {
        "medianFinal": s["medianFinal"],
        "meanFinal": s["meanFinal"],
        "stdFinal": s["stdFinal"],
        "successRate": s["successRate"],
        "outperformRate": s["outperformRate"],
        "endingPercentiles": s["endingPercentiles"],
        "trajectoriesFinalValues": {k: v[-1] for k, v in s["trajectories"].items()},
        "stockStats": s["stockStats"],
        "stockHistogramCountsSum": int(sum(s["stockHistogram"]["counts"])),
        "stockHistogramEdgesHead": s["stockHistogram"]["binEdges"][:3],
        "stockHistogramEdgesTail": s["stockHistogram"]["binEdges"][-3:],
        "negativeReturnsTable": s["negativeReturnsTable"],
    }


def _close(a, b, rtol=1e-12, atol=1e-9):
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            _close(a[k], b[k], rtol, atol)
    elif isinstance(a, list):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            _close(x, y, rtol, atol)
    elif isinstance(a, float) or isinstance(b, float):
        assert abs(float(a) - float(b)) <= atol + rtol * abs(float(b)), f"{a} vs {b}"
    else:
        assert a == b


def test_golden_snapshot():
    actual = _compute()
    if os.environ.get("REGEN_GOLDEN") == "1" or not SNAPSHOT.exists():
        SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
        SNAPSHOT.write_text(json.dumps(actual, indent=2, sort_keys=True))
        pytest.skip(f"Regenerated snapshot at {SNAPSHOT}")
    expected = json.loads(SNAPSHOT.read_text())
    _close(actual, expected)
