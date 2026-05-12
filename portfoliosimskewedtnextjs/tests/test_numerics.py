"""
Unit tests for numerical primitives.

These check the building blocks against closed-form / library-cross-check values
so any drift in numpy/scipy/arch versions is caught fast.
"""

import numpy as np
import pytest
from scipy.stats import norm, t as student_t

from engine import (
    SNP7424_GEOM_MEAN,
    SNP7424_LOG_STD,
    apply_withdrawal,
    calculate_negative_return_percentages,
    draw_skewt_log_returns,
    draw_stock_simple_returns,
    run_monte_carlo_simulation,
)


def test_norm_cdf_sanity():
    assert abs(norm.cdf(-1.959963984540054) - 0.025) < 1e-12
    assert abs(norm.cdf(0.0) - 0.5) < 1e-15


def test_skewt_lam_zero_matches_student_t_after_standardization():
    """With lam=0, Hansen skew-t collapses to standardized Student-t. After our
    empirical mean-0/std-1 rescale, the sample should match a Student-t sample
    that has been similarly standardized."""
    rng = np.random.default_rng(123)
    nu = 8.0
    sample = draw_skewt_log_returns((200_000,), nu=nu, lam=0.0, loc=0.0, scale=1.0, rng=rng)
    # Empirical mean ~ 0, std ~ 1 by construction
    assert abs(np.mean(sample)) < 1e-12
    assert abs(np.std(sample, ddof=1) - 1.0) < 1e-12
    # Tail quantiles roughly match a unit-variance Student-t.
    # Standardized t has var nu/(nu-2); divide by sqrt to compare.
    scale_t = np.sqrt(nu / (nu - 2))
    q_sample = np.quantile(sample, [0.01, 0.05, 0.95, 0.99])
    q_theory = student_t.ppf([0.01, 0.05, 0.95, 0.99], df=nu) / scale_t
    assert np.allclose(q_sample, q_theory, atol=0.05)


def test_skewt_negative_lam_produces_left_skew():
    rng = np.random.default_rng(7)
    s = draw_skewt_log_returns((200_000,), nu=5.0, lam=-0.5, loc=0.0, scale=1.0, rng=rng)
    # Empirical third moment should be negative
    m3 = float(np.mean((s - s.mean()) ** 3))
    assert m3 < -0.1


def test_simple_returns_floor():
    rng = np.random.default_rng(0)
    r = draw_stock_simple_returns((10_000,), nu=4.0, lam=-0.9, loc=-1.0, scale=2.0, rng=rng)
    assert r.min() >= -0.99 - 1e-15


def test_apply_withdrawal_proportional():
    stock = np.array([100.0, 50.0, 0.0])
    cash = np.array([100.0, 50.0, 0.0])
    amount = np.array([50.0, 50.0, 50.0])
    s_out, c_out = apply_withdrawal(None, amount, stock, cash)
    # First: 50/50 split -> 75, 75
    assert s_out[0] == pytest.approx(75.0)
    assert c_out[0] == pytest.approx(75.0)
    # Second: 50/50 split -> 25, 25
    assert s_out[1] == pytest.approx(25.0)
    assert c_out[1] == pytest.approx(25.0)
    # Third: total==0, falls to cash leg by convention, clamped to 0
    assert s_out[2] == 0.0
    assert c_out[2] == 0.0


def test_apply_withdrawal_clamps_negative_to_zero():
    stock = np.array([10.0])
    cash = np.array([10.0])
    s_out, c_out = apply_withdrawal(None, np.array([100.0]), stock, cash)
    assert s_out[0] == 0.0
    assert c_out[0] == 0.0


def test_apply_withdrawal_single_pot():
    pv = np.array([100.0, 30.0])
    out = apply_withdrawal(pv, np.array([50.0, 50.0]))
    assert out[0] == pytest.approx(50.0)
    assert out[1] == 0.0


def test_deterministic_one_year_one_run_soy_rebalance():
    """Closed-form check: SoY withdrawal, then full-year growth.
    100 - 10 = 90 ; 90 * (0.7*1.10 + 0.3*1.03) = 90 * 1.079 = 97.11
    Inflation 0 so real == nominal.
    """
    result = run_monte_carlo_simulation(
        start_value=100.0,
        real_spending=10.0,
        stock_prop=0.7,
        cash_prop=0.3,
        cash_return=0.03,
        cash_vol=0.0,
        inflation_rate=0.0,
        inflation_vol=0.0,
        stock_log_loc=np.log(1.10),  # geometric 10%
        stock_log_scale=0.0,  # zero vol => deterministic stock log return = log(1.10)
        skewt_nu=5.0,
        skewt_lambda=0.0,
        simulation_years=1,
        withdrawal_timing="Start of year",
        rebalance_each_year=True,
        num_runs=100,
        seed=1,
    )
    # All runs identical since vol is zero everywhere
    expected = (100.0 - 10.0) * (0.7 * 1.10 + 0.3 * 1.03)
    assert np.allclose(result.final_real, expected, atol=1e-9)


def test_deterministic_one_year_one_run_midyear_rebalance():
    """Mid-year: half growth -> withdraw -> half growth. With zero vol:
    stock half = sqrt(1.10), cash half = sqrt(1.03)
    Pre-withdraw legs: 70*sqrt(1.10), 30*sqrt(1.03)
    Withdraw 10 proportionally, then apply second half growth.
    """
    sh = np.sqrt(1.10)
    ch = np.sqrt(1.03)
    s0, c0 = 70.0 * sh, 30.0 * ch
    total = s0 + c0
    w = s0 / total
    s1 = s0 - 10.0 * w
    c1 = c0 - 10.0 * (1 - w)
    expected = s1 * sh + c1 * ch

    result = run_monte_carlo_simulation(
        start_value=100.0,
        real_spending=10.0,
        stock_prop=0.7,
        cash_prop=0.3,
        cash_return=0.03,
        cash_vol=0.0,
        inflation_rate=0.0,
        inflation_vol=0.0,
        stock_log_loc=np.log(1.10),
        stock_log_scale=0.0,
        skewt_nu=5.0,
        skewt_lambda=0.0,
        simulation_years=1,
        withdrawal_timing="Mid-year",
        rebalance_each_year=True,
        num_runs=50,
        seed=2,
    )
    assert np.allclose(result.final_real, expected, atol=1e-9)


def test_inflation_deflates_final_real():
    """With 5% deterministic inflation, real value at year 1 = nominal/1.05."""
    result = run_monte_carlo_simulation(
        start_value=1_000.0,
        real_spending=0.0,  # no withdrawal
        stock_prop=1.0,
        cash_prop=0.0,
        cash_return=0.0,
        cash_vol=0.0,
        inflation_rate=0.05,
        inflation_vol=0.0,
        stock_log_loc=np.log(1.10),
        stock_log_scale=0.0,
        skewt_nu=5.0,
        skewt_lambda=0.0,
        simulation_years=1,
        withdrawal_timing="Start of year",
        rebalance_each_year=True,
        num_runs=20,
        seed=3,
    )
    expected = 1_000.0 * 1.10 / 1.05
    assert np.allclose(result.final_real, expected, atol=1e-9)


def test_negative_returns_normal_column_uses_snp_constants():
    """Spot-check: for k=-20%, P(log r < log 0.8) = Phi((ln(0.8)-mu)/sigma)."""
    rng = np.random.default_rng(0)
    stock_dummy = rng.uniform(size=(10, 10))
    table = calculate_negative_return_percentages(stock_dummy, stock_dummy)
    mu = float(np.log(1.0 + SNP7424_GEOM_MEAN))
    sigma = SNP7424_LOG_STD
    row = next(r for r in table if abs(r["threshold"] - (-0.20)) < 1e-12)
    z = (np.log(0.80) - mu) / sigma
    expected = float(norm.cdf(z) * 100.0)
    assert abs(row["normalPct"] - expected) < 1e-9


def test_seed_reproducibility():
    """Same seed -> identical results; different seeds -> different results."""
    args = dict(
        start_value=1_000_000.0,
        real_spending=40_000.0,
        stock_prop=0.7,
        cash_prop=0.3,
        cash_return=0.025,
        cash_vol=0.01,
        inflation_rate=0.02,
        inflation_vol=0.015,
        stock_log_loc=float(np.log(1.07)),
        stock_log_scale=0.20,
        skewt_nu=5.0,
        skewt_lambda=-0.3,
        simulation_years=10,
        num_runs=500,
    )
    a = run_monte_carlo_simulation(seed=42, **args).final_real
    b = run_monte_carlo_simulation(seed=42, **args).final_real
    c = run_monte_carlo_simulation(seed=43, **args).final_real
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_no_rebalance_midyear_runs_without_error():
    """Smoke test for the non-rebalance mid-year path."""
    result = run_monte_carlo_simulation(
        start_value=1_000_000.0,
        real_spending=40_000.0,
        stock_prop=0.7,
        cash_prop=0.3,
        cash_return=0.025,
        cash_vol=0.01,
        inflation_rate=0.02,
        inflation_vol=0.015,
        stock_log_loc=float(np.log(1.07)),
        stock_log_scale=0.20,
        skewt_nu=5.0,
        skewt_lambda=-0.3,
        simulation_years=5,
        withdrawal_timing="Mid-year",
        rebalance_each_year=False,
        num_runs=200,
        seed=9,
    )
    assert result.final_real.shape == (200,)
    assert result.portfolio_returns.shape == (200, 5)
