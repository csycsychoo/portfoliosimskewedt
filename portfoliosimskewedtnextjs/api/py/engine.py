"""
Portfolio Monte Carlo engine.

Refactored from the Streamlit prototype with the following changes:
- Streamlit/Plotly dependencies removed; engine is pure compute.
- `seed` parameter added so runs are reproducible (uses np.random.default_rng).
- `num_runs` parameter added (was a module constant).
- B1: `calculate_negative_return_percentages` no longer accepts dead log-mean/sigma args.
- B2: docstring matches the strict-less-than semantics actually implemented.
- B3: non-rebalance + start-of-year branch now records portfolio return after withdrawal
       for consistency with the rebalance branch.
- B4: (preset reset) is a UI concern, handled in the Next.js layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from arch.univariate import SkewStudent
from scipy.stats import norm, skew, kurtosis

DEFAULT_NUM_RUNS = 10_000
DEFAULT_SIMULATION_YEARS = 50

MIN_SIMPLE_RETURN = -0.99
EPS = 1e-12
SQRT_FLOOR = 1e-12
MIN_R_FOR_SQRT = -0.999999
MIN_INFL_FACTOR = 1e-9

SNP7424_GEOM_MEAN = 0.1144
SNP7424_LOG_STD = 0.168824

WithdrawalTiming = Literal["Start of year", "Mid-year"]


def _as_rng(seed: int | None) -> np.random.Generator:
    return np.random.default_rng(seed)


def draw_skewt_log_returns(
    size: tuple[int, ...] | int,
    nu: float,
    lam: float,
    loc: float,
    scale: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Draw Hansen (1994) skewed-Student-t log returns, empirically standardized
    so the sample has mean=loc and std=scale exactly.
    """
    n = int(np.prod(size)) if isinstance(size, tuple) else int(size)
    dist = SkewStudent()

    p = rng.uniform(size=n)
    logr_unscaled = dist.ppf(p, [nu, lam])

    sample_mean = float(np.mean(logr_unscaled))
    sample_std = float(np.std(logr_unscaled, ddof=1))
    if not np.isfinite(sample_std) or sample_std < EPS:
        sample_std = 1.0
    standardized = (logr_unscaled - sample_mean) / sample_std

    logr = loc + scale * standardized
    return logr.reshape(size) if isinstance(size, tuple) else logr


def draw_stock_simple_returns(
    size: tuple[int, ...],
    nu: float,
    lam: float,
    loc: float,
    scale: float,
    rng: np.random.Generator,
) -> np.ndarray:
    logr = draw_skewt_log_returns(size, nu, lam, loc, scale, rng)
    r = np.expm1(logr)
    return np.maximum(r, MIN_SIMPLE_RETURN)


def apply_withdrawal(
    portfolio_values: np.ndarray | None,
    amount: np.ndarray,
    stock_values: np.ndarray | None = None,
    cash_values: np.ndarray | None = None,
):
    if stock_values is None or cash_values is None:
        portfolio_values -= amount
        portfolio_values[portfolio_values < 0] = 0.0
        return portfolio_values

    total = stock_values + cash_values
    w_stock = np.zeros_like(total, dtype=float)
    mask = total > 0
    np.divide(stock_values, total, out=w_stock, where=mask)

    stock_values -= amount * w_stock
    cash_values -= amount * (1.0 - w_stock)

    stock_values[stock_values < 0] = 0.0
    cash_values[cash_values < 0] = 0.0
    return stock_values, cash_values


@dataclass
class SimulationResult:
    final_real: np.ndarray
    stock_returns: np.ndarray
    portfolio_returns: np.ndarray
    portfolio_values_over_time: np.ndarray  # real terms, shape (runs, years+1)


def run_monte_carlo_simulation(
    start_value: float,
    real_spending: float,
    stock_prop: float,
    cash_prop: float,
    cash_return: float,
    cash_vol: float,
    inflation_rate: float,
    inflation_vol: float,
    stock_log_loc: float,
    stock_log_scale: float,
    skewt_nu: float,
    skewt_lambda: float,
    simulation_years: int,
    withdrawal_timing: WithdrawalTiming = "Mid-year",
    rebalance_each_year: bool = True,
    num_runs: int = DEFAULT_NUM_RUNS,
    seed: int | None = None,
) -> SimulationResult:
    rng = _as_rng(seed)

    inflation_matrix = rng.normal(
        loc=inflation_rate, scale=inflation_vol, size=(num_runs, simulation_years)
    )
    cash_returns_matrix = rng.normal(
        loc=cash_return, scale=cash_vol, size=(num_runs, simulation_years)
    )
    stock_returns_matrix = draw_stock_simple_returns(
        (num_runs, simulation_years),
        nu=skewt_nu,
        lam=skewt_lambda,
        loc=stock_log_loc,
        scale=stock_log_scale,
        rng=rng,
    )

    cumulative_inflation = np.ones(num_runs, dtype=np.float64)
    portfolio_returns_matrix = np.zeros((num_runs, simulation_years))
    portfolio_values_over_time = np.zeros((num_runs, simulation_years + 1), dtype=np.float64)
    portfolio_values_over_time[:, 0] = start_value

    if rebalance_each_year:
        portfolio_values = np.full(num_runs, start_value, dtype=np.float64)
        stock_values = cash_values = None
    else:
        portfolio_values = None
        stock_values = np.full(num_runs, start_value * stock_prop, dtype=np.float64)
        cash_values = np.full(num_runs, start_value * cash_prop, dtype=np.float64)

    for year in range(simulation_years):
        infl_factor = np.maximum(1.0 + inflation_matrix[:, year], MIN_INFL_FACTOR)
        spend_soy = real_spending * cumulative_inflation
        spend_mid = spend_soy * np.sqrt(np.maximum(infl_factor, SQRT_FLOOR))

        r_stock = stock_returns_matrix[:, year]
        r_cash = np.maximum(cash_returns_matrix[:, year], MIN_R_FOR_SQRT)

        if rebalance_each_year:
            if withdrawal_timing == "Start of year":
                r_port = stock_prop * r_stock + (1.0 - stock_prop) * r_cash
                portfolio_returns_matrix[:, year] = r_port
                portfolio_values = apply_withdrawal(portfolio_values, spend_soy)
                portfolio_values *= np.maximum(1.0 + r_port, 0.0)
            else:
                temp_stock = portfolio_values * stock_prop
                temp_cash = portfolio_values * (1.0 - stock_prop)

                stock_half = np.sqrt(np.maximum(1.0 + r_stock, SQRT_FLOOR))
                cash_half = np.sqrt(np.maximum(1.0 + r_cash, SQRT_FLOOR))

                temp_stock *= stock_half
                temp_cash *= cash_half
                temp_stock, temp_cash = apply_withdrawal(None, spend_mid, temp_stock, temp_cash)
                temp_stock *= stock_half
                temp_cash *= cash_half

                portfolio_values = temp_stock + temp_cash
                r_port = stock_prop * r_stock + (1.0 - stock_prop) * r_cash
                portfolio_returns_matrix[:, year] = r_port

            portfolio_values[portfolio_values < 0] = 0.0
        else:
            if withdrawal_timing == "Start of year":
                stock_values, cash_values = apply_withdrawal(
                    None, spend_soy, stock_values, cash_values
                )
                stock_values *= np.maximum(1.0 + r_stock, 0.0)
                cash_values *= np.maximum(1.0 + r_cash, 0.0)
                post_total = stock_values + cash_values
                w_stock_post = np.zeros_like(post_total)
                mask = post_total > 0
                np.divide(stock_values, post_total, out=w_stock_post, where=mask)
                r_port = w_stock_post * r_stock + (1.0 - w_stock_post) * r_cash
                portfolio_returns_matrix[:, year] = r_port
            else:
                pre_total = stock_values + cash_values
                w_stock_pre = np.zeros_like(pre_total)
                mask = pre_total > 0
                np.divide(stock_values, pre_total, out=w_stock_pre, where=mask)
                r_port = w_stock_pre * r_stock + (1.0 - w_stock_pre) * r_cash
                portfolio_returns_matrix[:, year] = r_port

                stock_half = np.sqrt(np.maximum(1.0 + r_stock, SQRT_FLOOR))
                cash_half = np.sqrt(np.maximum(1.0 + r_cash, SQRT_FLOOR))

                stock_values *= stock_half
                cash_values *= cash_half
                stock_values, cash_values = apply_withdrawal(
                    None, spend_mid, stock_values, cash_values
                )
                stock_values *= stock_half
                cash_values *= cash_half

            stock_values[stock_values < 0] = 0.0
            cash_values[cash_values < 0] = 0.0

        cumulative_inflation *= infl_factor
        current_nominal = (
            portfolio_values if rebalance_each_year else stock_values + cash_values
        )
        denom = np.maximum(cumulative_inflation, EPS)
        portfolio_values_over_time[:, year + 1] = current_nominal / denom

    final_nominal = portfolio_values if rebalance_each_year else stock_values + cash_values
    denom = np.maximum(cumulative_inflation, EPS)
    final_real = final_nominal / denom

    return SimulationResult(
        final_real=final_real,
        stock_returns=stock_returns_matrix,
        portfolio_returns=portfolio_returns_matrix,
        portfolio_values_over_time=portfolio_values_over_time,
    )


def calculate_negative_return_percentages(
    stock_returns: np.ndarray,
    portfolio_returns: np.ndarray,
) -> list[dict]:
    """
    For each threshold k in {-10%, -15%, ..., -50%}, return three percentages:
    (a) % of yearly stock-return draws strictly less than k,
    (b) % of yearly portfolio-return draws strictly less than k,
    (c) Φ((ln(1+k) - μ)/σ) using S&P 1974-2024 log-mean/log-std (independent of inputs).
    """
    stock_flat = stock_returns.flatten()
    portfolio_flat = portfolio_returns.flatten()
    thresholds = [-(i / 100.0) for i in range(10, 51, 5)]

    normal_mu = float(np.log(1.0 + SNP7424_GEOM_MEAN))
    normal_sigma = float(SNP7424_LOG_STD)

    rows = []
    for k in thresholds:
        stock_pct = float(np.mean(stock_flat < k) * 100)
        portfolio_pct = float(np.mean(portfolio_flat < k) * 100)

        one_plus_k = 1.0 + k
        if normal_sigma > 0.0 and one_plus_k > 0.0:
            z = (np.log(one_plus_k) - normal_mu) / normal_sigma
            normal_pct = float(norm.cdf(z) * 100.0)
        elif one_plus_k <= 0.0:
            normal_pct = 0.0
        else:
            r_det = float(np.exp(normal_mu) - 1.0)
            normal_pct = 100.0 if r_det < k else 0.0

        rows.append({
            "threshold": k,
            "stockPct": stock_pct,
            "portfolioPct": portfolio_pct,
            "normalPct": normal_pct,
        })
    return rows


def summarize(result: SimulationResult, start_value: float, simulation_years: int):
    final = result.final_real
    pct_levels = list(range(10, 100, 10))
    pct_values = np.percentile(final, pct_levels)

    order = np.argsort(final)
    n_runs = final.shape[0]
    pct_traj_levels = [10, 25, 50, 75, 90]
    trajectories: dict[str, list[float]] = {}
    for p in pct_traj_levels:
        rank = int(round((p / 100.0) * (n_runs - 1)))
        run_idx = int(order[rank])
        trajectories[f"p{p}"] = result.portfolio_values_over_time[run_idx, :].tolist()

    stock_flat = result.stock_returns.flatten()
    s_ql, s_qh = np.percentile(stock_flat, [1, 99])
    clipped = stock_flat[(stock_flat >= s_ql) & (stock_flat <= s_qh)]
    bin_edges = np.linspace(s_ql, s_qh, 201)
    counts, _ = np.histogram(clipped, bins=bin_edges)

    return {
        "medianFinal": float(np.median(final)),
        "meanFinal": float(np.mean(final)),
        "stdFinal": float(np.std(final)),
        "successRate": float(np.mean(final > 0) * 100),
        "outperformRate": float(np.mean(final > start_value) * 100),
        "endingPercentiles": {f"p{p}": float(v) for p, v in zip(pct_levels, pct_values)},
        "trajectories": trajectories,
        "stockStats": {
            "median": float(np.median(stock_flat)),
            "mean": float(np.mean(stock_flat)),
            "std": float(np.std(stock_flat)),
            "skew": float(skew(stock_flat, bias=False)),
            "kurtosis": float(kurtosis(stock_flat, fisher=True, bias=False)),
        },
        "stockHistogram": {
            "binEdges": bin_edges.tolist(),
            "counts": counts.tolist(),
        },
        "negativeReturnsTable": calculate_negative_return_percentages(
            result.stock_returns, result.portfolio_returns
        ),
        "years": simulation_years,
    }


def simulate(params: dict) -> dict:
    """Top-level entry: takes percent-typed UI params, returns summary dict."""
    stock_prop = float(params["stockPropPercent"]) / 100.0
    cash_prop = 1.0 - stock_prop

    growth = max(1.0 + float(params["stockGeomMeanPercent"]) / 100.0, MIN_INFL_FACTOR)
    stock_log_loc = float(np.log(growth))
    stock_log_scale = float(params["stockLogVolPercent"]) / 100.0

    sim_years = int(params.get("simulationYears", DEFAULT_SIMULATION_YEARS))
    num_runs = int(params.get("numRuns", DEFAULT_NUM_RUNS))
    seed = params.get("seed", None)
    if seed is not None:
        seed = int(seed)

    result = run_monte_carlo_simulation(
        start_value=float(params["startValue"]),
        real_spending=float(params["realSpending"]),
        stock_prop=stock_prop,
        cash_prop=cash_prop,
        cash_return=float(params["cashReturnPercent"]) / 100.0,
        cash_vol=float(params["cashVolPercent"]) / 100.0,
        inflation_rate=float(params["inflationRatePercent"]) / 100.0,
        inflation_vol=float(params["inflationVolPercent"]) / 100.0,
        stock_log_loc=stock_log_loc,
        stock_log_scale=stock_log_scale,
        skewt_nu=float(params["skewtNu"]),
        skewt_lambda=float(params["skewtLambda"]),
        simulation_years=sim_years,
        withdrawal_timing=params.get("withdrawalTiming", "Mid-year"),
        rebalance_each_year=bool(params.get("rebalanceEachYear", True)),
        num_runs=num_runs,
        seed=seed,
    )
    return summarize(result, float(params["startValue"]), sim_years)
