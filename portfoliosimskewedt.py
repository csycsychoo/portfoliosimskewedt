# --- Installation (run this cell first in Colab) ---
# !pip install gradio arch plotly pandas numpy scipy

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from arch.univariate import SkewStudent
from scipy.stats import norm, skew, kurtosis

# --- Configuration ---
SIMULATION_YEARS = 50
NUM_RUNS = 10_000

# App version (update this on each change)
APP_VERSION = "v1.3.14"

# Default Stock model parameters (Hansen skew-t on log-returns)
DEFAULT_SKEWT_NU = 5.0
DEFAULT_SKEWT_LAMBDA = -0.3
DEFAULT_STOCK_LOG_LOC = 0.067659   # 7% geometric => ln(1.07)
DEFAULT_STOCK_LOG_SCALE = 0.185    # 18.5% log scale
MIN_SIMPLE_RETURN = -0.99

# Chart clipping
PLOT_CLIP_LOW = 1
PLOT_CLIP_HIGH = 99

# Numerical safety constants
EPS = 1e-12
SQRT_FLOOR = 1e-12
MIN_R_FOR_SQRT = -0.999999        # ensure 1+r >= tiny positive before sqrt
MIN_INFL_FACTOR = 1e-9            # avoid sqrt(<=0) and division by ~0 later


# S&P 500 1974-2024 parameters for Normal comparison
SNP7424_GEOM_MEAN = 0.1144       # 11.44%
SNP7424_LOG_STD = 0.168824       # 16.8824%

# --- Helpers ---
def draw_skewt_log_returns(size, nu, lam, loc, scale):
    """
    Draws log-returns with a skewed Student-t shape and then standardizes the
    sample to mean 0 / std 1 before applying target loc (mean) and scale (std).
    log-return is same as continuously compounded return for 1 year i.e. r in exp(rt) where t is 1 year
    """
    n = int(np.prod(size))
    dist = SkewStudent()

    # Uniformly draw a set of percentiles 
    p = np.random.uniform(size=n)
    # PPF is percent point function, inverse of CDF. Given percentile, it gives the value of the skewed student t at that percentile with nu skewness and lam tails
    logr_unscaled = dist.ppf(p, [nu, lam])

    # The values generated from ppdf have random mean and variance. First standardize the sample to mean zero and std dev 1
    sample_mean = np.mean(logr_unscaled)
    sample_std = np.std(logr_unscaled, ddof=1)  # Use sample standard deviation
    if not np.isfinite(sample_std) or sample_std < EPS:
        # Extremely unlikely with large n, but guard anyway
        sample_std = 1.0
    standardized_logr = (logr_unscaled - sample_mean) / sample_std

    # Apply the target loc (mean) and scale (std dev)
    logr = loc + scale * standardized_logr
    return logr.reshape(size)


def draw_stock_simple_returns(size, nu, lam, loc, scale):
    logr = draw_skewt_log_returns(size, nu, lam, loc, scale)
    #Above returns the the continuous return "r". S(t) = S(t-1) exp (r). Also called log return because ln (St/St-1) = r
    #Taking exponent exp(r) = S(t)/S(t-1)=ratio of prices between years. Expm1 deducts 1, giving the simple return [S(t)-S(t-1)]/S(t-1)
    r = np.expm1(logr)
    return np.maximum(r, MIN_SIMPLE_RETURN)


def apply_withdrawal(portfolio_values, amount, stock_values=None, cash_values=None):
    """
    Withdraws 'amount' from a portfolio. If stock/cash legs are provided,
    withdraw proportionally to their weights, with safe masking to avoid
    invalid divisions when total == 0.
    """
    if stock_values is None or cash_values is None:
        portfolio_values -= amount
        portfolio_values[portfolio_values < 0] = 0.0
        return portfolio_values
    else:
        total = stock_values + cash_values
        w_stock = np.zeros_like(total, dtype=float)
        mask = total > 0
        # Safe division only where total > 0
        np.divide(stock_values, total, out=w_stock, where=mask)

        stock_values -= amount * w_stock
        cash_values -= amount * (1.0 - w_stock)

        stock_values[stock_values < 0] = 0.0
        cash_values[cash_values < 0] = 0.0
        return stock_values, cash_values

 

# --- Core simulation ---
def run_monte_carlo_simulation(
    start_value,
    real_spending,
    stock_prop,
    cash_prop,
    cash_return,
    cash_vol,
    inflation_rate,
    inflation_vol,
    stock_log_loc,
    stock_log_scale,
    skewt_nu,
    skewt_lambda,
    simulation_years,
    withdrawal_timing="Mid-year",
    rebalance_each_year=True,
):
    # Draw annual shocks
    inflation_matrix = np.random.normal(
        loc=inflation_rate, scale=inflation_vol, size=(NUM_RUNS, simulation_years)
    )
    cash_returns_matrix = np.random.normal(
        loc=cash_return, scale=cash_vol, size=(NUM_RUNS, simulation_years)
    )
    stock_returns_matrix = draw_stock_simple_returns(
        (NUM_RUNS, simulation_years),
        nu=skewt_nu,
        lam=skewt_lambda,
        loc=stock_log_loc,
        scale=stock_log_scale,
    )

    cumulative_inflation = np.ones(NUM_RUNS, dtype=np.float64)
    
    # Track portfolio returns for analysis
    portfolio_returns_matrix = np.zeros((NUM_RUNS, simulation_years))

    if rebalance_each_year:
        portfolio_values = np.full(NUM_RUNS, start_value, dtype=np.float64)
    else:
        stock_values = np.full(NUM_RUNS, start_value * stock_prop, dtype=np.float64)
        cash_values = np.full(NUM_RUNS, start_value * cash_prop, dtype=np.float64)

    for year in range(simulation_years):
        # Inflation factor for the year, clamped to be safely positive
        infl_factor = 1.0 + inflation_matrix[:, year]
        infl_factor = np.maximum(infl_factor, MIN_INFL_FACTOR)

        # Real spending scaled to SoY and mid-year (sqrt for half-year)
        spend_soy = real_spending * cumulative_inflation
        spend_mid = spend_soy * np.sqrt(np.maximum(infl_factor, SQRT_FLOOR))

        # Annual asset returns; clamp cash to avoid 1 + r < 0 in rare tails
        r_stock = stock_returns_matrix[:, year]                 # stock already floored at -0.99
        r_cash = np.maximum(cash_returns_matrix[:, year], MIN_R_FOR_SQRT)

        if rebalance_each_year:
            if withdrawal_timing == "Start of year":
                r_port = stock_prop * r_stock + (1 - stock_prop) * r_cash
                portfolio_returns_matrix[:, year] = r_port
                portfolio_values = apply_withdrawal(portfolio_values, spend_soy)
                portfolio_values *= np.maximum(1.0 + r_port, 0.0)
            else:
                # Mid-year: half growth -> withdraw -> half growth
                temp_stock_values = portfolio_values * stock_prop
                temp_cash_values = portfolio_values * (1 - stock_prop)

                stock_half = np.sqrt(np.maximum(1.0 + r_stock, SQRT_FLOOR))
                cash_half = np.sqrt(np.maximum(1.0 + r_cash, SQRT_FLOOR))

                temp_stock_values *= stock_half
                temp_cash_values *= cash_half

                temp_stock_values, temp_cash_values = apply_withdrawal(
                    None, spend_mid, temp_stock_values, temp_cash_values
                )

                temp_stock_values *= stock_half
                temp_cash_values *= cash_half
                portfolio_values = temp_stock_values + temp_cash_values
                
                # Calculate portfolio return for mid-year timing
                r_port = stock_prop * r_stock + (1 - stock_prop) * r_cash
                portfolio_returns_matrix[:, year] = r_port

            portfolio_values[portfolio_values < 0] = 0.0
        else:
            # Track stock/cash legs without annual rebalance
            # Calculate weighted portfolio return first (before withdrawals affect asset mix)
            current_total = stock_values + cash_values
            w_stock_current = np.zeros_like(current_total, dtype=float)
            mask = current_total > 0
            np.divide(stock_values, current_total, out=w_stock_current, where=mask)
            
            # Portfolio return is weighted average of asset returns
            r_port = w_stock_current * r_stock + (1.0 - w_stock_current) * r_cash
            portfolio_returns_matrix[:, year] = r_port
            
            if withdrawal_timing == "Start of year":
                stock_values, cash_values = apply_withdrawal(
                    None, spend_soy, stock_values, cash_values
                )
                stock_values *= np.maximum(1.0 + r_stock, 0.0)
                cash_values *= np.maximum(1.0 + r_cash, 0.0)
            else:
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

        # Update inflation compounding
        cumulative_inflation *= infl_factor

    final_nominal = portfolio_values if rebalance_each_year else stock_values + cash_values
    denom = np.maximum(cumulative_inflation, EPS)  # guard division
    final_real = final_nominal / denom
    return final_real, stock_returns_matrix, portfolio_returns_matrix


# --- Negative returns analysis ---
def calculate_negative_return_percentages(stock_returns, portfolio_returns, stock_log_mu, stock_log_sigma):
    """Calculate percentage of years at or below thresholds; return 3-column table including Normal model."""
    stock_flat = stock_returns.flatten()
    portfolio_flat = portfolio_returns.flatten()

    # Thresholds every 5% from -10% down to -50%
    thresholds = [-(i / 100.0) for i in range(10, 51, 5)]

    rows = []
    index_labels = []
    # Use S&P 1974-2024 parameters for Normal comparison regardless of inputs
    normal_mu = float(np.log(1.0 + SNP7424_GEOM_MEAN))
    normal_sigma = float(SNP7424_LOG_STD)

    for threshold in thresholds:
        # "Under" semantics: strictly less than threshold
        stock_pct = np.mean(stock_flat < threshold) * 100
        portfolio_pct = np.mean(portfolio_flat < threshold) * 100

        # Normal comparison with S&P '74-'24 geometric mean and log vol
        one_plus_k = 1.0 + threshold
        if normal_sigma > 0.0 and one_plus_k > 0.0:
            z = (np.log(one_plus_k) - normal_mu) / normal_sigma
            norm_pct = float(norm.cdf(z) * 100.0)
        else:
            # Degenerate cases: sigma == 0 => deterministic r = exp(mu)-1
            # If 1+k <= 0 (k <= -100%), probability is 0 since r > -100% always
            if one_plus_k <= 0.0:
                norm_pct = 0.0
            else:
                r_det = np.exp(normal_mu) - 1.0
                norm_pct = 100.0 if r_det < threshold else 0.0

        rows.append([f"{stock_pct:.1f}%", f"{portfolio_pct:.1f}%", f"{norm_pct:.1f}%"])
        index_labels.append(f"Under {int(threshold * 100)}%")

    return pd.DataFrame(
        rows,
        columns=[
            "Stock in your portfolio (%)",
            "Your overall portfolio (%)",
            "Normal distribution with same mean and vol as S&P '74 to '24 (%)",
        ],
        index=index_labels,
    )

# --- Utilities ---
def _normalize_column_name(name: str) -> str:
    """Normalize column names for robust matching (case-insensitive, remove punctuation)."""
    return "".join(ch for ch in str(name).lower() if ch.isalnum() or ch.isspace()).strip()


def extract_percent_list_from_df(df: pd.DataFrame, candidate_names):
    """
    Return a list[float] for the first matching column among candidate_names.
    Accepts either strings like "12.3%" or numeric dtype; strips '%' if present.
    Performs exact match on normalized names, then a loose token-inclusion match.
    """
    normalized_to_original = { _normalize_column_name(col): col for col in df.columns }

    # Exact normalized match
    for cand in candidate_names:
        key = _normalize_column_name(cand)
        if key in normalized_to_original:
            series = df[normalized_to_original[key]]
            if pd.api.types.is_numeric_dtype(series):
                return series.astype(float).tolist()
            return series.astype(str).str.rstrip('%').astype(float).tolist()

    # Loose token match (all tokens contained in the column name)
    for cand in candidate_names:
        tokens = _normalize_column_name(cand).split()
        for norm_col, orig_col in normalized_to_original.items():
            if all(tok in norm_col for tok in tokens if tok):
                series = df[orig_col]
                if pd.api.types.is_numeric_dtype(series):
                    return series.astype(float).tolist()
                return series.astype(str).str.rstrip('%').astype(float).tolist()

    raise KeyError(f"None of the candidate columns found: {candidate_names}")

# --- Results processor ---
def generate_simulation_results(
    start_value, real_spending, stock_prop_percent,
    inflation_rate_percent, inflation_vol_percent,
    cash_return_percent, cash_vol_percent,
    stock_log_loc_percent, stock_log_scale_percent,
    skewt_nu, skewt_lambda,
    simulation_years,
    withdrawal_timing, rebalance_each_year
):
    stock_prop = stock_prop_percent / 100.0
    cash_prop = 1 - stock_prop

    final_values, stock_returns, portfolio_returns = run_monte_carlo_simulation(
        start_value, real_spending, stock_prop, cash_prop,
        cash_return_percent / 100, cash_vol_percent / 100,
        inflation_rate_percent / 100, inflation_vol_percent / 100,
        stock_log_loc_percent / 100, stock_log_scale_percent / 100,
        skewt_nu, skewt_lambda,
        simulation_years,
        withdrawal_timing, rebalance_each_year
    )

    median_val = np.median(final_values)
    mean_val = np.mean(final_values)
    std_val = np.std(final_values)
    success_rate = np.mean(final_values > 0) * 100
    outperform_rate = np.mean(final_values > start_value) * 100

    # Portfolio values chart (ECDF line)
    df = pd.DataFrame(final_values, columns=["Final Real Value"])
    ql, qh = np.percentile(df["Final Real Value"], [PLOT_CLIP_LOW, PLOT_CLIP_HIGH])
    df_plot = df[(df["Final Real Value"] >= ql) & (df["Final Real Value"] <= qh)]
    fig = px.ecdf(df_plot, x="Final Real Value")
    fig.update_layout(yaxis_title="% of sims", title=f"Probability you have this amount or less after {simulation_years}y")
    fig.update_yaxes(tickformat=".0%", rangemode="tozero")
    fig.update_xaxes(rangemode="tozero")
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    fig.add_vline(x=start_value, line_dash="dash", line_color="red")
    fig.add_vline(x=median_val, line_dash="dash", line_color="green")

    # Stock returns chart
    flat = stock_returns.flatten()
    stock_median = np.median(flat)
    stock_mean = np.mean(flat) #this is the simple arithmetic mean, not geometric mean. The stock returns are not in sequence for a run, so geo mean cannot be calculated easily
    stock_std = np.std(flat)
    # Even if the underlying student log return "r" is negatively skewed, the simple return being Pt/Pt-1 = exp(r) will tend to be positive because "r" is exponentiated which makes minimum value 0 and positive "r" magnified
    stock_skew = skew(flat, bias=False)
    stock_kurt = kurtosis(flat, fisher=True, bias=False)
    s_ql, s_qh = np.percentile(flat, [PLOT_CLIP_LOW, PLOT_CLIP_HIGH])
    stock_df = pd.DataFrame(flat[(flat >= s_ql) & (flat <= s_qh)], columns=["Annual Return"])
    stock_fig = px.histogram(stock_df, x="Annual Return", nbins=200, histnorm="percent")
    stock_fig.update_layout(yaxis_title="%", title="Annual Stock Returns")
    stock_fig.update_xaxes(tickformat=".2%")
    stock_fig.update_xaxes(fixedrange=True)
    stock_fig.update_yaxes(fixedrange=True)

    # Negative returns analysis (include Normal comparison with same log-mean/vol)
    negative_returns_table = calculate_negative_return_percentages(
        stock_returns,
        portfolio_returns,
        stock_log_loc_percent / 100.0,
        stock_log_scale_percent / 100.0,
    )

    # Percentile summary: show only 10%..90% percentiles, formatted as $ with thousands separators
    pct_levels = list(range(10, 100, 10))  # 10,20,...,90
    pct_values = np.percentile(df["Final Real Value"].values, pct_levels)
    pct_rows = [f"{p}%" for p in pct_levels]
    pct_formatted = [f"${int(round(v)):,}" for v in pct_values]
    summary_percentiles_df = pd.DataFrame({"Outcome percentiles": pct_formatted}, index=pct_rows)

    # Formatted stat labels for display under histograms
    portfolio_stat_labels = (
        f"Median: ${median_val:,.0f}",
        f"Mean: ${mean_val:,.0f}",
        f"Std Dev: ${std_val:,.0f}"
    )
    stock_stat_labels = (
        f"Median of simple returns: {stock_median:.2%}",
        f"Arithmetic Mean of simple returns: {stock_mean:.2%}",
        f"Std Dev of simple returns: {stock_std:.2%}",
        f"Skew of simple returns: {stock_skew:.2f}",
        f"Kurtosis (fat tails) of simple returns: {stock_kurt:.2f}"
    )

    return f"${median_val:,.0f}", f"{success_rate:.1f}%", f"{outperform_rate:.1f}%", fig, stock_fig, summary_percentiles_df, negative_returns_table, portfolio_stat_labels, stock_stat_labels


# --- Streamlit UI ---
def main():
    st.set_page_config(page_title="Portfolio Monte Carlo Simulator", layout="wide")
    st.title("Retirement Portfolio Simulator with extreme events (fat tails, negative skew)")
    # No version display in sidebar

    if "results" not in st.session_state:
        st.session_state["results"] = None

    left_col, right_col = st.columns([1, 3])

    with left_col:
        # Easy defaults section
        st.subheader("🎯 Easy defaults")
        st.write("Quick preset buttons for common scenarios:")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🇺🇸 USA Today", help="Stock geo return 10.6%, Inflation 2.9%, Cash return 4.8%"):
                st.session_state["stock_geom_mean_percent"] = 10.6
                st.session_state["inflation_rate_percent"] = 2.9
                st.session_state["cash_return_percent"] = 4.8
                st.rerun()
        
        with col2:
            if st.button("📊 Trinity Study", help="Stock geo return 10.6%, Inflation 2.96%, Cash return 5.7%"):
                st.session_state["stock_geom_mean_percent"] = 10.6
                st.session_state["inflation_rate_percent"] = 2.96
                st.session_state["cash_return_percent"] = 5.7
                st.session_state["rebalance_each_year"] = True
                st.rerun()
        
        with col3:
            if st.button("🇸🇬 Singapore", help="Stock geo return 7%, Inflation 1.7%, Cash return 2.5%"):
                st.session_state["stock_geom_mean_percent"] = 7.0
                st.session_state["inflation_rate_percent"] = 1.7
                st.session_state["cash_return_percent"] = 2.5
                st.rerun()
        
        st.divider()
        
        with st.expander("Initial Setup", expanded=True):
            start_value = st.number_input("Starting Portfolio ($)", value=5_000_000, step=50_000, min_value=0)
            real_spending = st.number_input("Annual Withdrawal (grows with inflation) ($)", value=200_000, step=1_000, min_value=0)
            simulation_years = st.number_input("Simulation Years", value=50, step=1, min_value=1)
            stock_prop_percent = st.slider("Stock % in Portfolio", min_value=0, max_value=100, value=70, step=5)
        
        with st.expander("Return and economic assumptions", expanded=True):
            # Default geometric mean implied by default log mean
            # Avg geo mean + 1 = ((Pt)/(P0))^(1/t)=(exp(avg r * t))^ (1/t), where avg r is average log mean/continuously compounded rate
            # Avg geo mean = exp(avg r) -1
            _def_geom_mean_percent = float((np.exp(DEFAULT_STOCK_LOG_LOC) - 1.0) * 100.0)
            
            # Use session state values if set by buttons, otherwise use defaults
            default_stock_geom = st.session_state.get("stock_geom_mean_percent", _def_geom_mean_percent)
            default_cash_return = st.session_state.get("cash_return_percent", 2.5)
            default_inflation = st.session_state.get("inflation_rate_percent", 1.7)
            
            stock_geom_mean_percent = st.slider(
                "Stock Average Return % (Geometric mean)",
                min_value=-20.0, max_value=30.0,
                value=default_stock_geom, step=0.1,
            )
            cash_return_percent = st.slider("Cash/Bond Return (%)", min_value=0.0, max_value=10.0, value=default_cash_return, step=0.1)
            # Display equity risk premium under sliders
            erp_geo_percent = float(stock_geom_mean_percent - cash_return_percent)
            st.write(f"Implied Equity Risk Premium (stock return - cash): {erp_geo_percent:.2f}%")
            inflation_rate_percent = st.slider("Inflation Mean (%)", min_value=0.0, max_value=10.0, value=default_inflation, step=0.1)

        with st.expander("Policy Options", expanded=True):
            withdrawal_timing = st.radio("Withdrawal Timing", options=["Start of year", "Mid-year"], index=1)
            rebalance_each_year = st.checkbox("Rebalance annually", value=True)

        with st.expander("Advanced Options", expanded=False):
            inflation_vol_percent = st.slider("Inflation Vol (%)", min_value=0.0, max_value=5.0, value=2.0, step=0.1)
            cash_vol_percent = st.slider("Cash/Bond Vol (%)", min_value=0.0, max_value=5.0, value=1.0, step=0.1)
            stock_log_vol_percent = st.slider(
                "Stock Log Vol (%)",
                min_value=5.0, max_value=50.0,
                value=float(DEFAULT_STOCK_LOG_SCALE * 100.0), step=0.5,
            )
            skewt_nu = st.slider("Fat Tails (Nu)", min_value=3.0, max_value=20.0, value=float(DEFAULT_SKEWT_NU), step=0.5, help="Lower value = fatter tails")
            skewt_lambda = st.slider("Skewness (Lambda)", min_value=-0.9, max_value=0.9, value=float(DEFAULT_SKEWT_LAMBDA), step=0.05, help="Negative = left skew")

        # Compute implied log-mean (percent) for simulation
        _growth = 1.0 + (stock_geom_mean_percent / 100.0)
        _growth = np.maximum(_growth, MIN_INFL_FACTOR)
        implied_log_mean_percent = float(np.log(_growth) * 100.0)

        run_sim = st.button("🚀 Run Simulation")

    with right_col:
        if run_sim:
            st.session_state["results"] = generate_simulation_results(
                start_value,
                real_spending,
                stock_prop_percent,
                inflation_rate_percent,
                inflation_vol_percent,
                cash_return_percent,
                cash_vol_percent,
                    implied_log_mean_percent,
                    stock_log_vol_percent,
                skewt_nu,
                skewt_lambda,
                int(simulation_years),
                withdrawal_timing,
                rebalance_each_year,
            )

        if st.session_state["results"] is not None:
            (
                median_str,
                success_str,
                outperform_str,
                hist_fig,
                stock_fig,
                summary_df,
                negative_returns_df,
                portfolio_stat_labels,
                stock_stat_labels,
            ) = st.session_state["results"]

            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("Median Value at end\n(today's money)", median_str)
            with m2:
                st.metric("Likelihood do not run out of money\n(% runs end value more than zero)", success_str)
            with m3:
                st.metric("Likelihood end with more money than started (real terms)", outperform_str)

            # Move histograms after the tables
            st.subheader("Ending portfolio values in today's money")
            st.dataframe(summary_df)

            st.subheader("% of years with extreme negative returns compared to S&P historical and normal distribution")
            
            # Build grouped bar data (Stock, Portfolio, Normal, S&P historical) by threshold
            thresholds = [10, 15, 20, 25, 30, 35, 40, 45, 50]
            threshold_labels = [f"Under -{t}%" for t in thresholds]
            # Historical Sp 500 from 1974 to 2024 frequencies (string with % -> float)
            sp_values = [
                "12.5%",
                "10.0%",
                "7.5%",
                "5.0%",
                "2.5%",
                "2.5%",
                "0.0%",
                "0.0%",
                "0.0%",
            ]
            sp_numeric = [float(v.rstrip('%')) for v in sp_values]
            
            # Convert simulation percentages (strings like "12.3%" or numeric) to floats with robust column matching
            try:
                stock_pct = extract_percent_list_from_df(
                    negative_returns_df,
                    [
                        "Stock in your portfolio (%)",
                        "Stock in your portfolio",
                        "Stock",
                    ],
                )
                portfolio_pct = extract_percent_list_from_df(
                    negative_returns_df,
                    [
                        "Your overall portfolio (%)",
                        "Your overall portfolio",
                        "Portfolio",
                    ],
                )
                normal_pct = extract_percent_list_from_df(
                    negative_returns_df,
                    [
                        "Normal distribution with same mean and vol as S&P '74 to '24 (%)",
                        "Normal distribution with same mean and vol as S&P '74 to '24",
                        "Normal distribution",
                        "Normal",
                    ],
                )
            except KeyError as e:
                st.error(
                    "Results table is missing expected columns for the negative-returns chart. "
                    "Please re-run the simulation. If the issue persists, report this message: "
                    f"{e}"
                )
                # Show the table but skip the chart block
                stock_pct = portfolio_pct = normal_pct = None
            
            # Assemble long-form dataframe for grouped bars
            chart_df = pd.DataFrame({
                "Threshold": threshold_labels * 4,
                "Series": (["Stock in your portfolio"] * len(thresholds)
                          + ["Your overall portfolio"] * len(thresholds)
                          + ["Normal distribution with same mean and vol as S&P '74 to '24"] * len(thresholds)
                          + ["Sp 500 from 1974 to 2024"] * len(thresholds)),
                "Percent": stock_pct + portfolio_pct + normal_pct + sp_numeric,
            })
            
            if stock_pct is not None:
                neg_fig = px.line(
                    chart_df,
                    x="Threshold",
                    y="Percent",
                    color="Series",
                    markers=True,
                    labels={"Percent": "% of years"},
                )
                neg_fig.update_layout(
                    margin=dict(l=10, r=10, t=10, b=10),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
                    yaxis_title="% of years",
                    xaxis_title="",
                )
                neg_fig.update_yaxes(ticksuffix="%")
                neg_fig.update_xaxes(categoryorder="array", categoryarray=threshold_labels, tickangle=-30)
                
                # Disable zoom/pan on negative returns chart
                neg_fig.update_xaxes(fixedrange=True)
                neg_fig.update_yaxes(fixedrange=True)
                st.plotly_chart(neg_fig, use_container_width=True, config=dict(
                    scrollZoom=False,
                    displaylogo=False,
                    modeBarButtonsToRemove=[
                        "zoom2d","pan2d","select2d","lasso2d","zoomIn2d","zoomOut2d","autoScale2d","resetScale2d"
                    ],
                ))

            # Histograms at bottom
            # Disable zoom/pan on ECDF chart
            st.plotly_chart(hist_fig, use_container_width=True, config=dict(
                scrollZoom=False,
                displaylogo=False,
                modeBarButtonsToRemove=[
                    "zoom2d","pan2d","select2d","lasso2d","zoomIn2d","zoomOut2d","autoScale2d","resetScale2d"
                ],
            ))
            # Small horizontal labels beneath portfolio values histogram
            p_med, p_mean, p_std = portfolio_stat_labels
            c1, c2, c3 = st.columns(3)
            with c1:
                st.caption(p_med)
            with c2:
                st.caption(p_mean)
            with c3:
                st.caption(p_std)

            # Disable zoom/pan on stock returns histogram
            st.plotly_chart(stock_fig, use_container_width=True, config=dict(
                scrollZoom=False,
                displaylogo=False,
                modeBarButtonsToRemove=[
                    "zoom2d","pan2d","select2d","lasso2d","zoomIn2d","zoomOut2d","autoScale2d","resetScale2d"
                ],
            ))
            # Small horizontal labels beneath stock returns histogram
            # Support 4-5 labels (now includes Skew and Kurtosis)
            labels = list(stock_stat_labels)
            cols = st.columns(len(labels))
            for col, label in zip(cols, labels):
                with col:
                    st.caption(label)
            
        else:
            st.info("Set your assumptions on the left and click 'Run Simulation'.")

        # Version footer
        st.caption(f"App version: {APP_VERSION}")


if __name__ == "__main__":
    main()
