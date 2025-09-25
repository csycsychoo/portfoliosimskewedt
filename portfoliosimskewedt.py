# --- Installation (run this cell first in Colab) ---
# !pip install gradio arch plotly pandas numpy scipy

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from arch.univariate import SkewStudent
from scipy.stats import norm, skew, kurtosis

# --- Configuration ---
SIMULATION_YEARS = 40
NUM_RUNS = 10_000

# App version (update this on each change)
APP_VERSION = "v1.3.6"

# Default Stock model parameters (Hansen skew-t on log-returns)
DEFAULT_SKEWT_NU = 5.0
DEFAULT_SKEWT_LAMBDA = -0.3
DEFAULT_STOCK_LOG_LOC = 0.067657   # ~7% geometric => ln(1.07)
DEFAULT_STOCK_LOG_SCALE = 0.17     # 17% log scale
MIN_SIMPLE_RETURN = -0.99

# Chart clipping
PLOT_CLIP_LOW = 1
PLOT_CLIP_HIGH = 99

# Numerical safety constants
EPS = 1e-12
SQRT_FLOOR = 1e-12
MIN_R_FOR_SQRT = -0.999999        # ensure 1+r >= tiny positive before sqrt
MIN_INFL_FACTOR = 1e-9            # avoid sqrt(<=0) and division by ~0 later


# --- Helpers ---
def draw_skewt_log_returns(size, nu, lam, loc, scale):
    """
    Draws log-returns with a skewed Student-t shape and then standardizes the
    sample to mean 0 / std 1 before applying target loc (mean) and scale (std).
    """
    n = int(np.prod(size))
    dist = SkewStudent()

    # Draw raw samples to get desired distribution shape
    p = np.random.uniform(size=n)
    logr_unscaled = dist.ppf(p, [nu, lam])

    # Standardize the sample before applying loc and scale
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


# Conversion helpers between arithmetic simple-return stats and log-return stats
def arithmetic_to_log_params(mean_simple, vol_simple):
    """
    Convert arithmetic mean/vol of simple returns to implied log-mean/log-vol.

    Uses lognormal moment relationships as an approximation:
      sigma^2 = ln(1 + (vol^2) / (1 + mean)^2)
      mu      = ln(1 + mean) - 0.5 * sigma^2

    All inputs are in decimal (e.g., 0.08 for 8%).
    """
    mean_simple = np.clip(mean_simple, MIN_SIMPLE_RETURN + 1e-6, 1e6)
    vol_simple = max(0.0, float(vol_simple))

    growth = 1.0 + mean_simple
    growth = np.maximum(growth, MIN_INFL_FACTOR)
    denom = np.maximum(growth * growth, EPS)
    sigma2 = np.log(1.0 + (vol_simple * vol_simple) / denom)
    sigma2 = np.maximum(sigma2, 0.0)
    sigma = np.sqrt(sigma2)
    mu = np.log(growth) - 0.5 * sigma2
    return float(mu), float(sigma)


def log_to_arithmetic_params(mu, sigma):
    """
    Convert log-mean/log-vol to arithmetic mean/vol of simple returns, using
    lognormal moment relationships as an approximation:
      E[1+r] = exp(mu + 0.5*sigma^2)
      Var(1+r) = (exp(sigma^2) - 1) * exp(2*mu + sigma^2)
    """
    mu = float(mu)
    sigma = max(0.0, float(sigma))
    mean_simple = np.exp(mu + 0.5 * sigma * sigma) - 1.0
    var_simple = (np.exp(sigma * sigma) - 1.0) * np.exp(2.0 * mu + sigma * sigma)
    vol_simple = np.sqrt(np.maximum(var_simple, 0.0))
    return float(mean_simple), float(vol_simple)


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
    withdrawal_timing="Mid-year",
    rebalance_each_year=True,
):
    # Draw annual shocks
    inflation_matrix = np.random.normal(
        loc=inflation_rate, scale=inflation_vol, size=(NUM_RUNS, SIMULATION_YEARS)
    )
    cash_returns_matrix = np.random.normal(
        loc=cash_return, scale=cash_vol, size=(NUM_RUNS, SIMULATION_YEARS)
    )
    stock_returns_matrix = draw_stock_simple_returns(
        (NUM_RUNS, SIMULATION_YEARS),
        nu=skewt_nu,
        lam=skewt_lambda,
        loc=stock_log_loc,
        scale=stock_log_scale,
    )

    cumulative_inflation = np.ones(NUM_RUNS, dtype=np.float64)
    
    # Track portfolio returns for analysis
    portfolio_returns_matrix = np.zeros((NUM_RUNS, SIMULATION_YEARS))

    if rebalance_each_year:
        portfolio_values = np.full(NUM_RUNS, start_value, dtype=np.float64)
    else:
        stock_values = np.full(NUM_RUNS, start_value * stock_prop, dtype=np.float64)
        cash_values = np.full(NUM_RUNS, start_value * cash_prop, dtype=np.float64)

    for year in range(SIMULATION_YEARS):
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
    for threshold in thresholds:
        # "Under" semantics: strictly less than threshold
        stock_pct = np.mean(stock_flat < threshold) * 100
        portfolio_pct = np.mean(portfolio_flat < threshold) * 100

        # Normal comparison with same geometric mean (mu = ln(1+g)) and log vol (sigma)
        one_plus_k = 1.0 + threshold
        if stock_log_sigma > 0.0 and one_plus_k > 0.0:
            z = (np.log(one_plus_k) - float(stock_log_mu)) / float(stock_log_sigma)
            norm_pct = float(norm.cdf(z) * 100.0)
        else:
            # Degenerate cases: sigma == 0 => deterministic r = exp(mu)-1
            # If 1+k <= 0 (k <= -100%), probability is 0 since r > -100% always
            if one_plus_k <= 0.0:
                norm_pct = 0.0
            else:
                r_det = np.exp(float(stock_log_mu)) - 1.0
                norm_pct = 100.0 if r_det < threshold else 0.0

        rows.append([f"{stock_pct:.1f}%", f"{portfolio_pct:.1f}%", f"{norm_pct:.1f}%"])
        index_labels.append(f"Under {int(threshold * 100)}%")

    return pd.DataFrame(
        rows,
        columns=[
            "Stock Only (%)",
            "Overall Portfolio (%)",
            "Normal (same g, log vol) (%)",
        ],
        index=index_labels,
    )

# --- Results processor ---
def generate_simulation_results(
    start_value, real_spending, stock_prop_percent,
    inflation_rate_percent, inflation_vol_percent,
    cash_return_percent, cash_vol_percent,
    stock_log_loc_percent, stock_log_scale_percent,
    skewt_nu, skewt_lambda,
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
        withdrawal_timing, rebalance_each_year
    )

    median_val = np.median(final_values)
    mean_val = np.mean(final_values)
    std_val = np.std(final_values)
    success_rate = np.mean(final_values > 0) * 100
    outperform_rate = np.mean(final_values > start_value) * 100

    # Portfolio values chart
    df = pd.DataFrame(final_values, columns=["Final Real Value"])
    ql, qh = np.percentile(df["Final Real Value"], [PLOT_CLIP_LOW, PLOT_CLIP_HIGH])
    df_plot = df[(df["Final Real Value"] >= ql) & (df["Final Real Value"] <= qh)]
    fig = px.histogram(df_plot, x="Final Real Value", nbins=200, histnorm="percent")
    fig.update_layout(yaxis_title="%", title=f"{NUM_RUNS:,} Sims After {SIMULATION_YEARS}y (clipped)")
    fig.add_vline(x=start_value, line_dash="dash", line_color="red")
    fig.add_vline(x=median_val, line_dash="dash", line_color="green")

    # Stock returns chart
    flat = stock_returns.flatten()
    stock_median = np.median(flat)
    stock_mean = np.mean(flat)
    stock_std = np.std(flat)
    stock_skew = skew(flat, bias=False)
    stock_kurt = kurtosis(flat, fisher=True, bias=False)
    s_ql, s_qh = np.percentile(flat, [PLOT_CLIP_LOW, PLOT_CLIP_HIGH])
    stock_df = pd.DataFrame(flat[(flat >= s_ql) & (flat <= s_qh)], columns=["Annual Return"])
    stock_fig = px.histogram(stock_df, x="Annual Return", nbins=200, histnorm="percent")
    stock_fig.update_layout(yaxis_title="%", title="Annual Stock Returns (clipped)")
    stock_fig.update_xaxes(tickformat=".2%")

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
		f"Median: {stock_median:.2%}",
		f"Mean: {stock_mean:.2%}",
		f"Std Dev: {stock_std:.2%}",
		f"Skew: {stock_skew:.2f}",
		f"Kurtosis: {stock_kurt:.2f}"
	)

    return f"${median_val:,.0f}", f"{success_rate:.1f}%", f"{outperform_rate:.1f}%", fig, stock_fig, summary_percentiles_df, negative_returns_table, portfolio_stat_labels, stock_stat_labels


# --- Streamlit UI ---
def main():
    st.set_page_config(page_title="Portfolio Monte Carlo Simulator", layout="wide")
    st.title("💰 Portfolio Longevity Simulator")
    # No version display in sidebar

    if "results" not in st.session_state:
        st.session_state["results"] = None

    left_col, right_col = st.columns([1, 3])

    with left_col:
        with st.expander("Initial Setup", expanded=True):
            start_value = st.number_input("Starting Value ($)", value=7_000_000, step=50_000, min_value=0)
            real_spending = st.number_input("Annual Real Spending ($)", value=150_000, step=1_000, min_value=0)
        with st.expander("Asset Allocation", expanded=True):
            stock_prop_percent = st.slider("Stock %", min_value=0, max_value=100, value=70, step=5)
        with st.expander("Economic Assumptions", expanded=True):
            inflation_rate_percent = st.slider("Inflation Mean (%)", min_value=0.0, max_value=10.0, value=2.5, step=0.1)
            inflation_vol_percent = st.slider("Inflation Vol (%)", min_value=0.0, max_value=5.0, value=1.5, step=0.1)
        with st.expander("Cash Assumptions", expanded=True):
            cash_return_percent = st.slider("Cash Return (%)", min_value=0.0, max_value=10.0, value=2.0, step=0.1)
            cash_vol_percent = st.slider("Cash Vol (%)", min_value=0.0, max_value=5.0, value=1.5, step=0.1)

        with st.expander("Stock Model Assumptions", expanded=True):
            # Default geometric mean implied by default log mean
            _def_geom_mean_percent = float((np.exp(DEFAULT_STOCK_LOG_LOC) - 1.0) * 100.0)
            stock_geom_mean_percent = st.slider(
                "Stock Return Mean (geometric %)",
                min_value=-20.0, max_value=30.0,
                value=_def_geom_mean_percent, step=0.1,
            )
            stock_log_vol_percent = st.slider(
                "Stock Log Vol (%)",
                min_value=5.0, max_value=50.0,
                value=float(DEFAULT_STOCK_LOG_SCALE * 100.0), step=0.5,
            )

            # Compute implied log-mean (percent) for display and simulation
            _growth = 1.0 + (stock_geom_mean_percent / 100.0)
            _growth = np.maximum(_growth, MIN_INFL_FACTOR)
            implied_log_mean_percent = float(np.log(_growth) * 100.0)

            # Display removed per request; still computed for simulation

            skewt_nu = st.slider("Fat Tails (Nu)", min_value=3.0, max_value=20.0, value=float(DEFAULT_SKEWT_NU), step=0.5, help="Lower value = fatter tails")
            skewt_lambda = st.slider("Skewness (Lambda)", min_value=-0.9, max_value=0.9, value=float(DEFAULT_SKEWT_LAMBDA), step=0.05, help="Negative = left skew")

        with st.expander("Policy Options", expanded=True):
            withdrawal_timing = st.radio("Withdrawal Timing", options=["Start of year", "Mid-year"], index=1)
            rebalance_each_year = st.checkbox("Rebalance annually", value=True)

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
                st.metric("Success Rate\n(% runs with positive end value)", success_str)
            with m3:
                st.metric("Outperformance Rate\n(end value > starting value)", outperform_str)

            # Move histograms after the tables
            st.subheader("Ending portfolio values in today's money")
            st.dataframe(summary_df)

            st.subheader("% of years with extreme negative returns")
            
            # Create a combined comparison table
            sp_index = [f"Under -{p}%" for p in [10, 15, 20, 25, 30, 35, 40, 45, 50]]
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
            
            # Create comparison table with all three models
            comparison_data = []
            for i, threshold in enumerate([10, 15, 20, 25, 30, 35, 40, 45, 50]):
                row = [
                    f"Under -{threshold}%",
                    negative_returns_df.iloc[i]["Stock Only (%)"],
                    negative_returns_df.iloc[i]["Overall Portfolio (%)"],
                    negative_returns_df.iloc[i]["Normal (same g, log vol) (%)"],
                    sp_values[i]
                ]
                comparison_data.append(row)
            
            comparison_df = pd.DataFrame(
                comparison_data,
                columns=[
                    "Threshold",
                    "Stock Only",
                    "Portfolio",
                    "Normal Model",
                    "S&P 1974-2024"
                ]
            )
            
            # Add mobile-friendly CSS styling
            st.markdown("""
            <style>
            .comparison-table {
                font-size: 14px;
                overflow-x: auto;
            }
            @media (max-width: 768px) {
                .comparison-table {
                    font-size: 12px;
                }
            }
            </style>
            """, unsafe_allow_html=True)
            
            # Display the comparison table with mobile-friendly styling
            st.markdown('<div class="comparison-table">', unsafe_allow_html=True)
            st.dataframe(
                comparison_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Threshold": st.column_config.TextColumn("Threshold", width="small"),
                    "Stock Only": st.column_config.TextColumn("Stock Only", width="small"),
                    "Portfolio": st.column_config.TextColumn("Portfolio", width="small"),
                    "Normal Model": st.column_config.TextColumn("Normal Model", width="small"),
                    "S&P 1974-2024": st.column_config.TextColumn("S&P 1974-2024", width="small")
                }
            )
            st.markdown('</div>', unsafe_allow_html=True)

            # Histograms at bottom
            st.plotly_chart(hist_fig, use_container_width=True)
            # Small horizontal labels beneath portfolio values histogram
            p_med, p_mean, p_std = portfolio_stat_labels
            c1, c2, c3 = st.columns(3)
            with c1:
                st.caption(p_med)
            with c2:
                st.caption(p_mean)
            with c3:
                st.caption(p_std)

            st.plotly_chart(stock_fig, use_container_width=True)
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
