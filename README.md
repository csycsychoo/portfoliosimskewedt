# Portfolio Monte Carlo Simulator - Next.js

A modern Next.js implementation of a retirement portfolio simulator with extreme events modeling (fat tails, negative skew).

## Features

- **Monte Carlo Simulation**: 10,000 simulation runs over customizable time periods
- **Skewed Student-t Distribution**: Accurate modeling of extreme market events with fat tails and negative skew
- **Interactive UI**: Real-time parameter adjustments with preset scenarios (USA Today, Trinity Study, Singapore)
- **Comprehensive Visualizations**:
  - Probability distribution of final portfolio values (ECDF)
  - Annual stock returns histogram
  - Negative returns analysis vs. historical S&P 500 and normal distribution
- **Flexible Options**:
  - Customizable withdrawal timing (start of year or mid-year)
  - Annual rebalancing toggle
  - Advanced parameters for inflation, volatility, and distribution shape

## Technology Stack

- **Next.js 14** - React framework with App Router
- **TypeScript** - Type-safe code
- **Tailwind CSS** - Modern, responsive styling
- **Plotly.js** - Interactive data visualizations
- **Custom Statistical Library** - Accurate implementation of Hansen's Skewed Student-t distribution

## Key Implementation Details

### Skewed Student-t Distribution

The most critical part of this conversion is the accurate implementation of the **Hansen's Skewed Student-t distribution**. This distribution is used to model stock returns with:

1. **Fat Tails** (controlled by `nu` parameter): Captures extreme market events that occur more frequently than normal distributions predict
2. **Negative Skew** (controlled by `lambda` parameter): Models the empirical observation that market crashes (large negative returns) are more common than equivalent positive moves

The implementation in `/lib/skewed-t.ts` includes:

- **PPF (Percent Point Function)**: Inverse CDF that converts uniform random samples to skewed-t distributed values
- **Hansen's Parameterization**: Proper scaling and asymmetry handling
- **Numerical Stability**: Careful handling of edge cases and numerical precision

This matches the behavior of Python's `arch.univariate.SkewStudent()` library used in the original Streamlit app.

### Simulation Logic

The core simulation (`/lib/simulation.ts`) faithfully reproduces the Python implementation:

1. **Log Returns**: Stock returns are modeled as skewed-t distributed log returns, then converted to simple returns
2. **Standardization**: Generated samples are standardized to ensure correct mean and variance
3. **Withdrawal Strategies**: Supports both start-of-year and mid-year withdrawals with proper compounding
4. **Rebalancing**: Optional annual rebalancing to maintain target asset allocation
5. **Inflation Adjustment**: All final values are presented in real (inflation-adjusted) terms

## Getting Started

### Prerequisites

- Node.js 18.x or higher
- npm or yarn

### Installation

1. Install dependencies:

\`\`\`bash
npm install
\`\`\`

2. Run the development server:

\`\`\`bash
npm run dev
\`\`\`

3. Open [http://localhost:3000](http://localhost:3000) in your browser

### Building for Production

\`\`\`bash
npm run build
npm start
\`\`\`

## Usage

### Quick Start with Presets

1. Click one of the preset buttons:
   - **🇺🇸 USA Today**: Modern US market assumptions (8.67% stock return, 2.9% inflation)
   - **📊 Trinity Study**: Historical Trinity Study parameters (10.6% stock return, automatic rebalancing)
   - **🇸🇬 Singapore**: Singapore market assumptions (7% stock return, 1.7% inflation)

2. Adjust the starting portfolio and annual withdrawal amounts

3. Click "🚀 Run Simulation"

### Advanced Customization

Expand the "Advanced Options" section to fine-tune:

- **Inflation Vol**: Volatility of inflation rates
- **Cash/Bond Vol**: Volatility of cash/bond returns
- **Stock Log Vol**: Volatility of stock log returns (standard deviation)
- **Fat Tails (Nu)**: Lower values = fatter tails (more extreme events). Range: 3-20, default: 5
- **Skewness (Lambda)**: Negative values = left skew (more severe crashes). Range: -0.9 to 0.9, default: -0.3

## Understanding the Results

### Key Metrics

1. **Median Value**: The middle outcome - 50% of simulations end higher, 50% lower
2. **Success Rate**: Percentage of simulations where portfolio doesn't run out of money
3. **Outperform Rate**: Percentage of simulations where final value exceeds starting value (in real terms)

### Visualizations

1. **ECDF Chart**: Shows the probability of ending with a given amount or less
   - Red dashed line: Starting portfolio value
   - Green dashed line: Median outcome

2. **Negative Returns Analysis**: Compares your portfolio's extreme negative returns to:
   - Historical S&P 500 (1974-2024)
   - Normal distribution with same mean and volatility
   - Shows how fat tails and skew affect downside risk

3. **Stock Returns Histogram**: Distribution of annual stock returns
   - Shows skewness and kurtosis statistics
   - Reveals the fat-tailed, negatively-skewed nature of returns

## Technical Notes

### Why Skewed Student-t?

Normal distributions underestimate the probability of extreme events. The skewed Student-t distribution better captures:

- **Market Crashes**: The empirical fact that large negative returns happen more often than normal distributions predict
- **Asymmetry**: Markets tend to go up gradually but crash suddenly (negative skew)
- **Fat Tails**: Extreme events (both positive and negative) occur more frequently than in normal distributions

### Numerical Stability

The implementation includes several safety measures:

- Minimum return floors to prevent numerical instability
- Careful handling of division by zero
- Epsilon values for numerical precision
- Standardization of samples before applying target parameters

### Performance

The app runs 10,000 Monte Carlo simulations client-side. On modern hardware:

- Simulation typically completes in 2-5 seconds
- All calculations run in the browser (no server required)
- Results are cached until parameters change

## Project Structure

\`\`\`
/workspace
├── app/
│   ├── layout.tsx          # Root layout
│   ├── page.tsx            # Main page component
│   └── globals.css         # Global styles
├── components/
│   ├── SimulatorForm.tsx   # Input form with presets
│   └── ResultsDisplay.tsx  # Results and visualizations
├── lib/
│   ├── skewed-t.ts         # Skewed Student-t distribution
│   └── simulation.ts       # Monte Carlo simulation engine
├── package.json
├── tsconfig.json
├── tailwind.config.js
└── next.config.js
\`\`\`

## Conversion from Python

This Next.js app is a faithful port of the original Python/Streamlit application. Key conversion considerations:

1. **Statistical Functions**: All statistical functions (mean, std, percentile, skewness, kurtosis) are implemented from scratch to match NumPy/SciPy behavior

2. **Skewed-t Distribution**: The most critical component, carefully ported to match `arch.univariate.SkewStudent()` behavior using Hansen's parameterization

3. **Random Number Generation**: Uses Box-Muller transform for normal distributions and inverse transform sampling for skewed-t

4. **Visualization**: Plotly.js provides equivalent functionality to Python's Plotly, with the same chart types and interactivity

## License

This project is provided as-is for educational and planning purposes. Always consult with a qualified financial advisor for retirement planning decisions.

## Version

v1.3.14 - Matches the original Streamlit application version