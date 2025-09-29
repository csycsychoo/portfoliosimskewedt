# Migration Notes: Python/Streamlit to Next.js

## Critical Implementation: Skewed Student-t Distribution

The most important aspect of this conversion is the accurate implementation of **Hansen's Skewed Student-t distribution**. This is essential for maintaining the statistical accuracy of the portfolio simulator.

### Why This Matters

The skewed Student-t distribution is used to model stock returns because:

1. **Fat Tails**: Real market returns have more extreme events than normal distributions predict
2. **Negative Skew**: Market crashes (large negative returns) are more common than equivalent positive moves
3. **Empirical Accuracy**: This distribution better matches actual historical stock market behavior

### Implementation Details

#### Original Python (using `arch` library)

\`\`\`python
from arch.univariate import SkewStudent

dist = SkewStudent()
p = np.random.uniform(size=n)
logr_unscaled = dist.ppf(p, [nu, lam])
\`\`\`

#### Next.js TypeScript Implementation

Located in `/lib/skewed-t.ts`, the implementation includes:

1. **Gamma Function** (Lanczos approximation)
   - Required for Student's t distribution calculations
   - High accuracy with reasonable performance

2. **Student's t CDF and PPF**
   - CDF: Uses incomplete beta function
   - PPF: Newton-Raphson iteration with normal approximation as initial guess

3. **Hansen's Skewed-t PPF**
   - Implements the asymmetric transformation
   - Handles left and right tails separately
   - Parameters:
     - `nu`: degrees of freedom (controls tail fatness)
     - `lambda`: asymmetry parameter (controls skewness)

### Key Formula

Hansen's skewed-t uses these transformations:

\`\`\`
a = 4 * lambda * (nu - 2) / (nu - 1)
b = sqrt(1 + 3 * lambda^2 - a^2)
\`\`\`

For a given percentile `p`:
- If `p < (1 + lambda) / 2`: Left tail
  - `z = (t * b - a) / (1 + lambda)`
- Otherwise: Right tail
  - `z = (t * b - a) / (1 - lambda)`

Where `t` is the standard Student's t quantile.

### Validation

To validate the implementation:

1. **Mean and Variance**: After standardization, sample should have mean ≈ 0, std ≈ 1
2. **Skewness**: With `lambda = -0.3`, distribution should be left-skewed
3. **Kurtosis**: With `nu = 5`, distribution should have excess kurtosis > 0

### Testing Against Python

You can test the TypeScript implementation against Python:

\`\`\`python
# Python test
from arch.univariate import SkewStudent
import numpy as np

dist = SkewStudent()
p = np.array([0.01, 0.05, 0.1, 0.5, 0.9, 0.95, 0.99])
quantiles = dist.ppf(p, [5.0, -0.3])
print(quantiles)
\`\`\`

Compare with TypeScript:
\`\`\`typescript
const dist = new SkewedStudentT();
const p = [0.01, 0.05, 0.1, 0.5, 0.9, 0.95, 0.99];
const quantiles = dist.ppf(p, [5.0, -0.3]);
console.log(quantiles);
\`\`\`

Results should be very close (within numerical precision).

## Other Key Conversions

### Random Number Generation

**Python (NumPy):**
\`\`\`python
np.random.normal(loc=mean, scale=std, size=n)
\`\`\`

**TypeScript:**
\`\`\`typescript
// Box-Muller transform
function randomNormal(mean: number, stdDev: number, size: number): number[]
\`\`\`

### Statistical Functions

All implemented from scratch to match NumPy/SciPy:
- `mean()`: Arithmetic mean
- `std()`: Standard deviation with optional ddof (degrees of freedom)
- `median()`: Middle value
- `percentile()`: Linear interpolation between data points
- `skewness()`: Sample skewness (adjusted for bias)
- `kurtosis()`: Excess kurtosis (Fisher's definition)

### Matrix Operations

**Python:**
\`\`\`python
matrix = np.zeros((NUM_RUNS, simulation_years))
flat = matrix.flatten()
\`\`\`

**TypeScript:**
\`\`\`typescript
const matrix: number[][] = Array(NUM_RUNS).fill(0).map(() => Array(simulationYears).fill(0));
const flat = matrix.flat();
\`\`\`

### Visualization

**Python (Plotly):**
\`\`\`python
import plotly.express as px
fig = px.ecdf(df, x="Final Real Value")
\`\`\`

**Next.js (React Plotly):**
\`\`\`typescript
import Plot from 'react-plotly.js';
<Plot data={[{ x: sortedValues, y: ecdfY, type: 'scatter' }]} />
\`\`\`

## Performance Considerations

1. **Client-Side Execution**: All simulations run in the browser
   - Pros: No server costs, instant feedback
   - Cons: Limited by client hardware

2. **Optimization Opportunities**:
   - Web Workers for parallel simulation
   - WASM for numerical computations
   - Progressive rendering for large datasets

3. **Current Performance**: ~2-5 seconds for 10,000 runs on modern hardware

## UI/UX Improvements Over Streamlit

1. **Responsive Design**: Mobile-friendly with Tailwind CSS
2. **Instant Updates**: React state management for smooth interactions
3. **Better Preset System**: Visual feedback for active presets
4. **Modern Styling**: Clean, professional appearance
5. **Collapsible Sections**: Reduced clutter with details/summary

## Deployment Options

1. **Vercel** (Recommended)
   \`\`\`bash
   npm run build
   vercel deploy
   \`\`\`

2. **Netlify**
   - Build command: `npm run build`
   - Publish directory: `.next`

3. **Static Export** (for any static host)
   \`\`\`bash
   npm run build
   npm run export
   \`\`\`

## Maintenance

When updating parameters or formulas, key files to modify:

- `/lib/skewed-t.ts`: Distribution implementation
- `/lib/simulation.ts`: Simulation logic and constants
- `/components/SimulatorForm.tsx`: UI controls and presets
- `/components/ResultsDisplay.tsx`: Charts and metrics

## Known Differences from Python Version

1. **Numerical Precision**: Minor differences due to floating-point implementation
2. **Random Seeds**: No equivalent to NumPy's random seed (each run is truly random)
3. **Performance**: May be slower on low-end devices (Python runs server-side)
4. **Interactivity**: Charts are static by default (can enable zoom if needed)

## Future Enhancements

Potential improvements:
1. Add export to PDF/Excel functionality
2. Implement scenario comparison (side-by-side)
3. Add historical data overlay
4. Monte Carlo convergence diagnostics
5. Sensitivity analysis tools