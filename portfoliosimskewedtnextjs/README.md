# portfoliosimskewedtnextjs

Next.js + Vercel port of the Streamlit retirement Monte Carlo simulator.
Numerical engine is Python (numpy/scipy/arch); UI is Next.js (App Router) with
Plotly charts; deployed on Vercel with a Python serverless function.

## Layout

```
api/
  simulate.py              # Vercel Python serverless handler (POST /api/simulate)
  py/engine.py             # Pure-compute engine, refactored from the Streamlit prototype
app/
  page.tsx                 # Main UI (sidebar form + results)
  layout.tsx, globals.css
components/charts/         # Plotly chart components
lib/
  types.ts                 # Shared TS types for the API contract
  format.ts, presets.ts
tests/
  test_numerics.py         # Unit tests for numerical primitives
  test_golden.py           # Golden-master snapshot test (locks engine output)
  golden/snapshot.json     # Frozen expected output for seed=42 scenario
requirements.txt           # Python deps pinned to match original Streamlit app
package.json               # Next.js deps
vercel.json                # 60s function timeout, 1024MB memory
```

## Numerical correctness

The engine was refactored from `portfoliosimskewedt.py` with these substantive
changes:

| Change | Why |
|---|---|
| Added `seed` parameter via `np.random.default_rng` | Makes runs reproducible; required for tests |
| Added `num_runs` parameter | Exposed for tuning (was hard-coded at 10,000) |
| Dropped unused `stock_log_mu`/`stock_log_sigma` args from `calculate_negative_return_percentages` | Were dead arguments; normal comparison always uses S&P '74-'24 constants |
| Fixed strict-`<` vs "at or below" docstring inconsistency | Matches actual code semantics |
| In non-rebalance + start-of-year branch, compute `r_port` after withdrawal | Consistent with rebalance branch |
| Removed all Streamlit imports | Engine is pure compute |

Math that is unchanged and verified:
- Hansen skewed-t draw with empirical mean-0/std-1 standardization
- `expm1` log→simple conversion with `-0.99` floor
- Mid-year half-growth → withdraw → half-growth timing
- Proportional withdrawal with safe-divide on zero totals
- Real-terms reporting via `cumulative_inflation` denominator
- Normal-comparison cell uses `Φ((ln(1+k) − μ)/σ)` with S&P '74-'24 constants

## Running locally

### Python tests

```bash
pip install -r requirements.txt
pip install pytest
pytest tests/ -q
```

### Regenerate golden snapshot (only if engine intentionally changed)

```bash
REGEN_GOLDEN=1 pytest tests/test_golden.py
```

### Next.js dev

```bash
npm install
npm run dev
# Open http://localhost:3000
```

Note: `npm run dev` alone won't run the Python serverless function. For
end-to-end local testing including the API, use:

```bash
npm install -g vercel
vercel dev
```

## API contract

`POST /api/simulate` — see `lib/types.ts` for `SimulationRequest` and
`SimulationResponse`. The `seed` field is optional; pass an integer for
reproducible runs (e.g. for testing).

## Deploy

```bash
vercel deploy
```

The Python function is auto-detected via `api/simulate.py` and pinned to
`@vercel/python@4.3.0` in `vercel.json`. Function duration is set to 60s
(Vercel Pro). For Hobby plan reduce `numRuns` in requests to fit under 10s.
