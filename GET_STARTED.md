# 🚀 Get Started with Portfolio Simulator

Your Streamlit app has been successfully converted to Next.js!

## What Changed?

✅ **Same Features, Better Platform**
- All simulation logic preserved
- Same statistical accuracy (Skewed Student-t distribution)
- Modern, responsive UI
- Faster, more interactive

## Quick Setup (3 steps)

### 1. Install Dependencies

```bash
npm install
```

This will install:
- Next.js 14 (React framework)
- TypeScript (type safety)
- Tailwind CSS (styling)
- Plotly.js (charts)
- All necessary dependencies

### 2. Start Development Server

```bash
npm run dev
```

### 3. Open in Browser

Navigate to: **http://localhost:3000**

That's it! 🎉

## What You'll See

### Left Panel: Controls
- **Easy defaults**: Preset buttons (USA Today, Trinity Study, Singapore)
- **Initial Setup**: Starting portfolio, withdrawals, simulation years
- **Return assumptions**: Stock returns, cash returns, inflation
- **Policy Options**: Withdrawal timing, rebalancing
- **Advanced Options**: Volatilities, fat tails, skewness

### Right Panel: Results
- **Key Metrics**: Median value, success rate, outperform rate
- **Percentile Table**: Distribution of outcomes
- **Charts**:
  - Probability distribution (ECDF)
  - Negative returns analysis
  - Stock returns histogram

## First Simulation

1. Click **"🇺🇸 USA Today"** preset
2. Adjust starting portfolio if needed (default: $5M)
3. Click **"🚀 Run Simulation"**
4. Wait 2-5 seconds
5. Review results!

## Key Feature: Skewed Student-t Distribution

The heart of this simulator is the **Hansen's Skewed Student-t distribution**, which models:

- **Fat Tails**: Extreme events happen more often than normal distributions predict
- **Negative Skew**: Market crashes are more severe than equivalent rallies
- **Realistic Risk**: Better portfolio failure probability estimates

This was carefully ported from Python's ARCH library to TypeScript to maintain statistical accuracy.

## File Structure

```
/workspace/
├── app/                    # Next.js app directory
│   ├── page.tsx           # Main page
│   ├── layout.tsx         # Root layout
│   └── globals.css        # Global styles
├── components/            # React components
│   ├── SimulatorForm.tsx  # Input form
│   └── ResultsDisplay.tsx # Results visualization
├── lib/                   # Core libraries
│   ├── skewed-t.ts       # Skewed Student-t distribution ⭐
│   ├── simulation.ts     # Monte Carlo engine
│   └── types.ts          # TypeScript types
├── package.json          # Dependencies
├── tsconfig.json         # TypeScript config
└── README.md            # Full documentation
```

## Available Commands

```bash
# Development
npm run dev          # Start dev server (http://localhost:3000)

# Production
npm run build        # Build for production
npm start            # Start production server

# Code Quality
npm run lint         # Check code quality
```

## Verify Setup

Run the verification script:

```bash
./verify-setup.sh
```

This checks:
- Node.js version (18+)
- Required files and directories
- Dependencies installation

## Customization

### Change Presets

Edit `components/SimulatorForm.tsx`:

```typescript
const PRESETS: Preset[] = [
  {
    name: 'Your Custom Preset',
    stockGeomMean: 8.0,
    inflation: 2.5,
    cashReturn: 3.0,
    stockLogVol: 18.0,
    description: 'Your description'
  }
];
```

### Change Simulation Count

Edit `lib/simulation.ts`:

```typescript
export const NUM_RUNS = 10_000;  // Default: 10,000 runs
```

### Adjust Default Parameters

Edit `lib/simulation.ts`:

```typescript
export const DEFAULT_SKEWT_NU = 5.0;      // Fat tails (3-20)
export const DEFAULT_SKEWT_LAMBDA = -0.3;  // Skewness (-0.9 to 0.9)
```

## Deployment

### Option 1: Vercel (Easiest)

```bash
npm install -g vercel
vercel
```

### Option 2: Netlify

1. Push to GitHub
2. Connect to Netlify
3. Build command: `npm run build`
4. Publish directory: `.next`

### Option 3: Docker

```bash
docker build -t portfolio-simulator .
docker run -p 3000:3000 portfolio-simulator
```

## Troubleshooting

### "Module not found" errors

```bash
rm -rf node_modules package-lock.json
npm install
```

### Slow simulation

- Expected: 2-5 seconds for 10,000 runs
- If slower: Check browser console, close other tabs
- Consider reducing NUM_RUNS in `lib/simulation.ts`

### Charts not displaying

- Ensure JavaScript is enabled
- Check browser console for errors
- Try hard refresh (Ctrl+Shift+R)

### TypeScript errors

```bash
npm run build
```

This will show any type errors.

## Documentation

- **README.md**: Comprehensive documentation
- **QUICKSTART.md**: Quick start guide
- **MIGRATION_NOTES.md**: Technical implementation details
- **CONVERSION_SUMMARY.md**: Conversion overview

## Testing

### Manual Testing

1. Run app: `npm run dev`
2. Click "USA Today" preset
3. Click "Run Simulation"
4. Verify:
   - Results appear within ~5 seconds
   - Charts render correctly
   - Median value is reasonable ($5-10M range)
   - Success rate is high (>90%)

### Validate Distribution

The Skewed-t distribution implementation can be validated:

```bash
cd lib/__tests__
node -r ts-node/register skewed-t.test.ts
```

Expected output:
- Sample mean ≈ 0
- Sample skew < 0 (for lambda = -0.3)
- Left tail wider than right tail

## Key Differences from Streamlit

| Feature | Streamlit | Next.js |
|---------|-----------|---------|
| Framework | Python | TypeScript/React |
| Execution | Server-side | Client-side |
| UI Updates | Page reload | Instant React |
| Deployment | Python server | Static or serverless |
| Performance | Server-dependent | Client-dependent |
| Cost | Server costs | Free static hosting |

## Performance Notes

- **Simulation**: 10,000 runs × 50 years = 500,000 calculations
- **Expected Time**: 2-5 seconds on modern hardware
- **Browser Memory**: ~100-200 MB during simulation
- **No Server Required**: All processing in browser

## Support

If you encounter issues:

1. Check browser console (F12) for errors
2. Verify Node.js version: `node -v` (should be 18+)
3. Check all files are present: `./verify-setup.sh`
4. Try clean install: `rm -rf node_modules && npm install`

## Next Steps

1. ✅ Run `npm install`
2. ✅ Run `npm run dev`
3. ✅ Test with USA Today preset
4. 📊 Explore different scenarios
5. 🎨 Customize parameters
6. 🚀 Deploy to production

## Advanced Features

### Web Workers (Future)

For better performance, simulations could run in Web Workers:

```typescript
// Future enhancement
const worker = new Worker('simulation-worker.js');
worker.postMessage(params);
```

### Export Results (Future)

Add export functionality:

```typescript
// Future enhancement
const exportToPDF = () => { /* implementation */ };
const exportToCSV = () => { /* implementation */ };
```

## Version

**v1.3.14** - Matches original Streamlit app version

---

## Ready to Go! 🎉

You now have a modern, production-ready Next.js app with the same statistical accuracy as your original Streamlit application.

**Next command:**
```bash
npm install && npm run dev
```

Then open: **http://localhost:3000**

Happy simulating! 📈