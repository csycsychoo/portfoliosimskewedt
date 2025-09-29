# Quick Start Guide

## Installation and Setup

1. **Install dependencies:**

\`\`\`bash
npm install
\`\`\`

2. **Run development server:**

\`\`\`bash
npm run dev
\`\`\`

3. **Open your browser:**

Navigate to [http://localhost:3000](http://localhost:3000)

## First Simulation

1. **Use a preset** (recommended for first run):
   - Click "🇺🇸 USA Today" button for modern US market assumptions

2. **Adjust basic parameters:**
   - Starting Portfolio: $5,000,000 (default)
   - Annual Withdrawal: $200,000 (default)
   - Stock %: 70% (default)

3. **Click "🚀 Run Simulation"**

4. **Review results:**
   - Check the median final value
   - Review success rate (% chance of not running out of money)
   - Examine the probability distribution charts

## Understanding Your First Results

### Key Metrics
- **Median Value**: The middle outcome - half of simulations end higher, half lower
- **Success Rate**: % of simulations where you don't run out of money
- **Outperform Rate**: % of simulations where you end with more than you started (in real terms)

### Charts

1. **ECDF Chart** (top): 
   - Shows probability of ending with X dollars or less
   - Red line = your starting amount
   - Green line = median outcome

2. **Negative Returns Analysis**:
   - Compares your portfolio to historical S&P 500
   - Shows how often you experience severe losses
   - Notice the difference between "Normal" and actual distributions

3. **Stock Returns Histogram**:
   - Shows distribution of annual stock returns
   - Note the skewness and fat tails

## Customization Examples

### Conservative Retiree
- Stock %: 50%
- Withdrawal: Lower amount
- Preset: Trinity Study (enable rebalancing)

### Aggressive Growth
- Stock %: 90%
- Preset: USA Today
- Check "Advanced Options" and adjust Nu (fat tails) to 3.0 for more extreme events

### International Portfolio
- Preset: Singapore
- Adjust stock % based on risk tolerance
- Consider higher stock vol in advanced options

## Troubleshooting

### Slow Performance
- Simulations run 10,000 iterations in your browser
- Expected time: 2-5 seconds on modern hardware
- If slower, try closing other browser tabs

### Charts Not Showing
- Ensure JavaScript is enabled
- Try refreshing the page
- Check browser console for errors

### Installation Issues
- Make sure Node.js 18+ is installed
- Try deleting `node_modules` and running `npm install` again
- Check that all dependencies installed correctly

## Next Steps

1. **Experiment with presets**: Try all three presets to see different scenarios
2. **Adjust withdrawal amounts**: See how this affects success rate
3. **Explore advanced options**: Fine-tune fat tails and skewness
4. **Compare scenarios**: Run multiple simulations with different parameters

## Need Help?

See the full [README.md](README.md) for:
- Detailed technical documentation
- Statistical methodology
- Complete feature list
- Deployment options