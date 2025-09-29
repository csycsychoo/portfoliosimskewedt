'use client';

import { SimulationResult, SimulationParams, mean, median, std, percentile, skewness, kurtosis, calculateNegativeReturnPercentages, SNP7424_GEOM_MEAN, SNP7424_LOG_STD, PLOT_CLIP_LOW, PLOT_CLIP_HIGH } from '@/lib/simulation';
import dynamic from 'next/dynamic';

// Dynamically import Plot to avoid SSR issues
const Plot = dynamic(() => import('react-plotly.js'), { ssr: false });

interface Props {
  results: SimulationResult;
  params: SimulationParams;
}

export default function ResultsDisplay({ results, params }: Props) {
  const { finalReal, stockReturns, portfolioReturns } = results;

  // Calculate statistics
  const medianVal = median(finalReal);
  const meanVal = mean(finalReal);
  const stdVal = std(finalReal, 0);
  const successRate = (finalReal.filter(v => v > 0).length / finalReal.length) * 100;
  const outperformRate = (finalReal.filter(v => v > params.startValue).length / finalReal.length) * 100;

  // Stock statistics
  const stockFlat = stockReturns.flat();
  const stockMedian = median(stockFlat);
  const stockMean = mean(stockFlat);
  const stockStd = std(stockFlat, 0);
  const stockSkew = skewness(stockFlat);
  const stockKurt = kurtosis(stockFlat);

  // Percentiles for table
  const percentileLevels = [10, 20, 30, 40, 50, 60, 70, 80, 90];
  const percentileValues = percentileLevels.map(p => percentile(finalReal, p));

  // ECDF data for final values
  const ql = percentile(finalReal, PLOT_CLIP_LOW);
  const qh = percentile(finalReal, PLOT_CLIP_HIGH);
  const clippedValues = finalReal.filter(v => v >= ql && v <= qh);
  const sortedValues = [...clippedValues].sort((a, b) => a - b);
  const ecdfY = sortedValues.map((_, i) => (i + 1) / sortedValues.length);

  // Stock returns histogram
  const sQl = percentile(stockFlat, PLOT_CLIP_LOW);
  const sQh = percentile(stockFlat, PLOT_CLIP_HIGH);
  const stockClipped = stockFlat.filter(v => v >= sQl && v <= sQh);

  // Negative returns analysis
  const negativeReturns = calculateNegativeReturnPercentages(
    stockReturns,
    portfolioReturns,
    params.stockLogLoc,
    params.stockLogScale
  );

  // Historical S&P data
  const spHistorical = [12.5, 10.0, 7.5, 5.0, 2.5, 2.5, 0.0, 0.0, 0.0];

  // Extract data for line chart
  const thresholds = negativeReturns.map(r => r.threshold);
  const stockPcts = negativeReturns.map(r => parseFloat(r.stock));
  const portfolioPcts = negativeReturns.map(r => parseFloat(r.portfolio));
  const normalPcts = negativeReturns.map(r => parseFloat(r.normal));

  return (
    <div className="space-y-6">
      {/* Key Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="bg-white rounded-lg shadow-md p-6">
          <h3 className="text-sm text-gray-600 mb-2">Median Value at end (today&apos;s money)</h3>
          <p className="text-3xl font-bold text-blue-600">
            ${Math.round(medianVal).toLocaleString()}
          </p>
        </div>
        
        <div className="bg-white rounded-lg shadow-md p-6">
          <h3 className="text-sm text-gray-600 mb-2">Likelihood do not run out of money</h3>
          <p className="text-3xl font-bold text-green-600">{successRate.toFixed(1)}%</p>
          <p className="text-xs text-gray-500 mt-1">% runs end value more than zero</p>
        </div>
        
        <div className="bg-white rounded-lg shadow-md p-6">
          <h3 className="text-sm text-gray-600 mb-2">Likelihood end with more money than started</h3>
          <p className="text-3xl font-bold text-purple-600">{outperformRate.toFixed(1)}%</p>
          <p className="text-xs text-gray-500 mt-1">in real terms</p>
        </div>
      </div>

      {/* Percentiles Table */}
      <div className="bg-white rounded-lg shadow-md p-6">
        <h2 className="text-xl font-semibold mb-4">Ending portfolio values in today&apos;s money</h2>
        <div className="overflow-x-auto">
          <table className="min-w-full">
            <thead>
              <tr className="border-b">
                <th className="text-left py-2 px-4">Percentile</th>
                <th className="text-right py-2 px-4">Value</th>
              </tr>
            </thead>
            <tbody>
              {percentileLevels.map((p, i) => (
                <tr key={p} className="border-b hover:bg-gray-50">
                  <td className="py-2 px-4">{p}%</td>
                  <td className="text-right py-2 px-4">${Math.round(percentileValues[i]).toLocaleString()}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Negative Returns Analysis */}
      <div className="bg-white rounded-lg shadow-md p-6">
        <h2 className="text-xl font-semibold mb-4">
          % of years with extreme negative returns compared to S&P historical and normal distribution
        </h2>
        
        <div className="mb-6">
          <Plot
            data={[
              {
                x: thresholds,
                y: stockPcts,
                type: 'scatter',
                mode: 'lines+markers',
                name: 'Stock in your portfolio',
                line: { color: '#3b82f6' }
              },
              {
                x: thresholds,
                y: portfolioPcts,
                type: 'scatter',
                mode: 'lines+markers',
                name: 'Your overall portfolio',
                line: { color: '#10b981' }
              },
              {
                x: thresholds,
                y: normalPcts,
                type: 'scatter',
                mode: 'lines+markers',
                name: "Normal distribution with same mean and vol as S&P '74 to '24",
                line: { color: '#f59e0b' }
              },
              {
                x: thresholds,
                y: spHistorical,
                type: 'scatter',
                mode: 'lines+markers',
                name: 'S&P 500 from 1974 to 2024',
                line: { color: '#ef4444' }
              }
            ]}
            layout={{
              xaxis: { 
                title: '', 
                tickangle: -30,
                fixedrange: true
              },
              yaxis: { 
                title: '% of years',
                ticksuffix: '%',
                fixedrange: true
              },
              legend: {
                orientation: 'h',
                yanchor: 'bottom',
                y: 1.02,
                xanchor: 'center',
                x: 0.5
              },
              margin: { l: 50, r: 10, t: 10, b: 50 },
              height: 400
            }}
            config={{
              displayModeBar: false,
              staticPlot: true
            }}
            useResizeHandler
            style={{ width: '100%' }}
          />
        </div>
      </div>

      {/* ECDF Chart */}
      <div className="bg-white rounded-lg shadow-md p-6">
        <h2 className="text-xl font-semibold mb-4">
          Probability you have this amount or less after {params.simulationYears}y
        </h2>
        
        <Plot
          data={[
            {
              x: sortedValues,
              y: ecdfY,
              type: 'scatter',
              mode: 'lines',
              line: { color: '#3b82f6', width: 2 }
            }
          ]}
          layout={{
            xaxis: { 
              title: 'Final Real Value',
              rangemode: 'tozero',
              fixedrange: true
            },
            yaxis: { 
              title: '% of sims',
              tickformat: '.0%',
              rangemode: 'tozero',
              fixedrange: true
            },
            shapes: [
              {
                type: 'line',
                x0: params.startValue,
                x1: params.startValue,
                y0: 0,
                y1: 1,
                line: { color: 'red', width: 2, dash: 'dash' }
              },
              {
                type: 'line',
                x0: medianVal,
                x1: medianVal,
                y0: 0,
                y1: 1,
                line: { color: 'green', width: 2, dash: 'dash' }
              }
            ],
            height: 400
          }}
          config={{
            displayModeBar: false,
            staticPlot: true
          }}
          useResizeHandler
          style={{ width: '100%' }}
        />
        
        <div className="grid grid-cols-3 gap-4 mt-4 text-sm text-gray-600">
          <div>Median: ${Math.round(medianVal).toLocaleString()}</div>
          <div>Mean: ${Math.round(meanVal).toLocaleString()}</div>
          <div>Std Dev: ${Math.round(stdVal).toLocaleString()}</div>
        </div>
      </div>

      {/* Stock Returns Histogram */}
      <div className="bg-white rounded-lg shadow-md p-6">
        <h2 className="text-xl font-semibold mb-4">Annual Stock Returns</h2>
        
        <Plot
          data={[
            {
              x: stockClipped,
              type: 'histogram',
              nbinsx: 200,
              histnorm: 'percent',
              marker: { color: '#3b82f6' }
            }
          ]}
          layout={{
            xaxis: { 
              title: 'Annual Return',
              tickformat: '.2%',
              fixedrange: true
            },
            yaxis: { 
              title: '%',
              fixedrange: true
            },
            height: 400
          }}
          config={{
            displayModeBar: false,
            staticPlot: true
          }}
          useResizeHandler
          style={{ width: '100%' }}
        />
        
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-2 mt-4 text-xs text-gray-600">
          <div>Median of simple returns: {(stockMedian * 100).toFixed(2)}%</div>
          <div>Arithmetic Mean of simple returns: {(stockMean * 100).toFixed(2)}%</div>
          <div>Std Dev of simple returns: {(stockStd * 100).toFixed(2)}%</div>
          <div>Skew of simple returns: {stockSkew.toFixed(2)}</div>
          <div>Kurtosis (fat tails) of simple returns: {stockKurt.toFixed(2)}</div>
        </div>
      </div>
    </div>
  );
}