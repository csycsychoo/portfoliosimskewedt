'use client';

import { useState } from 'react';
import { SimulationParams, DEFAULT_SKEWT_NU, DEFAULT_SKEWT_LAMBDA, DEFAULT_STOCK_LOG_LOC, DEFAULT_STOCK_LOG_SCALE, SIMULATION_YEARS } from '@/lib/simulation';

interface Preset {
  name: string;
  stockGeomMean: number;
  inflation: number;
  cashReturn: number;
  stockLogVol: number;
  description: string;
  rebalance?: boolean;
}

const PRESETS: Preset[] = [
  {
    name: 'USA Today',
    stockGeomMean: 8.67,
    inflation: 2.9,
    cashReturn: 4.8,
    stockLogVol: 17.0,
    description: 'Stock geo return 8.67%, Inflation 2.9%, Cash return 4.8%'
  },
  {
    name: 'Trinity Study',
    stockGeomMean: 10.6,
    inflation: 2.96,
    cashReturn: 5.7,
    stockLogVol: 17.0,
    description: 'Stock geo return 10.6%, Inflation 2.96%, Cash return 5.7%',
    rebalance: true
  },
  {
    name: 'Singapore',
    stockGeomMean: 7.0,
    inflation: 1.7,
    cashReturn: 2.5,
    stockLogVol: 20.0,
    description: 'Stock geo return 7%, Inflation 1.7%, Cash return 2.5%'
  }
];

interface Props {
  onRunSimulation: (params: SimulationParams) => void;
  isSimulating: boolean;
}

export default function SimulatorForm({ onRunSimulation, isSimulating }: Props) {
  const defaultGeomMean = (Math.exp(DEFAULT_STOCK_LOG_LOC) - 1) * 100;
  
  const [currentPreset, setCurrentPreset] = useState<string>('Custom');
  const [startValue, setStartValue] = useState(5_000_000);
  const [realSpending, setRealSpending] = useState(200_000);
  const [simulationYears, setSimulationYears] = useState(SIMULATION_YEARS);
  const [stockPropPercent, setStockPropPercent] = useState(70);
  
  const [stockGeomMeanPercent, setStockGeomMeanPercent] = useState(defaultGeomMean);
  const [cashReturnPercent, setCashReturnPercent] = useState(2.5);
  const [inflationRatePercent, setInflationRatePercent] = useState(1.7);
  
  const [withdrawalTiming, setWithdrawalTiming] = useState<'Start of year' | 'Mid-year'>('Mid-year');
  const [rebalanceEachYear, setRebalanceEachYear] = useState(true);
  
  const [inflationVolPercent, setInflationVolPercent] = useState(2.0);
  const [cashVolPercent, setCashVolPercent] = useState(1.0);
  const [stockLogVolPercent, setStockLogVolPercent] = useState(DEFAULT_STOCK_LOG_SCALE * 100);
  const [skewTNu, setSkewTNu] = useState(DEFAULT_SKEWT_NU);
  const [skewTLambda, setSkewTLambda] = useState(DEFAULT_SKEWT_LAMBDA);

  const [showAdvanced, setShowAdvanced] = useState(false);

  const applyPreset = (preset: Preset) => {
    setStockGeomMeanPercent(preset.stockGeomMean);
    setInflationRatePercent(preset.inflation);
    setCashReturnPercent(preset.cashReturn);
    setStockLogVolPercent(preset.stockLogVol);
    if (preset.rebalance !== undefined) {
      setRebalanceEachYear(preset.rebalance);
    }
    setCurrentPreset(preset.name);
  };

  const handleSliderChange = () => {
    // Reset to custom when user manually changes values
    setCurrentPreset('Custom');
  };

  const handleRunSimulation = () => {
    const growth = 1.0 + stockGeomMeanPercent / 100;
    const impliedLogMean = Math.log(Math.max(growth, 1e-9));

    const params: SimulationParams = {
      startValue,
      realSpending,
      stockProp: stockPropPercent / 100,
      cashProp: 1 - stockPropPercent / 100,
      cashReturn: cashReturnPercent / 100,
      cashVol: cashVolPercent / 100,
      inflationRate: inflationRatePercent / 100,
      inflationVol: inflationVolPercent / 100,
      stockLogLoc: impliedLogMean,
      stockLogScale: stockLogVolPercent / 100,
      skewTNu,
      skewTLambda,
      simulationYears,
      withdrawalTiming,
      rebalanceEachYear
    };

    onRunSimulation(params);
  };

  const equityRiskPremium = stockGeomMeanPercent - cashReturnPercent;

  return (
    <div className="bg-white rounded-lg shadow-md p-6 space-y-6">
      {/* Easy Defaults */}
      <div>
        <h2 className="text-xl font-semibold mb-3 flex items-center">
          <span className="mr-2">🎯</span>
          Easy defaults
        </h2>
        <p className="text-sm text-gray-600 mb-3">Quick preset buttons for common scenarios:</p>
        
        {currentPreset !== 'Custom' && (
          <div className="mb-3 p-2 bg-green-50 border border-green-200 rounded text-sm text-green-800">
            ✅ Currently using: {currentPreset}
            {currentPreset === 'Trinity Study' && ' (with rebalancing enabled)'}
          </div>
        )}
        
        <div className="grid grid-cols-3 gap-2">
          {PRESETS.map((preset) => (
            <button
              key={preset.name}
              onClick={() => applyPreset(preset)}
              className="px-2 py-2 text-xs bg-blue-500 text-white rounded hover:bg-blue-600 transition"
              title={preset.description}
            >
              {preset.name === 'USA Today' && '🇺🇸 '}
              {preset.name === 'Trinity Study' && '📊 '}
              {preset.name === 'Singapore' && '🇸🇬 '}
              {preset.name}
            </button>
          ))}
        </div>
      </div>

      <hr />

      {/* Initial Setup */}
      <details open>
        <summary className="font-semibold cursor-pointer mb-3">Initial Setup</summary>
        <div className="space-y-3">
          <div>
            <label className="block text-sm font-medium mb-1">Starting Portfolio ($)</label>
            <input
              type="number"
              value={startValue}
              onChange={(e) => setStartValue(Number(e.target.value))}
              step={50000}
              min={0}
              className="w-full px-3 py-2 border rounded focus:ring-2 focus:ring-blue-500"
            />
          </div>
          
          <div>
            <label className="block text-sm font-medium mb-1">Annual Withdrawal (grows with inflation) ($)</label>
            <input
              type="number"
              value={realSpending}
              onChange={(e) => setRealSpending(Number(e.target.value))}
              step={1000}
              min={0}
              className="w-full px-3 py-2 border rounded focus:ring-2 focus:ring-blue-500"
            />
          </div>
          
          <div>
            <label className="block text-sm font-medium mb-1">Simulation Years</label>
            <input
              type="number"
              value={simulationYears}
              onChange={(e) => setSimulationYears(Number(e.target.value))}
              step={1}
              min={1}
              className="w-full px-3 py-2 border rounded focus:ring-2 focus:ring-blue-500"
            />
          </div>
          
          <div>
            <label className="block text-sm font-medium mb-1">Stock % in Portfolio: {stockPropPercent}%</label>
            <input
              type="range"
              value={stockPropPercent}
              onChange={(e) => setStockPropPercent(Number(e.target.value))}
              min={0}
              max={100}
              step={5}
              className="w-full"
            />
          </div>
        </div>
      </details>

      <hr />

      {/* Return and economic assumptions */}
      <details open>
        <summary className="font-semibold cursor-pointer mb-3">Return and economic assumptions</summary>
        <div className="space-y-3">
          <div>
            <label className="block text-sm font-medium mb-1">
              Stock Average Return % (Geometric mean): {stockGeomMeanPercent.toFixed(2)}%
            </label>
            <input
              type="range"
              value={stockGeomMeanPercent}
              onChange={(e) => { setStockGeomMeanPercent(Number(e.target.value)); handleSliderChange(); }}
              min={-20}
              max={30}
              step={0.1}
              className="w-full"
            />
          </div>
          
          <div>
            <label className="block text-sm font-medium mb-1">
              Cash/Bond Return: {cashReturnPercent.toFixed(2)}%
            </label>
            <input
              type="range"
              value={cashReturnPercent}
              onChange={(e) => { setCashReturnPercent(Number(e.target.value)); handleSliderChange(); }}
              min={0}
              max={10}
              step={0.1}
              className="w-full"
            />
          </div>
          
          <div className="text-sm text-gray-600 bg-gray-50 p-2 rounded">
            Implied Equity Risk Premium (stock return - cash): {equityRiskPremium.toFixed(2)}%
          </div>
          
          <div>
            <label className="block text-sm font-medium mb-1">
              Inflation Mean: {inflationRatePercent.toFixed(2)}%
            </label>
            <input
              type="range"
              value={inflationRatePercent}
              onChange={(e) => { setInflationRatePercent(Number(e.target.value)); handleSliderChange(); }}
              min={0}
              max={10}
              step={0.1}
              className="w-full"
            />
          </div>
        </div>
      </details>

      <hr />

      {/* Policy Options */}
      <details open>
        <summary className="font-semibold cursor-pointer mb-3">Policy Options</summary>
        <div className="space-y-3">
          <div>
            <label className="block text-sm font-medium mb-2">Withdrawal Timing</label>
            <div className="space-y-2">
              <label className="flex items-center">
                <input
                  type="radio"
                  checked={withdrawalTiming === 'Start of year'}
                  onChange={() => setWithdrawalTiming('Start of year')}
                  className="mr-2"
                />
                Start of year
              </label>
              <label className="flex items-center">
                <input
                  type="radio"
                  checked={withdrawalTiming === 'Mid-year'}
                  onChange={() => setWithdrawalTiming('Mid-year')}
                  className="mr-2"
                />
                Mid-year
              </label>
            </div>
          </div>
          
          <div>
            <label className="flex items-center">
              <input
                type="checkbox"
                checked={rebalanceEachYear}
                onChange={(e) => setRebalanceEachYear(e.target.checked)}
                className="mr-2"
              />
              Rebalance annually
            </label>
            
            {rebalanceEachYear ? (
              <div className="mt-2 text-xs text-blue-700 bg-blue-50 p-2 rounded">
                🔄 Rebalancing is enabled - portfolio will be rebalanced to target allocation each year
              </div>
            ) : (
              <div className="mt-2 text-xs text-yellow-700 bg-yellow-50 p-2 rounded">
                ⚠️ Rebalancing is disabled - portfolio allocation will drift over time
              </div>
            )}
          </div>
        </div>
      </details>

      <hr />

      {/* Advanced Options */}
      <details>
        <summary className="font-semibold cursor-pointer mb-3">Advanced Options</summary>
        <div className="space-y-3">
          <div>
            <label className="block text-sm font-medium mb-1">
              Inflation Vol: {inflationVolPercent.toFixed(2)}%
            </label>
            <input
              type="range"
              value={inflationVolPercent}
              onChange={(e) => setInflationVolPercent(Number(e.target.value))}
              min={0}
              max={5}
              step={0.1}
              className="w-full"
            />
          </div>
          
          <div>
            <label className="block text-sm font-medium mb-1">
              Cash/Bond Vol: {cashVolPercent.toFixed(2)}%
            </label>
            <input
              type="range"
              value={cashVolPercent}
              onChange={(e) => setCashVolPercent(Number(e.target.value))}
              min={0}
              max={5}
              step={0.1}
              className="w-full"
            />
          </div>
          
          <div>
            <label className="block text-sm font-medium mb-1">
              Stock Log Vol: {stockLogVolPercent.toFixed(2)}%
            </label>
            <input
              type="range"
              value={stockLogVolPercent}
              onChange={(e) => setStockLogVolPercent(Number(e.target.value))}
              min={5}
              max={50}
              step={0.5}
              className="w-full"
            />
          </div>
          
          <div>
            <label className="block text-sm font-medium mb-1">
              Fat Tails (Nu): {skewTNu.toFixed(2)}
            </label>
            <input
              type="range"
              value={skewTNu}
              onChange={(e) => setSkewTNu(Number(e.target.value))}
              min={3}
              max={20}
              step={0.5}
              className="w-full"
              title="Lower value = fatter tails"
            />
            <p className="text-xs text-gray-500 mt-1">Lower value = fatter tails</p>
          </div>
          
          <div>
            <label className="block text-sm font-medium mb-1">
              Skewness (Lambda): {skewTLambda.toFixed(2)}
            </label>
            <input
              type="range"
              value={skewTLambda}
              onChange={(e) => setSkewTLambda(Number(e.target.value))}
              min={-0.9}
              max={0.9}
              step={0.05}
              className="w-full"
              title="Negative = left skew"
            />
            <p className="text-xs text-gray-500 mt-1">Negative = left skew</p>
          </div>
        </div>
      </details>

      <button
        onClick={handleRunSimulation}
        disabled={isSimulating}
        className="w-full bg-blue-600 text-white py-3 px-4 rounded-lg font-semibold hover:bg-blue-700 transition disabled:bg-gray-400 disabled:cursor-not-allowed"
      >
        {isSimulating ? 'Running...' : '🚀 Run Simulation'}
      </button>
    </div>
  );
}