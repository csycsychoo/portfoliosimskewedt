/**
 * Portfolio Monte Carlo Simulation Library
 * Ported from Python (portfoliosimskewedt.py)
 */

import { SkewedStudentT } from './skewed-t';

// Configuration constants
export const SIMULATION_YEARS = 50;
export const NUM_RUNS = 10_000;
export const APP_VERSION = "v1.3.14";

// Default parameters
export const DEFAULT_SKEWT_NU = 5.0;
export const DEFAULT_SKEWT_LAMBDA = -0.3;
export const DEFAULT_STOCK_LOG_LOC = 0.067659; // 7% geometric => ln(1.07)
export const DEFAULT_STOCK_LOG_SCALE = 0.17; // 17% log scale
export const MIN_SIMPLE_RETURN = -0.99;

// Chart clipping
export const PLOT_CLIP_LOW = 1;
export const PLOT_CLIP_HIGH = 99;

// Numerical safety constants
export const EPS = 1e-12;
export const SQRT_FLOOR = 1e-12;
export const MIN_R_FOR_SQRT = -0.999999;
export const MIN_INFL_FACTOR = 1e-9;

// S&P 500 1974-2024 parameters for Normal comparison
export const SNP7424_GEOM_MEAN = 0.1144; // 11.44%
export const SNP7424_LOG_STD = 0.168824; // 16.8824%

/**
 * Statistical helper functions
 */

/**
 * Calculate mean of an array
 */
export function mean(arr: number[]): number {
  return arr.reduce((sum, val) => sum + val, 0) / arr.length;
}

/**
 * Calculate standard deviation
 */
export function std(arr: number[], ddof: number = 0): number {
  const m = mean(arr);
  const variance = arr.reduce((sum, val) => sum + Math.pow(val - m, 2), 0) / (arr.length - ddof);
  return Math.sqrt(variance);
}

/**
 * Calculate median
 */
export function median(arr: number[]): number {
  const sorted = [...arr].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0 ? (sorted[mid - 1] + sorted[mid]) / 2 : sorted[mid];
}

/**
 * Calculate percentile
 */
export function percentile(arr: number[], p: number): number {
  const sorted = [...arr].sort((a, b) => a - b);
  const index = (p / 100) * (sorted.length - 1);
  const lower = Math.floor(index);
  const upper = Math.ceil(index);
  const weight = index - lower;
  return sorted[lower] * (1 - weight) + sorted[upper] * weight;
}

/**
 * Calculate skewness
 */
export function skewness(arr: number[]): number {
  const m = mean(arr);
  const s = std(arr, 1);
  const n = arr.length;
  
  const skew = arr.reduce((sum, val) => sum + Math.pow((val - m) / s, 3), 0);
  return (n / ((n - 1) * (n - 2))) * skew;
}

/**
 * Calculate kurtosis (excess kurtosis, Fisher's definition)
 */
export function kurtosis(arr: number[]): number {
  const m = mean(arr);
  const s = std(arr, 1);
  const n = arr.length;
  
  const kurt = arr.reduce((sum, val) => sum + Math.pow((val - m) / s, 4), 0);
  const adjustment = (n * (n + 1)) / ((n - 1) * (n - 2) * (n - 3));
  const bias = (3 * (n - 1) * (n - 1)) / ((n - 2) * (n - 3));
  
  return adjustment * kurt - bias;
}

/**
 * Normal distribution CDF
 */
function normalCDF(x: number, mean: number = 0, stdDev: number = 1): number {
  const z = (x - mean) / stdDev;
  const t = 1 / (1 + 0.2316419 * Math.abs(z));
  const d = 0.3989423 * Math.exp(-z * z / 2);
  const p = d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))));
  return z > 0 ? 1 - p : p;
}

/**
 * Generate random normal samples (Box-Muller transform)
 */
export function randomNormal(mean: number, stdDev: number, size: number): number[] {
  const samples: number[] = [];
  
  for (let i = 0; i < size; i += 2) {
    const u1 = Math.random();
    const u2 = Math.random();
    
    const z0 = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    const z1 = Math.sqrt(-2 * Math.log(u1)) * Math.sin(2 * Math.PI * u2);
    
    samples.push(mean + stdDev * z0);
    if (samples.length < size) {
      samples.push(mean + stdDev * z1);
    }
  }
  
  return samples.slice(0, size);
}

/**
 * Draw skewed-t log returns
 * Matches the Python function draw_skewt_log_returns
 */
export function drawSkewTLogReturns(
  rows: number,
  cols: number,
  nu: number,
  lambda: number,
  loc: number,
  scale: number
): number[][] {
  const n = rows * cols;
  const dist = new SkewedStudentT();
  
  // Generate uniform random percentiles
  const p: number[] = [];
  for (let i = 0; i < n; i++) {
    p.push(Math.random());
  }
  
  // Get quantiles from skewed-t distribution
  const logr_unscaled = dist.ppf(p, [nu, lambda]) as number[];
  
  // Standardize to mean 0, std 1
  const sampleMean = mean(logr_unscaled);
  let sampleStd = std(logr_unscaled, 1);
  
  if (!isFinite(sampleStd) || sampleStd < EPS) {
    sampleStd = 1.0;
  }
  
  const standardized = logr_unscaled.map(val => (val - sampleMean) / sampleStd);
  
  // Apply target loc and scale
  const logr = standardized.map(val => loc + scale * val);
  
  // Reshape to 2D array
  const result: number[][] = [];
  for (let i = 0; i < rows; i++) {
    result.push(logr.slice(i * cols, (i + 1) * cols));
  }
  
  return result;
}

/**
 * Draw stock simple returns from log returns
 */
export function drawStockSimpleReturns(
  rows: number,
  cols: number,
  nu: number,
  lambda: number,
  loc: number,
  scale: number
): number[][] {
  const logr = drawSkewTLogReturns(rows, cols, nu, lambda, loc, scale);
  
  // Convert log returns to simple returns: exp(r) - 1
  return logr.map(row => 
    row.map(val => Math.max(Math.exp(val) - 1, MIN_SIMPLE_RETURN))
  );
}

/**
 * Apply withdrawal to portfolio
 */
export function applyWithdrawal(
  portfolioValues: number[],
  amount: number,
  stockValues?: number[],
  cashValues?: number[]
): { portfolioValues?: number[], stockValues?: number[], cashValues?: number[] } {
  if (!stockValues || !cashValues) {
    const newPortfolio = portfolioValues.map(val => Math.max(val - amount, 0));
    return { portfolioValues: newPortfolio };
  } else {
    const total = stockValues.map((s, i) => s + cashValues[i]);
    const wStock = total.map((t, i) => t > 0 ? stockValues[i] / t : 0);
    
    const newStock = stockValues.map((s, i) => Math.max(s - amount * wStock[i], 0));
    const newCash = cashValues.map((c, i) => Math.max(c - amount * (1 - wStock[i]), 0));
    
    return { stockValues: newStock, cashValues: newCash };
  }
}

export interface SimulationParams {
  startValue: number;
  realSpending: number;
  stockProp: number;
  cashProp: number;
  cashReturn: number;
  cashVol: number;
  inflationRate: number;
  inflationVol: number;
  stockLogLoc: number;
  stockLogScale: number;
  skewTNu: number;
  skewTLambda: number;
  simulationYears: number;
  withdrawalTiming: 'Start of year' | 'Mid-year';
  rebalanceEachYear: boolean;
}

export interface SimulationResult {
  finalReal: number[];
  stockReturns: number[][];
  portfolioReturns: number[][];
}

/**
 * Run Monte Carlo simulation
 * This is the main simulation function ported from Python
 */
export function runMonteCarloSimulation(params: SimulationParams): SimulationResult {
  const {
    startValue,
    realSpending,
    stockProp,
    cashProp,
    cashReturn,
    cashVol,
    inflationRate,
    inflationVol,
    stockLogLoc,
    stockLogScale,
    skewTNu,
    skewTLambda,
    simulationYears,
    withdrawalTiming,
    rebalanceEachYear
  } = params;

  // Generate random matrices for all runs and years
  const inflationMatrix: number[][] = [];
  const cashReturnsMatrix: number[][] = [];
  
  for (let i = 0; i < NUM_RUNS; i++) {
    inflationMatrix.push(randomNormal(inflationRate, inflationVol, simulationYears));
    cashReturnsMatrix.push(randomNormal(cashReturn, cashVol, simulationYears));
  }
  
  const stockReturnsMatrix = drawStockSimpleReturns(
    NUM_RUNS,
    simulationYears,
    skewTNu,
    skewTLambda,
    stockLogLoc,
    stockLogScale
  );

  let cumulativeInflation = new Array(NUM_RUNS).fill(1.0);
  const portfolioReturnsMatrix: number[][] = Array(NUM_RUNS).fill(0).map(() => Array(simulationYears).fill(0));

  let portfolioValues: number[] | undefined;
  let stockValues: number[] | undefined;
  let cashValues: number[] | undefined;

  if (rebalanceEachYear) {
    portfolioValues = new Array(NUM_RUNS).fill(startValue);
  } else {
    stockValues = new Array(NUM_RUNS).fill(startValue * stockProp);
    cashValues = new Array(NUM_RUNS).fill(startValue * cashProp);
  }

  for (let year = 0; year < simulationYears; year++) {
    // Inflation factor for the year
    const inflFactor = inflationMatrix.map(row => Math.max(1.0 + row[year], MIN_INFL_FACTOR));
    
    // Real spending
    const spendSoy = cumulativeInflation.map(ci => realSpending * ci);
    const spendMid = spendSoy.map((s, i) => s * Math.sqrt(Math.max(inflFactor[i], SQRT_FLOOR)));
    
    // Asset returns
    const rStock = stockReturnsMatrix.map(row => row[year]);
    const rCash = cashReturnsMatrix.map(row => Math.max(row[year], MIN_R_FOR_SQRT));

    if (rebalanceEachYear && portfolioValues) {
      if (withdrawalTiming === 'Start of year') {
        const rPort = rStock.map((rs, i) => stockProp * rs + (1 - stockProp) * rCash[i]);
        portfolioReturnsMatrix.forEach((row, i) => row[year] = rPort[i]);
        
        const withdrawn = applyWithdrawal(portfolioValues, 0, undefined, undefined);
        portfolioValues = withdrawn.portfolioValues!.map((v, i) => v - spendSoy[i]);
        portfolioValues = portfolioValues.map((v, i) => Math.max(v * Math.max(1.0 + rPort[i], 0), 0));
      } else {
        // Mid-year
        let tempStockValues = portfolioValues.map(v => v * stockProp);
        let tempCashValues = portfolioValues.map(v => v * (1 - stockProp));
        
        const stockHalf = rStock.map(r => Math.sqrt(Math.max(1.0 + r, SQRT_FLOOR)));
        const cashHalf = rCash.map(r => Math.sqrt(Math.max(1.0 + r, SQRT_FLOOR)));
        
        tempStockValues = tempStockValues.map((v, i) => v * stockHalf[i]);
        tempCashValues = tempCashValues.map((v, i) => v * cashHalf[i]);
        
        const withdrawn = applyWithdrawal([], 0, tempStockValues, tempCashValues);
        tempStockValues = withdrawn.stockValues!.map((v, i) => v - spendMid[i] * (v / (v + withdrawn.cashValues![i] || 1)));
        tempCashValues = withdrawn.cashValues!.map((v, i) => v - spendMid[i] * (v / (v + withdrawn.stockValues![i] || 1)));
        
        // Proper withdrawal
        const total = tempStockValues.map((s, i) => s + tempCashValues[i]);
        const wStock = total.map((t, i) => t > 0 ? tempStockValues[i] / t : 0);
        tempStockValues = tempStockValues.map((s, i) => Math.max(s - spendMid[i] * wStock[i], 0));
        tempCashValues = tempCashValues.map((c, i) => Math.max(c - spendMid[i] * (1 - wStock[i]), 0));
        
        tempStockValues = tempStockValues.map((v, i) => v * stockHalf[i]);
        tempCashValues = tempCashValues.map((v, i) => v * cashHalf[i]);
        
        portfolioValues = tempStockValues.map((s, i) => Math.max(s + tempCashValues[i], 0));
        
        const rPort = rStock.map((rs, i) => stockProp * rs + (1 - stockProp) * rCash[i]);
        portfolioReturnsMatrix.forEach((row, i) => row[year] = rPort[i]);
      }
    } else if (stockValues && cashValues) {
      // No rebalancing
      const currentTotal = stockValues.map((s, i) => s + cashValues[i]);
      const wStockCurrent = currentTotal.map((t, i) => t > 0 ? stockValues[i] / t : 0);
      
      const rPort = wStockCurrent.map((w, i) => w * rStock[i] + (1 - w) * rCash[i]);
      portfolioReturnsMatrix.forEach((row, i) => row[year] = rPort[i]);
      
      if (withdrawalTiming === 'Start of year') {
        const total = stockValues.map((s, i) => s + cashValues[i]);
        const wStock = total.map((t, i) => t > 0 ? stockValues[i] / t : 0);
        
        stockValues = stockValues.map((s, i) => Math.max((s - spendSoy[i] * wStock[i]) * Math.max(1.0 + rStock[i], 0), 0));
        cashValues = cashValues.map((c, i) => Math.max((c - spendSoy[i] * (1 - wStock[i])) * Math.max(1.0 + rCash[i], 0), 0));
      } else {
        const stockHalf = rStock.map(r => Math.sqrt(Math.max(1.0 + r, SQRT_FLOOR)));
        const cashHalf = rCash.map(r => Math.sqrt(Math.max(1.0 + r, SQRT_FLOOR)));
        
        stockValues = stockValues.map((v, i) => v * stockHalf[i]);
        cashValues = cashValues.map((v, i) => v * cashHalf[i]);
        
        const total = stockValues.map((s, i) => s + cashValues[i]);
        const wStock = total.map((t, i) => t > 0 ? stockValues[i] / t : 0);
        
        stockValues = stockValues.map((s, i) => Math.max((s - spendMid[i] * wStock[i]) * stockHalf[i], 0));
        cashValues = cashValues.map((c, i) => Math.max((c - spendMid[i] * (1 - wStock[i])) * cashHalf[i], 0));
      }
    }
    
    // Update cumulative inflation
    cumulativeInflation = cumulativeInflation.map((ci, i) => ci * inflFactor[i]);
  }

  const finalNominal = rebalanceEachYear 
    ? portfolioValues! 
    : stockValues!.map((s, i) => s + cashValues![i]);
  
  const finalReal = finalNominal.map((nom, i) => nom / Math.max(cumulativeInflation[i], EPS));

  return {
    finalReal,
    stockReturns: stockReturnsMatrix,
    portfolioReturns: portfolioReturnsMatrix
  };
}

/**
 * Calculate negative return statistics
 */
export function calculateNegativeReturnPercentages(
  stockReturns: number[][],
  portfolioReturns: number[][],
  stockLogMu: number,
  stockLogSigma: number
) {
  const stockFlat = stockReturns.flat();
  const portfolioFlat = portfolioReturns.flat();
  
  const thresholds = [-0.10, -0.15, -0.20, -0.25, -0.30, -0.35, -0.40, -0.45, -0.50];
  
  // Use S&P parameters for Normal comparison
  const normalMu = Math.log(1.0 + SNP7424_GEOM_MEAN);
  const normalSigma = SNP7424_LOG_STD;
  
  const rows = thresholds.map(threshold => {
    const stockPct = (stockFlat.filter(r => r < threshold).length / stockFlat.length) * 100;
    const portfolioPct = (portfolioFlat.filter(r => r < threshold).length / portfolioFlat.length) * 100;
    
    // Normal comparison
    const onePlusK = 1.0 + threshold;
    let normPct: number;
    
    if (normalSigma > 0 && onePlusK > 0) {
      const z = (Math.log(onePlusK) - normalMu) / normalSigma;
      normPct = normalCDF(z, 0, 1) * 100;
    } else {
      normPct = onePlusK <= 0 ? 0 : (Math.exp(normalMu) - 1 < threshold ? 100 : 0);
    }
    
    return {
      threshold: `Under ${Math.abs(threshold * 100)}%`,
      stock: `${stockPct.toFixed(1)}%`,
      portfolio: `${portfolioPct.toFixed(1)}%`,
      normal: `${normPct.toFixed(1)}%`
    };
  });
  
  return rows;
}