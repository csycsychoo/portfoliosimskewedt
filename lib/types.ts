/**
 * Type definitions for the portfolio simulator
 */

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

export interface NegativeReturnRow {
  threshold: string;
  stock: string;
  portfolio: string;
  normal: string;
}

export interface PercentileData {
  percentile: number;
  value: number;
}

export interface StatisticsData {
  median: number;
  mean: number;
  std: number;
  successRate: number;
  outperformRate: number;
}