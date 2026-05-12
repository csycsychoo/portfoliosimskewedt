export type WithdrawalTiming = "Start of year" | "Mid-year";

export interface SimulationRequest {
  startValue: number;
  realSpending: number;
  stockPropPercent: number;
  inflationRatePercent: number;
  inflationVolPercent: number;
  cashReturnPercent: number;
  cashVolPercent: number;
  stockGeomMeanPercent: number;
  stockLogVolPercent: number;
  skewtNu: number;
  skewtLambda: number;
  simulationYears: number;
  withdrawalTiming: WithdrawalTiming;
  rebalanceEachYear: boolean;
  numRuns?: number;
  seed?: number | null;
}

export interface NegativeReturnRow {
  threshold: number;
  stockPct: number;
  portfolioPct: number;
  normalPct: number;
}

export interface SimulationResponse {
  medianFinal: number;
  meanFinal: number;
  stdFinal: number;
  successRate: number;
  outperformRate: number;
  endingPercentiles: Record<string, number>;
  trajectories: Record<"p10" | "p25" | "p50" | "p75" | "p90", number[]>;
  stockStats: {
    median: number;
    mean: number;
    std: number;
    skew: number;
    kurtosis: number;
  };
  stockHistogram: {
    binEdges: number[];
    counts: number[];
  };
  negativeReturnsTable: NegativeReturnRow[];
  years: number;
}

export interface SimulationError {
  error: string;
  message: string;
}
