export interface Preset {
  name: string;
  emoji: string;
  description: string;
  stockGeomMeanPercent: number;
  inflationRatePercent: number;
  cashReturnPercent: number;
  stockLogVolPercent: number;
  rebalanceEachYear: boolean;
}

export const PRESETS: Preset[] = [
  {
    name: "USA Today",
    emoji: "🇺🇸",
    description: "Stock geo return 8.67%, Inflation 2.9%, Cash return 4.8%",
    stockGeomMeanPercent: 8.67,
    inflationRatePercent: 2.9,
    cashReturnPercent: 4.8,
    stockLogVolPercent: 17.0,
    rebalanceEachYear: true,
  },
  {
    name: "Trinity Study",
    emoji: "📊",
    description: "Stock geo return 10.6%, Inflation 2.96%, Cash return 5.7%",
    stockGeomMeanPercent: 10.6,
    inflationRatePercent: 2.96,
    cashReturnPercent: 5.7,
    stockLogVolPercent: 17.0,
    rebalanceEachYear: true,
  },
  {
    name: "Singapore",
    emoji: "🇸🇬",
    description: "Stock geo return 7%, Inflation 1.7%, Cash return 2.5%",
    stockGeomMeanPercent: 7.0,
    inflationRatePercent: 1.7,
    cashReturnPercent: 2.5,
    stockLogVolPercent: 20.0,
    rebalanceEachYear: true,
  },
];

export const DEFAULT_PRESET_NAME = "Singapore";

export const SP_HISTORICAL_NEGATIVE_RETURNS: number[] = [
  12.5, 10.0, 7.5, 5.0, 2.5, 2.5, 0.0, 0.0, 0.0,
];
