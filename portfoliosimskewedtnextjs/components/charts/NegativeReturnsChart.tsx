"use client";

import { PlotlyChart } from "./PlotlyChart";
import type { Data } from "plotly.js";
import type { SimulationResponse } from "@/lib/types";
import { SP_HISTORICAL_NEGATIVE_RETURNS } from "@/lib/presets";

export function NegativeReturnsChart({ result }: { result: SimulationResponse }) {
  const rows = result.negativeReturnsTable;
  const xLabels = rows.map((r) => `Under ${Math.round(r.threshold * 100)}%`);
  const series: { name: string; values: number[]; color: string }[] = [
    {
      name: "Stock in your portfolio",
      values: rows.map((r) => r.stockPct),
      color: "#3b82f6",
    },
    {
      name: "Your overall portfolio",
      values: rows.map((r) => r.portfolioPct),
      color: "#10b981",
    },
    {
      name: "Normal w/ S&P '74-'24 mean & vol",
      values: rows.map((r) => r.normalPct),
      color: "#f59e0b",
    },
    {
      name: "S&P 500 1974-2024 historical",
      values: SP_HISTORICAL_NEGATIVE_RETURNS,
      color: "#ef4444",
    },
  ];

  const traces: Data[] = series.map((s) => ({
    type: "scatter",
    mode: "lines+markers",
    name: s.name,
    x: xLabels,
    y: s.values,
    line: { color: s.color },
  }));

  return (
    <PlotlyChart
      data={traces}
      layout={{
        xaxis: { categoryorder: "array", categoryarray: xLabels, tickangle: -30, fixedrange: true },
        yaxis: { title: { text: "% of years" }, ticksuffix: "%", fixedrange: true },
        margin: { l: 60, r: 10, t: 20, b: 80 },
        legend: {
          orientation: "h",
          yanchor: "bottom",
          y: 1.02,
          xanchor: "center",
          x: 0.5,
        },
      }}
    />
  );
}
