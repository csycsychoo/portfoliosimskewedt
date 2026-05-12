"use client";

import { PlotlyChart } from "./PlotlyChart";
import type { Data } from "plotly.js";
import type { SimulationResponse } from "@/lib/types";

export function StockHistogram({ result }: { result: SimulationResponse }) {
  const { binEdges, counts } = result.stockHistogram;
  // Bin centers for bar chart
  const centers: number[] = [];
  for (let i = 0; i < counts.length; i++) {
    centers.push((binEdges[i] + binEdges[i + 1]) / 2);
  }
  const total = counts.reduce((a, b) => a + b, 0) || 1;
  const percents = counts.map((c) => (c / total) * 100);

  const traces: Data[] = [
    {
      type: "bar",
      x: centers,
      y: percents,
      marker: { color: "#3b82f6" },
      hovertemplate: "%{x:.2%}: %{y:.2f}%<extra></extra>",
    },
  ];

  return (
    <PlotlyChart
      data={traces}
      layout={{
        title: { text: "Annual Stock Returns" },
        xaxis: { title: { text: "Annual Return" }, tickformat: ".2%", fixedrange: true },
        yaxis: { title: { text: "%" }, fixedrange: true },
        bargap: 0.05,
        margin: { l: 50, r: 20, t: 50, b: 50 },
      }}
    />
  );
}
