"use client";

import { PlotlyChart } from "./PlotlyChart";
import type { Data } from "plotly.js";
import type { SimulationResponse } from "@/lib/types";

const COLORS: Record<string, string> = {
  p10: "#ef4444",
  p25: "#f59e0b",
  p50: "#10b981",
  p75: "#3b82f6",
  p90: "#8b5cf6",
};

export function TrajectoryChart({ result }: { result: SimulationResponse }) {
  const years = Array.from({ length: result.years + 1 }, (_, i) => i);
  const traces: Data[] = (
    Object.entries(result.trajectories) as [keyof typeof COLORS, number[]][]
  ).map(([key, values]) => ({
    type: "scatter",
    mode: "lines",
    name: `${key.slice(1)}th`,
    x: years,
    y: values,
    line: { color: COLORS[key] },
  }));

  return (
    <PlotlyChart
      data={traces}
      layout={{
        title: { text: "Percentile Trajectories by Final Outcome (Real Terms)" },
        xaxis: { title: { text: "Year" }, fixedrange: true },
        yaxis: { title: { text: "Portfolio Value ($)" }, rangemode: "tozero", fixedrange: true },
        hovermode: "x unified",
        margin: { l: 70, r: 20, t: 50, b: 50 },
      }}
    />
  );
}
