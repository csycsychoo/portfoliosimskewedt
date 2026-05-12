"use client";

import dynamic from "next/dynamic";
import type { Layout, Data, Config } from "plotly.js";

const Plot = dynamic(() => import("react-plotly.js"), { ssr: false });

const DEFAULT_CONFIG: Partial<Config> = {
  displaylogo: false,
  scrollZoom: false,
  modeBarButtonsToRemove: [
    "zoom2d",
    "pan2d",
    "select2d",
    "lasso2d",
    "zoomIn2d",
    "zoomOut2d",
    "autoScale2d",
    "resetScale2d",
  ],
  responsive: true,
};

export function PlotlyChart({
  data,
  layout,
  config,
  style,
}: {
  data: Data[];
  layout: Partial<Layout>;
  config?: Partial<Config>;
  style?: React.CSSProperties;
}) {
  return (
    <Plot
      data={data}
      layout={{ autosize: true, ...layout }}
      config={{ ...DEFAULT_CONFIG, ...config }}
      style={{ width: "100%", height: 380, ...style }}
      useResizeHandler
    />
  );
}
