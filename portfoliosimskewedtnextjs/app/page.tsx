"use client";

import { useMemo, useState } from "react";
import type { SimulationRequest, SimulationResponse, WithdrawalTiming } from "@/lib/types";
import {
  PRESETS,
  DEFAULT_PRESET_NAME,
  type Preset,
} from "@/lib/presets";
import {
  formatCurrencyInput,
  parseCurrencyInput,
  usd,
  pct,
  pct2,
  num,
} from "@/lib/format";
import { TrajectoryChart } from "@/components/charts/TrajectoryChart";
import { StockHistogram } from "@/components/charts/StockHistogram";
import { NegativeReturnsChart } from "@/components/charts/NegativeReturnsChart";

const APP_VERSION = "v2.0.0-next";

const DEFAULT_FORM: SimulationRequest = {
  startValue: 5_000_000,
  realSpending: 150_000,
  stockPropPercent: 70,
  inflationRatePercent: 1.7,
  inflationVolPercent: 2.0,
  cashReturnPercent: 2.5,
  cashVolPercent: 1.0,
  stockGeomMeanPercent: 7.0,
  stockLogVolPercent: 20.0,
  skewtNu: 5.0,
  skewtLambda: -0.3,
  simulationYears: 50,
  withdrawalTiming: "Mid-year",
  rebalanceEachYear: true,
};

function applyPreset(form: SimulationRequest, p: Preset): SimulationRequest {
  return {
    ...form,
    stockGeomMeanPercent: p.stockGeomMeanPercent,
    inflationRatePercent: p.inflationRatePercent,
    cashReturnPercent: p.cashReturnPercent,
    stockLogVolPercent: p.stockLogVolPercent,
    rebalanceEachYear: p.rebalanceEachYear,
  };
}

function presetMatches(form: SimulationRequest, p: Preset): boolean {
  return (
    form.stockGeomMeanPercent === p.stockGeomMeanPercent &&
    form.inflationRatePercent === p.inflationRatePercent &&
    form.cashReturnPercent === p.cashReturnPercent &&
    form.stockLogVolPercent === p.stockLogVolPercent
  );
}

export default function Home() {
  const initialPreset = PRESETS.find((p) => p.name === DEFAULT_PRESET_NAME)!;
  const [form, setForm] = useState<SimulationRequest>(
    applyPreset(DEFAULT_FORM, initialPreset),
  );
  const [currencyText, setCurrencyText] = useState({
    startValue: formatCurrencyInput(DEFAULT_FORM.startValue),
    realSpending: formatCurrencyInput(DEFAULT_FORM.realSpending),
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<SimulationResponse | null>(null);

  const currentPreset = useMemo(
    () => PRESETS.find((p) => presetMatches(form, p))?.name ?? "Custom",
    [form],
  );

  const erpGeo = form.stockGeomMeanPercent - form.cashReturnPercent;

  const setField = <K extends keyof SimulationRequest>(
    key: K,
    value: SimulationRequest[K],
  ) => setForm((f) => ({ ...f, [key]: value }));

  const handleRun = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch("/api/simulate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(form),
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.message || `HTTP ${res.status}`);
      }
      const data: SimulationResponse = await res.json();
      setResult(data);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="container">
      <h1>Retirement Portfolio Simulator with extreme events (fat tails, negative skew)</h1>
      <div className="layout">
        <aside>
          <div className="panel">
            <div className="panel-title">🎯 Easy defaults</div>
            <div style={{ fontSize: "0.85rem", color: "#6b7280", marginBottom: "0.5rem" }}>
              Quick preset buttons for common scenarios:
            </div>
            {currentPreset !== "Custom" && (
              <div className="preset-status">
                ✅ Currently using: {currentPreset}
                {currentPreset === "Trinity Study" && " (with rebalancing enabled)"}
              </div>
            )}
            <div className="presets">
              {PRESETS.map((p) => (
                <button
                  key={p.name}
                  className={`preset-btn ${currentPreset === p.name ? "active" : ""}`}
                  title={p.description}
                  onClick={() => setForm((f) => applyPreset(f, p))}
                >
                  {p.emoji} {p.name}
                </button>
              ))}
            </div>
          </div>

          <details className="panel" open>
            <summary>Initial Setup</summary>
            <div className="field">
              <label>Starting Portfolio ($)</label>
              <input
                type="text"
                value={currencyText.startValue}
                onChange={(e) => {
                  const v = parseCurrencyInput(e.target.value);
                  setCurrencyText((c) => ({ ...c, startValue: formatCurrencyInput(v) }));
                  setField("startValue", v);
                }}
              />
            </div>
            <div className="field">
              <label>Annual Withdrawal (grows with inflation) ($)</label>
              <input
                type="text"
                value={currencyText.realSpending}
                onChange={(e) => {
                  const v = parseCurrencyInput(e.target.value);
                  setCurrencyText((c) => ({ ...c, realSpending: formatCurrencyInput(v) }));
                  setField("realSpending", v);
                }}
              />
            </div>
            <div className="field">
              <label>Simulation Years</label>
              <input
                type="number"
                min={1}
                value={form.simulationYears}
                onChange={(e) => setField("simulationYears", parseInt(e.target.value || "1", 10))}
              />
            </div>
            <SliderField
              label="Stock % in Portfolio"
              value={form.stockPropPercent}
              min={0}
              max={100}
              step={5}
              format={(v) => `${v}%`}
              onChange={(v) => setField("stockPropPercent", v)}
            />
          </details>

          <details className="panel" open>
            <summary>Return and economic assumptions</summary>
            <SliderField
              label="Stock Average Return % (Geometric mean)"
              value={form.stockGeomMeanPercent}
              min={-20}
              max={30}
              step={0.1}
              format={(v) => `${v.toFixed(1)}%`}
              onChange={(v) => setField("stockGeomMeanPercent", v)}
            />
            <SliderField
              label="Cash/Bond Return (%)"
              value={form.cashReturnPercent}
              min={0}
              max={10}
              step={0.1}
              format={(v) => `${v.toFixed(1)}%`}
              onChange={(v) => setField("cashReturnPercent", v)}
            />
            <div style={{ fontSize: "0.85rem", color: "#4b5563", marginBottom: "0.5rem" }}>
              Implied Equity Risk Premium (stock − cash): {erpGeo.toFixed(2)}%
            </div>
            <SliderField
              label="Inflation Mean (%)"
              value={form.inflationRatePercent}
              min={0}
              max={10}
              step={0.1}
              format={(v) => `${v.toFixed(1)}%`}
              onChange={(v) => setField("inflationRatePercent", v)}
            />
          </details>

          <details className="panel" open>
            <summary>Policy Options</summary>
            <div className="field">
              <label>Withdrawal Timing</label>
              <div>
                {(["Start of year", "Mid-year"] as WithdrawalTiming[]).map((opt) => (
                  <label key={opt} style={{ marginRight: "1rem", fontWeight: "normal" }}>
                    <input
                      type="radio"
                      name="withdrawalTiming"
                      checked={form.withdrawalTiming === opt}
                      onChange={() => setField("withdrawalTiming", opt)}
                    />{" "}
                    {opt}
                  </label>
                ))}
              </div>
            </div>
            <div className="field checkbox-row">
              <input
                id="rebalance"
                type="checkbox"
                checked={form.rebalanceEachYear}
                onChange={(e) => setField("rebalanceEachYear", e.target.checked)}
              />
              <label htmlFor="rebalance" style={{ margin: 0 }}>
                Rebalance annually
              </label>
            </div>
            {form.rebalanceEachYear ? (
              <div className="info">
                🔄 Rebalancing enabled — portfolio resets to target allocation each year.
              </div>
            ) : (
              <div className="warning">
                ⚠️ Rebalancing disabled — allocation will drift over time.
              </div>
            )}
          </details>

          <details className="panel">
            <summary>Advanced Options</summary>
            <SliderField
              label="Inflation Vol (%)"
              value={form.inflationVolPercent}
              min={0}
              max={5}
              step={0.1}
              format={(v) => `${v.toFixed(1)}%`}
              onChange={(v) => setField("inflationVolPercent", v)}
            />
            <SliderField
              label="Cash/Bond Vol (%)"
              value={form.cashVolPercent}
              min={0}
              max={5}
              step={0.1}
              format={(v) => `${v.toFixed(1)}%`}
              onChange={(v) => setField("cashVolPercent", v)}
            />
            <SliderField
              label="Stock Log Vol (%)"
              value={form.stockLogVolPercent}
              min={5}
              max={50}
              step={0.5}
              format={(v) => `${v.toFixed(1)}%`}
              onChange={(v) => setField("stockLogVolPercent", v)}
            />
            <SliderField
              label="Fat Tails (Nu) — lower = fatter"
              value={form.skewtNu}
              min={3}
              max={20}
              step={0.5}
              format={(v) => v.toFixed(1)}
              onChange={(v) => setField("skewtNu", v)}
            />
            <SliderField
              label="Skewness (Lambda) — negative = left skew"
              value={form.skewtLambda}
              min={-0.9}
              max={0.9}
              step={0.05}
              format={(v) => v.toFixed(2)}
              onChange={(v) => setField("skewtLambda", v)}
            />
            <div className="field">
              <label>Seed (leave blank for random)</label>
              <input
                type="number"
                value={form.seed ?? ""}
                onChange={(e) =>
                  setField(
                    "seed",
                    e.target.value === "" ? null : parseInt(e.target.value, 10),
                  )
                }
              />
            </div>
          </details>

          <button className="run-btn" onClick={handleRun} disabled={loading}>
            {loading ? "Simulating…" : "🚀 Run Simulation"}
          </button>
        </aside>

        <section>
          {error && <div className="error">Error: {error}</div>}
          {!result && !error && (
            <div className="empty">Set your assumptions on the left and click &apos;Run Simulation&apos;.</div>
          )}
          {result && (
            <>
              <div className="metrics">
                <div className="metric">
                  <div className="metric-label">{"Median Value at end\n(today's money)"}</div>
                  <div className="metric-value">{usd(result.medianFinal)}</div>
                </div>
                <div className="metric">
                  <div className="metric-label">
                    {"Likelihood do not run out of money\n(% runs end value > 0)"}
                  </div>
                  <div className="metric-value">{pct(result.successRate)}</div>
                </div>
                <div className="metric">
                  <div className="metric-label">
                    Likelihood end with more money than started (real terms)
                  </div>
                  <div className="metric-value">{pct(result.outperformRate)}</div>
                </div>
              </div>

              <TrajectoryChart result={result} />

              <h2>Ending portfolio values in today&apos;s money</h2>
              <table className="summary">
                <thead>
                  <tr>
                    <th></th>
                    <th>Outcome percentiles</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(result.endingPercentiles).map(([k, v]) => (
                    <tr key={k}>
                      <td>{k.slice(1)}%</td>
                      <td>{usd(v)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>

              <h2>% of years with extreme negative returns vs S&amp;P historical and normal distribution</h2>
              <NegativeReturnsChart result={result} />

              <StockHistogram result={result} />
              <div className="stat-row">
                <span>Median of simple returns: {pct2(result.stockStats.median)}</span>
                <span>Arithmetic mean: {pct2(result.stockStats.mean)}</span>
                <span>Std Dev: {pct2(result.stockStats.std)}</span>
                <span>Skew: {num(result.stockStats.skew)}</span>
                <span>Kurtosis (fat tails): {num(result.stockStats.kurtosis)}</span>
              </div>
            </>
          )}
          <div className="version">App version: {APP_VERSION}</div>
        </section>
      </div>
    </main>
  );
}

function SliderField({
  label,
  value,
  min,
  max,
  step,
  format,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  format: (v: number) => string;
  onChange: (v: number) => void;
}) {
  return (
    <div className="field">
      <label>{label}</label>
      <div className="range-row">
        <input
          type="range"
          min={min}
          max={max}
          step={step}
          value={value}
          onChange={(e) => onChange(parseFloat(e.target.value))}
        />
        <span>{format(value)}</span>
      </div>
    </div>
  );
}
