"""
Vercel Python serverless entry point.

Deployed at: POST /api/simulate
Request body (JSON):
{
  "startValue": 5000000,
  "realSpending": 150000,
  "stockPropPercent": 70,
  "inflationRatePercent": 1.7,
  "inflationVolPercent": 2.0,
  "cashReturnPercent": 2.5,
  "cashVolPercent": 1.0,
  "stockGeomMeanPercent": 7.0,
  "stockLogVolPercent": 20.0,
  "skewtNu": 5.0,
  "skewtLambda": -0.3,
  "simulationYears": 50,
  "withdrawalTiming": "Mid-year",     // or "Start of year"
  "rebalanceEachYear": true,
  "numRuns": 10000,                   // optional
  "seed": null                        // optional int for reproducibility
}
"""

from __future__ import annotations

import json
import sys
from http.server import BaseHTTPRequestHandler
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "py"))

from engine import simulate  # noqa: E402


def _cors_headers(handler: BaseHTTPRequestHandler) -> None:
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
    handler.send_header("Access-Control-Allow-Headers", "Content-Type")


class handler(BaseHTTPRequestHandler):  # noqa: N801 — Vercel convention
    def do_OPTIONS(self):  # noqa: N802
        self.send_response(204)
        _cors_headers(self)
        self.end_headers()

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", "0") or "0")
        raw = self.rfile.read(length) if length else b"{}"
        try:
            params = json.loads(raw.decode("utf-8") or "{}")
            result = simulate(params)
            body = json.dumps(result).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            _cors_headers(self)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except Exception as exc:  # surface a clean JSON error
            err = json.dumps({"error": type(exc).__name__, "message": str(exc)}).encode("utf-8")
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            _cors_headers(self)
            self.send_header("Content-Length", str(len(err)))
            self.end_headers()
            self.wfile.write(err)
