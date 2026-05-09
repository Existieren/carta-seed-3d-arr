"""
Carta Seed Stage Sector Map -> 3D with estimated ARR (Z axis).

Source chart: Carta "Seed Stage Sector Map", 1,346 US seed rounds, Apr 2025 - Mar 2026.
X = median round size ($M), Y = median post-money valuation ($M), bubble area = # of rounds.

Z (this script) = blended ARR-at-seed estimate from a 3-estimator model:
  A. Stage-conditional revenue-multiple inversion
  B. Burn-implied revenue (18-mo runway / sector NBM)
  C. Empirical Bayesian prior (median seed ARR per sector, public sources)
Blend = weighted geometric mean (revenues are log-normal). Sensitivity = +/-50% on inputs.

See methodology.md in the same folder for sources, formulas, and limitations.
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.interpolate import griddata

OUT_DIR = Path(__file__).resolve().parent
HTML_OUT = OUT_DIR / "index.html"

# ---------------------------------------------------------------------------
# 1. SECTOR DATA (centroids read from the Carta chart; per-sector table not
#    publicly published by Carta, see methodology.md for caveat)
# ---------------------------------------------------------------------------
SECTORS = [
    # (name, round_M, post_M, rounds_proxy, class, pricing_paradigm)
    ("AI Infra",         13.5, 65.0,  70, "ai_infra",         "mixed"),
    ("Analytics",         5.5, 36.0, 220, "ai_app",           "ARR"),
    ("AI Applications",   4.0, 30.0, 350, "ai_app",           "ARR"),
    ("Web3",              5.2, 28.0,  75, "web3",             "token-FDV"),
    ("Fintech",           4.0, 26.0, 130, "fintech",          "ARR"),
    ("Semiconductors",    7.5, 24.0,  60, "deeptech_silicon", "NPV/option"),
    ("Hardware",          5.3, 23.0,  70, "hardware",         "NPV/option"),
    ("Cybersecurity",     5.0, 21.0,  90, "saas",             "ARR"),
    ("Transport",         3.5, 22.0,  50, "hardware",         "NPV/option"),
    ("Media",             3.0, 20.0,  40, "consumer",         "ARR"),
    ("Renewables",        5.0, 19.0,  55, "deeptech_climate", "NPV/option"),
    ("Marketplace",       5.0, 18.0,  60, "marketplace",      "GMV-take-rate"),
    ("Healthtech",        4.5, 17.0,  65, "healthtech",       "ARR"),
    ("Proptech",          2.7, 18.0,  45, "saas",             "ARR"),
    ("Personal",          2.0, 16.0,  35, "consumer",         "ARR"),
    ("Med Devices",       2.5, 13.0,  30, "biotech",          "rNPV"),
    ("Biotech",           4.0, 13.0,  50, "biotech",          "rNPV"),
    ("HR",                3.5, 13.0,  45, "saas",             "ARR"),
    ("Logistics",         2.5, 12.0,  35, "saas",             "ARR"),
]

# ---------------------------------------------------------------------------
# 2. ESTIMATOR INPUTS PER SECTOR CLASS
#    All numbers triangulated from: Finro 2025 sector reports, Aventis multiples,
#    Bessemer Cloud 100 / State of AI 2025, High Alpha 2025 SaaS Benchmarks,
#    Carta Q1-Q3 2025 valuation caps, Sacra company breakdowns, named-deal
#    triangulation (Cursor, Mercor, Thinking Machines, Figure, Anvil).
# ---------------------------------------------------------------------------

# Estimator A: post_money / multiple at seed (low, central, high)
M_R = {
    "ai_infra":         (50,  150,  1000),
    "ai_app":           (25,   60,   200),
    "saas":             (15,   40,   100),
    "fintech":          (15,   35,    90),
    "marketplace":      ( 8,   25,    70),
    "hardware":         (50,  200,  2000),  # inf -> 2000 cap
    "deeptech_silicon": (75,  250,  2000),
    "deeptech_climate": (25,  100,  1000),
    "web3":             (30,  150,  2000),
    "healthtech":       (15,   40,   100),
    "biotech":          (100, 500,  5000),
    "consumer":         ( 5,   15,    50),
}

# Estimator B: net burn multiple at seed (low, central, high). None = not meaningful.
# Source-anchored to High Alpha 2025 SaaS Benchmarks, ICONIQ State of Software 2025,
# Bessemer State of AI 2025, ICanPitch burn benchmarks 2025, CFO Advisors 2025.
# AI App tailwind: Bessemer/ICONIQ/High Alpha consistently report sub-1.5x for AI-native.
NBM = {
    "ai_infra":         (1.5, 2.5, 4.0),
    "ai_app":           (0.8, 1.2, 2.0),
    "saas":             (1.5, 2.5, 3.5),
    "fintech":          (2.0, 3.5, 5.0),
    "marketplace":      (2.0, 3.0, 5.0),
    "hardware":         None,
    "deeptech_silicon": None,
    "deeptech_climate": None,
    "web3":             None,        # token-fundraising, ARR not the right metric
    "healthtech":       (2.0, 3.0, 4.5),
    "biotech":          None,
    "consumer":         (2.5, 4.0, 6.0),
}

# Estimator C: empirical median seed ARR ($M) (low, central, high)
# Source-anchored to Carta seed/pre-seed reports, Sacra (Cursor, Mercor, Glean),
# Metal.so 2025 SaaS seed benchmarks, Pitchwise median seed by industry 2026.
# AI Infra cohort median is dominated by frontier pre-revenue plays (Thinking
# Machines $0 / $12B); tooling outliers like Together AI sit at the high end.
ARR_PRIOR = {
    "ai_infra":         (0.0,  0.2,  1.0),
    "ai_app":           (0.3,  0.6,  1.5),
    "saas":             (0.3,  0.5,  1.0),
    "fintech":          (0.2,  0.4,  0.8),
    "marketplace":      (0.3,  0.6,  1.5),
    "hardware":         (0.0,  0.05, 0.3),
    "deeptech_silicon": (0.0,  0.0,  0.0),
    "deeptech_climate": (0.0,  0.1,  0.5),
    "web3":             (0.0,  0.05, 0.5),
    "healthtech":       (0.1,  0.3,  0.8),
    "biotech":          (0.0,  0.0,  0.0),
    "consumer":         (0.1,  0.3,  0.7),
}

# Blend weights (w_A, w_B, w_C). Multiples lose meaning in pre-revenue,
# so the prior dominates for option-priced / rNPV sectors.
WEIGHTS = {
    "ai_infra":         (0.35, 0.25, 0.40),
    "ai_app":           (0.40, 0.25, 0.35),
    "saas":             (0.40, 0.30, 0.30),
    "fintech":          (0.35, 0.30, 0.35),
    "marketplace":      (0.30, 0.30, 0.40),
    "hardware":         (0.05, 0.00, 0.95),
    "deeptech_silicon": (0.05, 0.00, 0.95),
    "deeptech_climate": (0.10, 0.00, 0.90),
    "web3":             (0.20, 0.00, 0.80),
    "healthtech":       (0.25, 0.25, 0.50),
    "biotech":          (0.02, 0.00, 0.98),
    "consumer":         (0.30, 0.25, 0.45),
}

# Visual class -> color
CLASS_COLOR = {
    "ai_infra":         "#000000",
    "ai_app":           "#9b59b6",
    "saas":             "#2980b9",
    "fintech":          "#3498db",
    "marketplace":      "#f39c12",
    "hardware":         "#a04020",
    "deeptech_silicon": "#7f8c8d",
    "deeptech_climate": "#27ae60",
    "web3":             "#e67e22",
    "healthtech":       "#e91e63",
    "biotech":          "#0a3d62",
    "consumer":         "#c0392b",
}

# Pre-revenue paradigms render with open-diamond markers
NON_ARR_PARADIGMS = {"NPV/option", "rNPV", "token-FDV", "mixed"}

EPS = 1e-3  # log floor


def estimate_arr(round_M, post_M, klass, perturb=None):
    """Return (ARR_A, ARR_B, ARR_C, ARR_blend) for one sector.

    perturb in {None, 'low', 'high'} -> apply +/-50% sensitivity.
    """
    mr_lo, mr_ce, mr_hi = M_R[klass]
    nbm = NBM[klass]
    pr_lo, pr_ce, pr_hi = ARR_PRIOR[klass]

    if perturb == "low":
        mr = mr_hi  # higher multiple -> lower implied ARR
        nbm_v = nbm[2] if nbm else None
        pr = pr_lo
    elif perturb == "high":
        mr = mr_lo
        nbm_v = nbm[0] if nbm else None
        pr = pr_hi
    else:
        mr = mr_ce
        nbm_v = nbm[1] if nbm else None
        pr = pr_ce

    arr_A = post_M / mr
    if nbm_v is not None:
        arr_B = (round_M * (12.0 / 18.0)) / nbm_v
    else:
        arr_B = 0.0
    arr_C = pr

    wA, wB, wC = WEIGHTS[klass]
    log_blend = (
        wA * np.log(max(arr_A, EPS))
        + wB * np.log(max(arr_B, EPS))
        + wC * np.log(max(arr_C, EPS))
    )
    arr_blend = float(np.exp(log_blend))
    return arr_A, arr_B, arr_C, arr_blend


def confidence_grade(arr_A, arr_B, arr_C, klass):
    """A: rev-bearing, 3 estimators within 2x. B: rev-bearing, divergent. C: prior-dominated."""
    if WEIGHTS[klass][2] >= 0.7:
        return "C"
    vals = [v for v in (arr_A, arr_B, arr_C) if v > 0]
    if not vals:
        return "C"
    spread = max(vals) / max(min(vals), EPS)
    return "A" if spread <= 2.0 else "B"


def build():
    rows = []
    for name, round_M, post_M, n, klass, paradigm in SECTORS:
        a_ce, b_ce, c_ce, blend_ce = estimate_arr(round_M, post_M, klass, None)
        _, _, _, blend_lo = estimate_arr(round_M, post_M, klass, "low")
        _, _, _, blend_hi = estimate_arr(round_M, post_M, klass, "high")
        rows.append(dict(
            sector=name, round_M=round_M, post_M=post_M, n=n,
            klass=klass, paradigm=paradigm,
            ARR_A=a_ce, ARR_B=b_ce, ARR_C=c_ce,
            ARR_blend=blend_ce, ARR_low=blend_lo, ARR_high=blend_hi,
            grade=confidence_grade(a_ce, b_ce, c_ce, klass),
        ))
    return pd.DataFrame(rows)


def make_figure(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()

    # --- Implied ARR surface (revenue-bearing sectors only) ---
    revbearing = df[~df["paradigm"].isin(NON_ARR_PARADIGMS)]
    if len(revbearing) >= 6:
        xg, yg = np.meshgrid(
            np.linspace(df["round_M"].min() * 0.7, df["round_M"].max() * 1.05, 60),
            np.linspace(df["post_M"].min() * 0.7,  df["post_M"].max()  * 1.05, 60),
        )
        zg = griddata(
            list(zip(revbearing["round_M"], revbearing["post_M"])),
            revbearing["ARR_blend"].values,
            (xg, yg),
            method="cubic",
        )
        fig.add_trace(go.Surface(
            x=xg, y=yg, z=zg,
            opacity=0.30, showscale=False,
            colorscale="Viridis",
            name="Implied ARR surface (rev-bearing fit)",
            hoverinfo="skip",
        ))

    # --- Sector bubbles ---
    # Split into ARR vs non-ARR paradigm so we can use different marker symbols.
    for label, subset, symbol in [
        ("Revenue-bearing (ARR/GMV)", df[~df["paradigm"].isin(NON_ARR_PARADIGMS)], "circle"),
        ("Pricing not ARR-anchored",  df[ df["paradigm"].isin(NON_ARR_PARADIGMS)], "diamond-open"),
    ]:
        if subset.empty:
            continue
        sizes = (np.sqrt(subset["n"]) * 1.6 + 6).tolist()
        colors = [CLASS_COLOR[k] for k in subset["klass"]]
        custom = subset[[
            "ARR_A", "ARR_B", "ARR_C", "ARR_low", "ARR_high",
            "paradigm", "grade", "n",
        ]].values
        fig.add_trace(go.Scatter3d(
            x=subset["round_M"], y=subset["post_M"], z=subset["ARR_blend"],
            mode="markers+text",
            text=subset["sector"],
            textposition="top center",
            textfont=dict(size=10, color="#222"),
            marker=dict(
                size=sizes, color=colors, opacity=0.9, symbol=symbol,
                line=dict(color="black", width=1),
            ),
            error_z=dict(
                type="data",
                symmetric=False,
                array=(subset["ARR_high"] - subset["ARR_blend"]).tolist(),
                arrayminus=(subset["ARR_blend"] - subset["ARR_low"]).clip(lower=0).tolist(),
                color="rgba(0,0,0,0.4)", thickness=1.4, width=4,
            ),
            customdata=custom,
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Round size: $%{x:.1f}M<br>"
                "Post-money: $%{y:.1f}M<br>"
                "<b>ARR (blended): $%{z:.2f}M</b><br>"
                "ARR band: $%{customdata[3]:.2f}M to $%{customdata[4]:.2f}M<br>"
                "<br>Estimator A (multiple inv): $%{customdata[0]:.2f}M"
                "<br>Estimator B (burn implied): $%{customdata[1]:.2f}M"
                "<br>Estimator C (empirical prior): $%{customdata[2]:.2f}M"
                "<br>Pricing paradigm: %{customdata[5]}"
                "<br>Confidence grade: %{customdata[6]}"
                "<br>Rounds (proxy): %{customdata[7]}"
                "<extra></extra>"
            ),
            name=label,
        ))

    fig.update_layout(
        title=dict(
            text=(
                "<b>Seed Stage Sector Map - 3D</b><br>"
                "<sub>X: median round $M | Y: median post-money $M | "
                "Z: estimated ARR at seed $M (3-estimator blend)<br>"
                "Source: Carta seed data Apr 2025 - Mar 2026, 1,346 US rounds. "
                "<a href='methodology.md' style='color:#2563eb'>Methodology</a></sub>"
            ),
            x=0.5, xanchor="center",
            font=dict(size=15),
        ),
        scene=dict(
            xaxis=dict(title="Round Size ($M)", backgroundcolor="white",
                       gridcolor="rgba(0,0,0,0.1)", title_font=dict(size=11)),
            yaxis=dict(title="Post-Money ($M)", backgroundcolor="white",
                       gridcolor="rgba(0,0,0,0.1)", title_font=dict(size=11)),
            zaxis=dict(title="Est. ARR ($M)", backgroundcolor="white",
                       gridcolor="rgba(0,0,0,0.1)", title_font=dict(size=11)),
            camera=dict(eye=dict(x=1.7, y=1.7, z=0.9)),
            aspectmode="cube",
        ),
        legend=dict(
            x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.7)",
            bordercolor="rgba(0,0,0,0.2)", borderwidth=1,
            font=dict(size=11),
        ),
        autosize=True,
        margin=dict(l=0, r=0, t=80, b=0),
        paper_bgcolor="white",
        hoverlabel=dict(font=dict(size=13), bgcolor="white",
                        bordercolor="rgba(0,0,0,0.3)"),
    )
    return fig


HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="theme-color" content="#ffffff">
<title>Seed Stage Sector Map 3D - Carta Apr 2025 to Mar 2026</title>
<meta name="description" content="3D bubble chart of US seed rounds by sector, with estimated ARR at seed (Z axis). Source: Carta, 1,346 rounds Apr 2025 to Mar 2026.">
<style>
*, *::before, *::after { box-sizing: border-box; }
html, body {
  margin: 0; padding: 0;
  height: 100%; width: 100%;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  background: #fff; color: #111;
  overflow: hidden;
  -webkit-font-smoothing: antialiased;
}
#chart {
  position: fixed; inset: 0;
  width: 100vw; height: 100vh;
  height: 100dvh;
}
.bar {
  position: fixed; top: 0; left: 0; right: 0;
  display: flex; align-items: center; gap: 8px;
  padding: 8px 12px;
  background: rgba(255,255,255,0.92);
  backdrop-filter: saturate(160%) blur(8px);
  -webkit-backdrop-filter: saturate(160%) blur(8px);
  border-bottom: 1px solid rgba(0,0,0,0.08);
  z-index: 10;
  font-size: 13px;
}
.bar select {
  flex: 1; min-width: 0;
  padding: 6px 8px;
  font-size: 14px;
  border: 1px solid rgba(0,0,0,0.15);
  border-radius: 6px;
  background: #fff;
  -webkit-appearance: none; appearance: none;
  background-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='12' height='12' viewBox='0 0 24 24' fill='none' stroke='%23666' stroke-width='2'><polyline points='6 9 12 15 18 9'/></svg>");
  background-repeat: no-repeat;
  background-position: right 8px center;
  padding-right: 28px;
}
.bar a {
  color: #2563eb; text-decoration: none;
  padding: 6px 10px; border-radius: 6px;
  border: 1px solid rgba(37,99,235,0.25);
  font-weight: 500; white-space: nowrap;
}
.bar a:active { background: rgba(37,99,235,0.08); }
.panel {
  position: fixed; left: 12px; right: 12px; bottom: 12px;
  max-width: 520px;
  margin: 0 auto;
  background: rgba(255,255,255,0.96);
  backdrop-filter: saturate(160%) blur(8px);
  -webkit-backdrop-filter: saturate(160%) blur(8px);
  border: 1px solid rgba(0,0,0,0.1);
  border-radius: 12px;
  padding: 12px 14px;
  box-shadow: 0 4px 24px rgba(0,0,0,0.08);
  font-size: 13px;
  line-height: 1.45;
  z-index: 9;
  display: none;
}
.panel.open { display: block; }
.panel h3 { margin: 0 0 6px; font-size: 16px; }
.panel .row { display: flex; justify-content: space-between; gap: 12px; }
.panel .row + .row { margin-top: 2px; }
.panel .row span:first-child { color: #555; }
.panel .row span:last-child { font-variant-numeric: tabular-nums; font-weight: 500; }
.panel .grade {
  display: inline-block; margin-left: 6px; padding: 1px 6px;
  border-radius: 999px; font-size: 11px; font-weight: 600;
  background: #f1f5f9; color: #334155;
}
.panel .close {
  position: absolute; top: 8px; right: 10px;
  background: none; border: 0; font-size: 22px; cursor: pointer;
  color: #888; padding: 0; line-height: 1;
}
.bar .icon-btn {
  flex: 0 0 auto;
  padding: 6px 10px;
  border-radius: 6px; border: 1px solid rgba(0,0,0,0.15);
  background: #fff; cursor: pointer;
  font-size: 13px;
}
@media (max-width: 600px) {
  .bar { font-size: 12px; padding: 6px 8px; gap: 6px; }
  .bar select { font-size: 13px; padding: 5px 7px; padding-right: 26px; }
  .bar a, .bar .icon-btn { padding: 5px 8px; font-size: 12px; }
  #chart { padding-top: 0; }
}
.legend-key {
  position: fixed; bottom: 12px; left: 12px;
  background: rgba(255,255,255,0.95);
  border: 1px solid rgba(0,0,0,0.08);
  border-radius: 8px;
  padding: 8px 10px;
  font-size: 11px; line-height: 1.4;
  z-index: 8;
  max-width: 200px;
}
.legend-key b { display: block; margin-bottom: 4px; font-size: 12px; }
.legend-key .key-row { display: flex; align-items: center; gap: 6px; }
.legend-key .dot { width: 10px; height: 10px; border-radius: 50%; background: #2980b9; flex: 0 0 auto; }
.legend-key .diamond { width: 10px; height: 10px; border: 1.5px solid #444; transform: rotate(45deg); flex: 0 0 auto; }
@media (max-width: 600px) {
  .legend-key { font-size: 10px; padding: 6px 8px; max-width: 160px; }
}
</style>
</head>
<body>
<div class="bar" role="toolbar">
  <select id="sector-select" aria-label="Jump to sector">
    <option value="">Jump to sector...</option>
    __SECTOR_OPTIONS__
  </select>
  <button class="icon-btn" id="reset-cam" title="Reset view">Reset</button>
  <a href="methodology.md">Methodology</a>
</div>

<div id="chart">__PLOTLY_DIV__</div>

<div class="legend-key">
  <b>Marker key</b>
  <div class="key-row"><span class="dot"></span> ARR-anchored</div>
  <div class="key-row"><span class="diamond"></span> Pre-revenue / option-priced</div>
</div>

<div class="panel" id="info" role="dialog" aria-live="polite">
  <button class="close" id="info-close" aria-label="Close">&times;</button>
  <h3 id="info-title"></h3>
  <div class="row"><span>Round size</span><span id="info-round"></span></div>
  <div class="row"><span>Post-money</span><span id="info-post"></span></div>
  <div class="row"><span>Blended ARR <span class="grade" id="info-grade"></span></span><span id="info-arr"></span></div>
  <div class="row"><span>ARR band</span><span id="info-band"></span></div>
  <div class="row"><span>Estimator A (multiple)</span><span id="info-a"></span></div>
  <div class="row"><span>Estimator B (burn)</span><span id="info-b"></span></div>
  <div class="row"><span>Estimator C (prior)</span><span id="info-c"></span></div>
  <div class="row"><span>Pricing paradigm</span><span id="info-paradigm"></span></div>
</div>

<script>
const SECTOR_DATA = __SECTOR_DATA_JSON__;
const sel = document.getElementById('sector-select');
const panel = document.getElementById('info');
const closeBtn = document.getElementById('info-close');
const resetBtn = document.getElementById('reset-cam');

function fmt(v) { return (v == null) ? '-' : ('$' + Number(v).toFixed(2) + 'M'); }

function showPanel(name) {
  const d = SECTOR_DATA[name];
  if (!d) return;
  document.getElementById('info-title').textContent = name;
  document.getElementById('info-round').textContent = fmt(d.round);
  document.getElementById('info-post').textContent = fmt(d.post);
  document.getElementById('info-arr').textContent = fmt(d.arr);
  document.getElementById('info-band').textContent = fmt(d.low) + ' to ' + fmt(d.high);
  document.getElementById('info-a').textContent = fmt(d.a);
  document.getElementById('info-b').textContent = fmt(d.b);
  document.getElementById('info-c').textContent = fmt(d.c);
  document.getElementById('info-paradigm').textContent = d.paradigm;
  document.getElementById('info-grade').textContent = d.grade;
  panel.classList.add('open');
}

sel.addEventListener('change', () => {
  if (sel.value) showPanel(sel.value);
});
closeBtn.addEventListener('click', () => {
  panel.classList.remove('open');
  sel.value = '';
});

const gd = document.querySelector('#chart .plotly-graph-div') || document.querySelector('#chart .js-plotly-plot');
function bindPlotlyEvents() {
  const el = document.querySelector('#chart .js-plotly-plot') || document.querySelector('#chart .plotly-graph-div');
  if (!el || !el.on) { setTimeout(bindPlotlyEvents, 100); return; }
  el.on('plotly_click', (e) => {
    if (!e || !e.points || !e.points.length) return;
    const name = e.points[0].text;
    if (name) showPanel(name);
  });
  resetBtn.addEventListener('click', () => {
    Plotly.relayout(el, {
      'scene.camera': { eye: {x: 1.7, y: 1.7, z: 0.9} },
    });
  });
}
bindPlotlyEvents();
window.addEventListener('resize', () => {
  const el = document.querySelector('#chart .js-plotly-plot');
  if (el && window.Plotly) Plotly.Plots.resize(el);
});
</script>
</body>
</html>
"""


def write_mobile_html(fig: go.Figure, df: pd.DataFrame, out_path: Path):
    """Wrap Plotly output in a mobile-friendly HTML template."""
    div_html = fig.to_html(
        include_plotlyjs="cdn",
        full_html=False,
        config={
            "responsive": True,
            "displaylogo": False,
            "modeBarButtonsToRemove": [
                "toImage", "sendDataToCloud", "lasso2d", "select2d",
                "hoverClosestCartesian", "hoverCompareCartesian",
            ],
            "scrollZoom": True,
        },
    )
    sector_options = "\n    ".join(
        f'<option value="{r.sector}">{r.sector}</option>'
        for _, r in df.sort_values("sector").iterrows()
    )
    sector_data = {
        r.sector: dict(
            round=r.round_M, post=r.post_M, arr=r.ARR_blend,
            low=r.ARR_low, high=r.ARR_high,
            a=r.ARR_A, b=r.ARR_B, c=r.ARR_C,
            paradigm=r.paradigm, grade=r.grade,
        )
        for _, r in df.iterrows()
    }
    import json
    html = (HTML_TEMPLATE
            .replace("__PLOTLY_DIV__", div_html)
            .replace("__SECTOR_OPTIONS__", sector_options)
            .replace("__SECTOR_DATA_JSON__", json.dumps(sector_data)))
    out_path.write_text(html, encoding="utf-8")


def main():
    df = build()
    df = df.sort_values("ARR_blend", ascending=False).reset_index(drop=True)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    fig = make_figure(df)
    write_mobile_html(fig, df, HTML_OUT)
    print(f"\nWrote interactive 3D chart -> {HTML_OUT}")
    print(f"\nSector count: {len(df)}")
    print(f"Total rounds (proxy sum): {int(df['n'].sum())}")
    print(f"Median round (sector-equal-weighted): ${df['round_M'].median():.2f}M  "
          f"(Carta chart anchor: $3.5M)")
    print(f"Median post-money (sector-equal-weighted): ${df['post_M'].median():.2f}M  "
          f"(Carta chart anchor: $19.8M)")
    print(f"Round-weighted mean ARR: ${(df['ARR_blend']*df['n']).sum()/df['n'].sum():.2f}M")

    # Markdown table for methodology.md
    cols = ["sector", "klass", "paradigm", "round_M", "post_M", "n",
            "ARR_A", "ARR_B", "ARR_C", "ARR_blend", "ARR_low", "ARR_high", "grade"]
    md = df[cols].to_markdown(index=False, floatfmt=".2f")
    print("\nMarkdown table for methodology.md:\n")
    print(md)


if __name__ == "__main__":
    main()
