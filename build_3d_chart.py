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

    # Plain-language pricing paradigm
    paradigm_plain = {
        "ARR": "priced on revenue",
        "GMV-take-rate": "priced on revenue",
        "mixed": "mixed (priced on narrative + revenue)",
        "NPV/option": "priced on milestones",
        "rNPV": "priced on milestones",
        "token-FDV": "priced on narrative (tokens)",
    }
    df = df.copy()
    df["paradigm_plain"] = df["paradigm"].map(paradigm_plain).fillna(df["paradigm"])

    # --- Sector bubbles, hybrid color: ARR-anchored = Plasma; pre-revenue = grey ---
    arr_max = max(df["ARR_blend"].max(), 1.2)
    arr_anchored = df[~df["paradigm"].isin(NON_ARR_PARADIGMS)]
    pre_revenue = df[df["paradigm"].isin(NON_ARR_PARADIGMS)]

    def _hover(custom_idx_paradigm=5):
        return (
            "<b>%{customdata[8]}</b><br>"
            "<i style='color:#fcd45f'>%{customdata[9]}</i><br>"
            "<br>"
            "Round size: <b>$%{x:.1f}M</b>  &nbsp; Post-money: <b>$%{y:.1f}M</b><br>"
            "Blended ARR: <b>$%{z:.2f}M</b>  "
            "(±50% band $%{customdata[3]:.2f}-%{customdata[4]:.2f}M)<br>"
            "<br>"
            "&nbsp;A multiple-inv: $%{customdata[0]:.2f}M<br>"
            "&nbsp;B burn-implied: $%{customdata[1]:.2f}M<br>"
            "&nbsp;C empirical prior: $%{customdata[2]:.2f}M<br>"
            "<br>Confidence grade: %{customdata[6]}  &nbsp;"
            "Rounds: %{customdata[7]}"
            "<extra></extra>"
        )

    if not arr_anchored.empty:
        custom = arr_anchored[[
            "ARR_A", "ARR_B", "ARR_C", "ARR_low", "ARR_high",
            "paradigm", "grade", "n", "sector", "paradigm_plain",
        ]].values
        sizes = (np.sqrt(arr_anchored["n"]) * 2.8 + 11).tolist()
        fig.add_trace(go.Scatter3d(
            x=arr_anchored["round_M"], y=arr_anchored["post_M"], z=arr_anchored["ARR_blend"],
            mode="markers",
            text=arr_anchored["sector"],
            marker=dict(
                size=sizes,
                color=arr_anchored["ARR_blend"],
                colorscale="Plasma",
                cmin=0, cmax=arr_max,
                opacity=0.95, symbol="circle",
                line=dict(color="rgba(255,255,255,0.55)", width=1.3),
                showscale=False,
            ),
            customdata=custom,
            projection=dict(z=dict(show=True, opacity=0.32, scale=0.7)),
            hovertemplate=_hover(),
            name="Priced on revenue",
        ))

    if not pre_revenue.empty:
        custom = pre_revenue[[
            "ARR_A", "ARR_B", "ARR_C", "ARR_low", "ARR_high",
            "paradigm", "grade", "n", "sector", "paradigm_plain",
        ]].values
        sizes = (np.sqrt(pre_revenue["n"]) * 2.6 + 10).tolist()
        fig.add_trace(go.Scatter3d(
            x=pre_revenue["round_M"], y=pre_revenue["post_M"], z=pre_revenue["ARR_blend"],
            mode="markers",
            text=pre_revenue["sector"],
            marker=dict(
                size=sizes,
                color="rgba(140,150,180,0.55)",
                opacity=0.85, symbol="diamond-open",
                line=dict(color="rgba(180,190,210,0.7)", width=1.4),
            ),
            customdata=custom,
            projection=dict(z=dict(show=True, opacity=0.22, scale=0.66)),
            hovertemplate=_hover(),
            name="Priced on milestones / narrative",
        ))

    # --- Headline annotations (Knaflic + Burn-Murdoch: argue, don't decorate) ---
    annotations_3d = []

    if "AI Infra" in df["sector"].values:
        r = df[df["sector"] == "AI Infra"].iloc[0]
        annotations_3d.append(dict(
            x=r["round_M"], y=r["post_M"], z=r["ARR_blend"],
            text=("<b>AI Infra</b><br>"
                  "<span style='font-size:10px;color:rgba(245,247,255,0.7)'>"
                  "$65M post, $0.54M ARR.<br>"
                  "Thinking Machines $12B / $0 rev<br>drags the cohort.</span>"),
            showarrow=True, arrowhead=0,
            arrowcolor="rgba(245,180,80,0.75)",
            arrowsize=1.0, arrowwidth=1.2, ax=80, ay=20,
            font=dict(size=11.5, color="rgba(252,205,103,0.98)",
                      family="Inter, system-ui, sans-serif"),
            align="left",
            bgcolor="rgba(8,10,18,0.92)",
            bordercolor="rgba(245,180,80,0.45)",
            borderpad=7, borderwidth=1,
            opacity=0.97,
        ))

    if "Analytics" in df["sector"].values:
        r = df[df["sector"] == "Analytics"].iloc[0]
        annotations_3d.append(dict(
            x=r["round_M"], y=r["post_M"], z=r["ARR_blend"],
            text=("<b>Analytics — ARR leader</b><br>"
                  "<span style='font-size:10px;color:rgba(245,247,255,0.7)'>"
                  "$0.96M ARR on $36M post.</span>"),
            showarrow=True, arrowhead=0,
            arrowcolor="rgba(232,90,140,0.75)",
            arrowsize=1.0, arrowwidth=1.2, ax=-95, ay=15,
            font=dict(size=11.5, color="rgba(252,123,170,0.98)",
                      family="Inter, system-ui, sans-serif"),
            align="left",
            bgcolor="rgba(8,10,18,0.92)",
            bordercolor="rgba(232,90,140,0.4)",
            borderpad=7, borderwidth=1,
            opacity=0.97,
        ))

    # Quiet edge labels for the supporting cast
    quiet_sectors = [
        ("AI Applications", -70, 25),
        ("Marketplace",     -65, 35),
        ("Cybersecurity",    70, 30),
        ("Biotech",         -55, 35),
        ("Fintech",          65, -5),
    ]
    for sector_name, ax, ay in quiet_sectors:
        rows = df[df["sector"] == sector_name]
        if rows.empty:
            continue
        r = rows.iloc[0]
        annotations_3d.append(dict(
            x=r["round_M"], y=r["post_M"], z=r["ARR_blend"],
            text=f"{sector_name}",
            showarrow=True, arrowhead=0,
            arrowcolor="rgba(245,247,255,0.25)",
            arrowsize=1.0, arrowwidth=0.8, ax=ax, ay=ay,
            font=dict(size=10.5, color="rgba(245,247,255,0.65)",
                      family="Inter, system-ui, sans-serif"),
            bgcolor="rgba(8,10,18,0.5)",
            bordercolor="rgba(245,247,255,0.08)",
            borderpad=3, borderwidth=1,
            opacity=0.85,
        ))

    axis_common = dict(
        tickfont=dict(size=10, color="rgba(245,247,255,0.62)",
                      family="Inter, system-ui, sans-serif"),
        gridcolor="rgba(245,247,255,0.05)",
        zerolinecolor="rgba(245,247,255,0.12)",
        showbackground=True,
        showspikes=True, spikecolor="rgba(245,180,80,0.55)",
        spikesides=False, spikethickness=1.5,
    )
    fig.update_layout(
        title=None,  # title handled in HTML hero
        scene=dict(
            xaxis=dict(
                title=dict(text="Round Size ($M)",
                           font=dict(size=12, color="rgba(245,247,255,0.85)",
                                     family="Inter, system-ui, sans-serif")),
                backgroundcolor="rgba(20,24,40,0.45)",
                **axis_common,
            ),
            yaxis=dict(
                title=dict(text="Post-Money ($M)",
                           font=dict(size=12, color="rgba(245,247,255,0.85)",
                                     family="Inter, system-ui, sans-serif")),
                backgroundcolor="rgba(14,18,32,0.45)",
                **axis_common,
            ),
            zaxis=dict(
                title=dict(text="Est. ARR at Seed ($M)",
                           font=dict(size=12, color="rgba(252,205,103,0.92)",
                                     family="Inter, system-ui, sans-serif")),
                backgroundcolor="rgba(8,10,20,0.45)",
                **axis_common,
            ),
            camera=dict(eye=dict(x=1.35, y=1.35, z=0.85),
                        center=dict(x=0, y=0, z=-0.08)),
            aspectmode="auto",
            domain=dict(x=[0, 1], y=[0, 1]),
            bgcolor="rgba(0,0,0,0)",
            annotations=annotations_3d,
        ),
        showlegend=False,  # legend handled in CSS overlay
        autosize=True,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, system-ui, sans-serif",
                  color="rgba(245,247,255,0.85)"),
        hoverlabel=dict(
            font=dict(size=12, color="#f5f7ff",
                      family="Inter, system-ui, sans-serif"),
            bgcolor="rgba(8,10,18,0.96)",
            bordercolor="rgba(245,180,80,0.5)",
        ),
    )
    return fig


HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="theme-color" content="#0a0d18">
<title>Seed Stage Sector Map 3D - Carta Apr 2025 to Mar 2026</title>
<meta name="description" content="3D bubble chart of 1,346 US seed rounds by sector with estimated ARR at seed. Carta data Apr 2025 to Mar 2026.">

<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500&display=swap">

<style>
*, *::before, *::after { box-sizing: border-box; }
:root {
  --bg-0: #06080f;
  --bg-1: #0d1220;
  --bg-2: #161b2c;
  --text-0: #f5f7ff;
  --text-1: rgba(245,247,255,0.78);
  --text-2: rgba(245,247,255,0.54);
  --accent-warm: #f5b450;
  --accent-pink: #e85a8c;
  --accent-cyan: #58e1ff;
  --line: rgba(245,247,255,0.08);
  --line-2: rgba(245,247,255,0.14);
  --glass: rgba(20,24,40,0.55);
}
html, body {
  margin: 0; padding: 0;
  height: 100%; width: 100%;
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  color: var(--text-0);
  overflow: hidden;
  -webkit-font-smoothing: antialiased;
  font-feature-settings: 'cv11','ss01','ss03';
}
body {
  background:
    radial-gradient(1000px 600px at 80% -10%, rgba(232,90,140,0.16), transparent 60%),
    radial-gradient(900px 700px at -10% 110%, rgba(88,225,255,0.12), transparent 55%),
    radial-gradient(800px 500px at 50% 50%, rgba(245,180,80,0.05), transparent 70%),
    linear-gradient(180deg, var(--bg-0) 0%, var(--bg-1) 60%, var(--bg-2) 100%);
}
/* SVG noise grain overlay for atmosphere (no extra HTTP request) */
body::before {
  content: ""; position: fixed; inset: 0; z-index: 0;
  pointer-events: none;
  background-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='200' height='200'><filter id='n'><feTurbulence type='fractalNoise' baseFrequency='0.85' numOctaves='2'/><feColorMatrix values='0 0 0 0 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0.6 0'/></filter><rect width='100%25' height='100%25' filter='url(%23n)' opacity='0.5'/></svg>");
  opacity: 0.05;
  mix-blend-mode: overlay;
}
/* Focus rings - Linear-tier accessibility */
:focus-visible {
  outline: none;
  box-shadow: 0 0 0 2px var(--accent-warm), 0 0 0 4px rgba(245,180,80,0.18);
  border-radius: 8px;
}

/* ---------- HERO ---------- */
.hero {
  position: fixed; top: 0; left: 0; right: 0;
  z-index: 12;
  padding: 18px 24px 14px;
  display: grid;
  grid-template-columns: minmax(0,1fr) auto;
  gap: 14px 18px;
  align-items: end;
  pointer-events: none;
}
.hero-titles { pointer-events: auto; max-width: 760px; }
.hero-eyebrow {
  font-family: 'JetBrains Mono', monospace;
  font-size: 11px; letter-spacing: 0.16em; text-transform: uppercase;
  color: var(--accent-warm); margin-bottom: 6px;
  opacity: 0.9;
}
.hero-title {
  font-size: clamp(20px, 3.2vw, 32px);
  font-weight: 800; letter-spacing: -0.025em;
  line-height: 1.08;
  background: linear-gradient(180deg, #ffffff 0%, #c8cee0 100%);
  -webkit-background-clip: text; background-clip: text;
  -webkit-text-fill-color: transparent;
  margin: 0;
}
.hero-title em {
  font-style: normal;
  background: linear-gradient(95deg, #fcd45f 0%, #e85a8c 70%);
  -webkit-background-clip: text; background-clip: text;
  -webkit-text-fill-color: transparent;
}
.hero-sub {
  font-size: clamp(12px, 1.35vw, 14px);
  color: var(--text-1);
  margin-top: 8px;
  max-width: 60ch;
  line-height: 1.5;
}
.hero-sub a { color: var(--accent-warm); text-decoration: none; border-bottom: 1px dotted var(--accent-warm); }

/* Contrast stat cards: not metadata, the argument itself */
.stats {
  display: flex; gap: 10px; pointer-events: auto;
}
.stat {
  background: var(--glass);
  border: 1px solid var(--line-2);
  border-radius: 14px;
  padding: 10px 14px;
  min-width: 132px;
  backdrop-filter: blur(12px) saturate(160%);
  -webkit-backdrop-filter: blur(12px) saturate(160%);
  position: relative;
  overflow: hidden;
}
.stat::before {
  content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 3px;
  background: var(--accent-warm);
}
.stat[data-tone="leader"]::before { background: linear-gradient(180deg, #fcd45f, #e85a8c); }
.stat[data-tone="laggard"]::before { background: rgba(245,180,80,0.45); }
.stat[data-tone="grey"]::before { background: rgba(160,170,200,0.45); }
.stat .l {
  font-size: 9.5px; letter-spacing: 0.12em; text-transform: uppercase;
  color: var(--text-2); margin-bottom: 4px;
}
.stat .v {
  font-size: clamp(14px, 1.7vw, 17px);
  font-weight: 700; letter-spacing: -0.02em;
  font-variant-numeric: tabular-nums;
  color: var(--text-0);
}
.stat .s {
  font-size: 10.5px;
  color: var(--text-2); margin-top: 2px;
  line-height: 1.3;
}
.stat .s b { color: var(--accent-warm); font-weight: 600; }

/* ---------- CONTROL DOCK (right side, vertically centered) ---------- */
.dock {
  position: fixed; top: 50%; right: 16px; z-index: 13;
  transform: translateY(-50%);
  display: flex; flex-direction: column; gap: 10px; align-items: stretch;
  pointer-events: auto;
}
.dock-card {
  background: var(--glass);
  border: 1px solid var(--line-2);
  border-radius: 12px;
  padding: 8px;
  backdrop-filter: blur(12px) saturate(160%);
  -webkit-backdrop-filter: blur(12px) saturate(160%);
  display: flex; gap: 6px; flex-wrap: wrap;
  font-size: 12px;
}
.dock-card .label {
  width: 100%;
  font-family: 'JetBrains Mono', monospace;
  font-size: 9.5px; letter-spacing: 0.14em; text-transform: uppercase;
  color: var(--text-2);
  padding: 0 4px 2px;
}
.btn {
  background: rgba(245,247,255,0.05);
  color: var(--text-0);
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 7px 11px; font-size: 12px; font-weight: 500;
  cursor: pointer;
  transition: background 120ms, border-color 120ms, color 120ms;
  white-space: nowrap;
  font-family: inherit;
}
.btn:hover { background: rgba(245,247,255,0.10); border-color: var(--line-2); }
.btn.active {
  background: rgba(245,180,80,0.16);
  border-color: rgba(245,180,80,0.55);
  color: var(--accent-warm);
}
.btn:active { transform: translateY(1px); }
select.btn {
  -webkit-appearance: none; appearance: none;
  background-image:
    linear-gradient(rgba(245,247,255,0.05),rgba(245,247,255,0.05)),
    url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='12' height='12' viewBox='0 0 24 24' fill='none' stroke='%23bbb' stroke-width='2'><polyline points='6 9 12 15 18 9'/></svg>");
  background-repeat: no-repeat;
  background-position: right 8px center;
  padding-right: 26px;
  min-width: 160px;
}
/* Native dropdown options - force dark theme so they're readable when open */
select.btn option {
  background-color: #161b2c;
  color: var(--text-0);
  padding: 6px 8px;
}
select.btn option:checked,
select.btn option:hover {
  background-color: #2a3050;
  color: var(--accent-warm);
}

/* ---------- CHART ---------- */
#chart {
  position: fixed; inset: 0;
  width: 100vw; height: 100vh; height: 100dvh;
}
.js-plotly-plot, .plotly-graph-div { background: transparent !important; }

/* ---------- COLORBAR OVERLAY ---------- */
.cbar {
  position: fixed; left: 16px; top: 50%; transform: translateY(-50%);
  z-index: 9;
  background: var(--glass);
  border: 1px solid var(--line-2);
  border-radius: 14px;
  padding: 14px 12px 12px;
  display: flex; flex-direction: column; align-items: center;
  backdrop-filter: blur(12px) saturate(160%);
  -webkit-backdrop-filter: blur(12px) saturate(160%);
  pointer-events: none;
}
.cbar .cbar-title {
  font-family: 'JetBrains Mono', monospace;
  font-size: 9.5px; letter-spacing: 0.14em; text-transform: uppercase;
  color: var(--text-2); margin-bottom: 8px; white-space: nowrap;
}
.cbar .cbar-bar {
  width: 14px; height: 220px; border-radius: 8px;
  background: linear-gradient(to top,
    #0d0887 0%, #46039f 14%, #7201a8 28%, #9c179e 42%,
    #bd3786 56%, #d8576b 70%, #ed7953 82%, #fb9f3a 92%, #fdca26 100%);
  border: 1px solid var(--line-2);
  position: relative;
  margin-bottom: 6px;
}
.cbar .cbar-ticks {
  position: absolute; right: 22px; top: 0; bottom: 0;
  display: flex; flex-direction: column; justify-content: space-between;
  font-size: 10px; color: var(--text-2);
  font-variant-numeric: tabular-nums;
}
.cbar .cbar-unit {
  font-size: 10px; color: var(--text-2); margin-top: 4px;
}
@media (max-width: 720px) {
  .cbar {
    right: 12px; top: auto; bottom: 96px; transform: none;
    flex-direction: row; padding: 8px 10px; gap: 8px;
    align-items: center;
  }
  .cbar .cbar-bar { width: 120px; height: 10px;
    background: linear-gradient(to right,
      #0d0887 0%, #46039f 14%, #7201a8 28%, #9c179e 42%,
      #bd3786 56%, #d8576b 70%, #ed7953 82%, #fb9f3a 92%, #fdca26 100%);
    margin: 0; }
  .cbar .cbar-title { margin: 0; font-size: 9px; }
  .cbar .cbar-unit { display: none; }
}

/* ---------- LEGEND KEY ---------- */
.legend-key {
  position: fixed; bottom: 16px; left: 16px;
  background: var(--glass);
  border: 1px solid var(--line-2);
  border-radius: 12px;
  padding: 10px 12px;
  font-size: 11px; line-height: 1.5;
  z-index: 8; max-width: 220px;
  backdrop-filter: blur(12px) saturate(160%);
  -webkit-backdrop-filter: blur(12px) saturate(160%);
  color: var(--text-1);
}
.legend-key .lk-title {
  font-family: 'JetBrains Mono', monospace;
  font-size: 9.5px; letter-spacing: 0.14em; text-transform: uppercase;
  color: var(--text-2); margin-bottom: 4px;
}
.legend-key .row { display: flex; align-items: center; gap: 8px; margin-top: 4px; }
.legend-key .dot {
  width: 12px; height: 12px; border-radius: 50%;
  background: linear-gradient(135deg, #fcd45f, #e85a8c, #5b3aaf);
  flex: 0 0 auto;
  box-shadow: 0 0 8px rgba(232,90,140,0.45);
}
.legend-key .diamond {
  width: 11px; height: 11px;
  border: 1.5px solid var(--text-1); transform: rotate(45deg);
  flex: 0 0 auto;
}

/* ---------- INFO PANEL ---------- */
.panel {
  position: fixed; right: 16px; bottom: 16px;
  width: min(360px, calc(100vw - 32px));
  background: var(--glass);
  border: 1px solid var(--line-2);
  border-radius: 16px;
  padding: 16px 18px;
  box-shadow: 0 20px 60px rgba(0,0,0,0.45);
  font-size: 13px;
  line-height: 1.5;
  z-index: 9;
  display: none;
  backdrop-filter: blur(16px) saturate(170%);
  -webkit-backdrop-filter: blur(16px) saturate(170%);
  color: var(--text-1);
}
.panel.open { display: block; animation: slide-up 220ms cubic-bezier(.2,.8,.2,1); }
@keyframes slide-up {
  from { opacity: 0; transform: translateY(8px); }
  to   { opacity: 1; transform: translateY(0); }
}
.panel h3 {
  margin: 0 0 4px; font-size: 18px; font-weight: 700; letter-spacing: -0.01em;
  color: var(--text-0);
}
.panel .meta {
  font-family: 'JetBrains Mono', monospace;
  font-size: 10.5px; letter-spacing: 0.08em; text-transform: uppercase;
  color: var(--text-2); margin-bottom: 12px;
}
.panel .row {
  display: flex; justify-content: space-between; gap: 14px;
  padding: 4px 0;
}
.panel .row + .row { border-top: 1px solid var(--line); }
.panel .row span:first-child { color: var(--text-2); }
.panel .row span:last-child {
  font-variant-numeric: tabular-nums;
  color: var(--text-0); font-weight: 600;
}
.panel .arr-row span:last-child {
  font-size: 16px; color: var(--accent-warm);
}
.panel .est {
  margin-top: 10px; padding-top: 10px;
  border-top: 1px dashed var(--line-2);
  font-size: 12px;
}
.panel .grade {
  display: inline-block; margin-left: 8px;
  padding: 1px 8px; border-radius: 999px; font-size: 10px;
  font-family: 'JetBrains Mono', monospace; font-weight: 600;
  background: rgba(88,225,255,0.14); color: var(--accent-cyan);
  border: 1px solid rgba(88,225,255,0.3);
}
.panel .grade[data-g="C"] { background: rgba(245,180,80,0.14); color: var(--accent-warm); border-color: rgba(245,180,80,0.3); }
.panel .grade[data-g="B"] { background: rgba(232,90,140,0.14); color: var(--accent-pink); border-color: rgba(232,90,140,0.3); }
.panel .close {
  position: absolute; top: 10px; right: 12px;
  background: none; border: 0; font-size: 20px; cursor: pointer;
  color: var(--text-2); padding: 0; line-height: 1;
  transition: color 120ms;
}
.panel .close:hover { color: var(--text-0); }

/* ---------- BOTTOM-LINE CALLOUT (McKinsey-style finding strip) ---------- */
.bottom-line {
  position: fixed; bottom: 16px; left: 50%; transform: translateX(-50%);
  max-width: min(900px, calc(100vw - 32px));
  display: grid; grid-template-columns: auto 1fr; gap: 14px;
  align-items: center;
  padding: 12px 18px;
  background: var(--glass);
  border: 1px solid var(--line-2);
  border-radius: 14px;
  backdrop-filter: blur(14px) saturate(170%);
  -webkit-backdrop-filter: blur(14px) saturate(170%);
  z-index: 7;
  pointer-events: auto;
  box-shadow: 0 12px 40px rgba(0,0,0,0.4);
}
.bottom-line .label {
  font-family: 'JetBrains Mono', monospace;
  font-size: 9.5px; letter-spacing: 0.16em; text-transform: uppercase;
  color: var(--accent-warm);
  border-right: 1px solid var(--line-2);
  padding-right: 14px;
  white-space: nowrap;
}
.bottom-line ul {
  margin: 0; padding: 0; list-style: none;
  display: flex; flex-direction: column; gap: 4px;
  font-size: 12.5px; line-height: 1.45;
  color: var(--text-1);
}
.bottom-line li::before {
  content: "→ ";
  color: var(--accent-warm);
  font-family: 'JetBrains Mono', monospace;
  margin-right: 4px;
}
.bottom-line b { color: var(--text-0); font-weight: 600; }
@media (max-width: 720px) {
  .bottom-line { display: none; }
}

/* ---------- IDLE HINT (only when user is stuck) ---------- */
.hint {
  position: fixed; bottom: 16px; left: 50%; transform: translate(-50%, 8px);
  background: var(--glass); border: 1px solid var(--line-2);
  border-radius: 999px; padding: 7px 16px; font-size: 11.5px;
  color: var(--text-1); z-index: 8;
  backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px);
  pointer-events: none;
  opacity: 0;
  transition: opacity 320ms ease-out, transform 320ms ease-out;
}
.hint.show { opacity: 1; transform: translate(-50%, 0); }

/* ---------- CHART ENTRANCE ---------- */
#chart {
  opacity: 0;
  transition: opacity 700ms cubic-bezier(0.2, 0.8, 0.2, 1);
}
#chart.entered { opacity: 1; }

/* ---------- MOBILE ---------- */
@media (max-width: 720px) {
  .hero {
    padding: 10px 12px 8px;
    grid-template-columns: 1fr;
    gap: 6px;
    background: linear-gradient(180deg, rgba(6,8,15,0.92) 0%, rgba(6,8,15,0.6) 80%, transparent 100%);
    backdrop-filter: blur(6px);
    -webkit-backdrop-filter: blur(6px);
  }
  .hero-eyebrow { font-size: 9.5px; letter-spacing: 0.12em; margin-bottom: 2px; }
  .hero-title { font-size: 17px; line-height: 1.15; }
  .hero-sub {
    /* on mobile, the subtitle is too verbose to live in the always-visible hero */
    display: none;
  }
  .hero-titles { padding-right: 0; }
  /* Stats scroll horizontally on small screens */
  .stats {
    overflow-x: auto;
    scrollbar-width: none;
    margin: 4px -12px 0;
    padding: 0 12px 4px;
    scroll-snap-type: x mandatory;
  }
  .stats::-webkit-scrollbar { display: none; }
  .stat {
    flex: 0 0 auto; width: 200px; min-width: 0;
    scroll-snap-align: start;
    padding: 7px 11px;
  }
  .stat .v { font-size: 12.5px; }
  .stat .l { font-size: 9px; }
  .stat .s { font-size: 9.5px; }
  .dock {
    top: auto; bottom: 88px; right: 12px; left: 12px;
    flex-direction: row; justify-content: center;
    z-index: 11;
  }
  .dock-card { padding: 6px; gap: 4px; }
  .dock-card .label { display: none; }
  select.btn { min-width: 0; flex: 1; max-width: 140px; }
  .legend-key { display: none; }
  .panel {
    left: 12px; right: 12px; width: auto; bottom: 152px;
    padding: 12px 14px;
  }
  .hint { display: none; }  /* on touch devices the hint is misleading */
}
@media (max-width: 480px) {
  .hero-title { font-size: 15px; }
  .stat { width: 180px; }
}
</style>
</head>
<body>

<header class="hero">
  <div class="hero-titles">
    <div class="hero-eyebrow">Carta · Apr 2025 to Mar 2026 · 1,346 US seed rounds</div>
    <h1 class="hero-title">Highest seed valuations <em>don't earn</em> the highest ARR.</h1>
    <p class="hero-sub">
      AI Infra raises at $65M post-money but the cohort sits at $0.54M ARR — frontier model labs
      (Thinking Machines $12B / $0 rev) drag the median to the floor. Analytics leads at $0.96M.
      Pre-revenue sectors are not priced on revenue at all. <a href="methodology.md">Methodology</a>.
    </p>
  </div>
  <div class="stats">
    <div class="stat" data-tone="laggard">
      <div class="l">Highest Valuation</div>
      <div class="v">AI Infra · $65M post</div>
      <div class="s">only <b>$0.54M</b> ARR · priced on narrative</div>
    </div>
    <div class="stat" data-tone="leader">
      <div class="l">Highest ARR</div>
      <div class="v">Analytics · $0.96M ARR</div>
      <div class="s">on $36M post · priced on revenue</div>
    </div>
    <div class="stat" data-tone="grey">
      <div class="l">5 Sectors Pre-Revenue</div>
      <div class="v">Biotech · Hardware · Semis · Renewables · MedDev</div>
      <div class="s">priced on milestones, not multiples</div>
    </div>
  </div>
</header>

<div class="dock">
  <div class="dock-card">
    <div class="label">View</div>
    <button class="btn view-btn active" data-view="iso">Iso</button>
    <button class="btn view-btn" data-view="top">Top (Carta)</button>
    <button class="btn view-btn" data-view="front">ARR vs Round</button>
    <button class="btn view-btn" data-view="side">ARR vs Val</button>
  </div>
  <div class="dock-card">
    <div class="label">Sector</div>
    <select id="sector-select" class="btn" aria-label="Jump to sector">
      <option value="">Jump to...</option>
      __SECTOR_OPTIONS__
    </select>
  </div>
</div>

<div id="chart">__PLOTLY_DIV__</div>

<aside class="legend-key" aria-hidden="true">
  <div class="lk-title">Reading this chart</div>
  <div class="row"><span class="dot"></span><span><b>Color circle</b> · priced on revenue</span></div>
  <div class="row"><span class="diamond"></span><span><b>Grey diamond</b> · priced on milestones</span></div>
  <div class="row" style="margin-top:6px; color:var(--text-2); font-size:10.5px;">
    Bubble color = ARR ($M). Size = round count. Shadow on floor recreates the original 2D Carta map.
  </div>
</aside>

<div class="panel" id="info" role="dialog" aria-live="polite">
  <button class="close" id="info-close" aria-label="Close">&times;</button>
  <h3 id="info-title"></h3>
  <div class="meta" id="info-class"></div>
  <div class="row arr-row">
    <span>Blended ARR <span class="grade" id="info-grade"></span></span>
    <span id="info-arr"></span>
  </div>
  <div class="row"><span>Sensitivity band</span><span id="info-band"></span></div>
  <div class="row"><span>Round size</span><span id="info-round"></span></div>
  <div class="row"><span>Post-money</span><span id="info-post"></span></div>
  <div class="est">
    <div class="row"><span>A — multiple inversion</span><span id="info-a"></span></div>
    <div class="row"><span>B — burn implied</span><span id="info-b"></span></div>
    <div class="row"><span>C — empirical prior</span><span id="info-c"></span></div>
  </div>
</div>

<aside class="bottom-line" aria-label="Key takeaways">
  <div class="label">Bottom<br>Line</div>
  <ul>
    <li>If you're writing seed checks, don't anchor <b>AI Infra</b> valuations to ARR multiples. The comp set is bimodal: frontier labs at $0 revenue + tooling players at $1-5M.</li>
    <li><b>Analytics, AI Apps, Marketplace, Cybersecurity</b> are the cleanest sectors for revenue-anchored seed comps. Trust the multiple.</li>
    <li>For <b>Biotech, Hardware, Semis, Renewables</b>, replace the multiple with a milestone schedule. Asking "what's the ARR" is the wrong question.</li>
  </ul>
</aside>

<div class="hint" id="idle-hint">
  drag to rotate · pinch to zoom · tap a bubble for detail
</div>

<script>
const SECTOR_DATA = __SECTOR_DATA_JSON__;
const VIEW_CAMS = {
  iso:   { eye: {x: 1.35, y: 1.35, z: 0.85}, up: {x:0,y:0,z:1}, center: {x:0,y:0,z:-0.08} },
  top:   { eye: {x: 0.001, y: 0.001, z: 2.4}, up: {x:0,y:1,z:0}, center: {x:0,y:0,z:0} },
  front: { eye: {x: 0.001, y: 2.5, z: 0.001}, up: {x:0,y:0,z:1}, center: {x:0,y:0,z:0} },
  side:  { eye: {x: 2.5, y: 0.001, z: 0.001}, up: {x:0,y:0,z:1}, center: {x:0,y:0,z:0} },
};

const sel = document.getElementById('sector-select');
const panel = document.getElementById('info');
const closeBtn = document.getElementById('info-close');
const hint = document.getElementById('idle-hint');

function fmt(v) { return (v == null) ? '-' : ('$' + Number(v).toFixed(2) + 'M'); }

function showPanel(name) {
  const d = SECTOR_DATA[name];
  if (!d) return;
  document.getElementById('info-title').textContent = name;
  document.getElementById('info-class').textContent = d.paradigm_plain + ' · ' + d.klass;
  document.getElementById('info-round').textContent = fmt(d.round);
  document.getElementById('info-post').textContent = fmt(d.post);
  document.getElementById('info-arr').textContent = fmt(d.arr);
  document.getElementById('info-band').textContent = fmt(d.low) + ' – ' + fmt(d.high);
  document.getElementById('info-a').textContent = fmt(d.a);
  document.getElementById('info-b').textContent = fmt(d.b);
  document.getElementById('info-c').textContent = fmt(d.c);
  const g = document.getElementById('info-grade');
  g.textContent = d.grade; g.dataset.g = d.grade;
  panel.classList.add('open');
}

sel.addEventListener('change', () => { if (sel.value) showPanel(sel.value); });
closeBtn.addEventListener('click', () => {
  panel.classList.remove('open'); sel.value = '';
});

function getPlot() {
  return document.querySelector('#chart .js-plotly-plot') || document.querySelector('#chart .plotly-graph-div');
}

/* ------- Camera transitions: use Plotly's native animate (GPU, single call) ------- */
function setCamera(el, target) {
  // Plotly.animate handles the transition in one call; safe and smooth.
  if (window.Plotly && Plotly.animate) {
    Plotly.animate(el, {
      layout: { 'scene.camera': target }
    }, {
      transition: { duration: 700, easing: 'cubic-out' },
      frame: { duration: 700, redraw: false }
    });
  } else {
    Plotly.relayout(el, { 'scene.camera': target });
  }
}

/* ------- Hover: just change the cursor; skip restyle, it stutters the WebGL scene ------- */
function onHover() { document.body.style.cursor = 'pointer'; }
function onUnhover() { document.body.style.cursor = ''; }

/* ------- Idle-detected hint: show only when user hasn't engaged ------- */
let _idleTimer = null;
function resetIdleTimer() {
  clearTimeout(_idleTimer);
  hint.classList.remove('show');
  if (sessionStorage.getItem('seen-hint')) return;
  _idleTimer = setTimeout(() => {
    hint.classList.add('show');
    setTimeout(() => hint.classList.remove('show'), 4000);
    sessionStorage.setItem('seen-hint', '1');
  }, 5000);
}

function bindPlotlyEvents() {
  const el = getPlot();
  if (!el || !el.on) { setTimeout(bindPlotlyEvents, 100); return; }

  // Click to pin info panel
  el.on('plotly_click', (e) => {
    if (!e || !e.points || !e.points.length) return;
    const name = e.points[0].text;
    if (name) showPanel(name);
    resetIdleTimer();
  });

  // Hover: cursor only (no per-frame restyle — that locks the WebGL scene)
  el.on('plotly_hover', () => { onHover(); resetIdleTimer(); });
  el.on('plotly_unhover', onUnhover);

  // View preset buttons (Plotly.animate handles the smooth transition)
  document.querySelectorAll('.view-btn').forEach((b) => {
    b.addEventListener('click', () => {
      document.querySelectorAll('.view-btn').forEach((x) => x.classList.remove('active'));
      b.classList.add('active');
      setCamera(el, VIEW_CAMS[b.dataset.view]);
      resetIdleTimer();
    });
  });

  // Entrance: fade chart container in (CSS handles the smooth fade)
  const wrap = document.getElementById('chart');
  wrap.classList.add('entered');

  resetIdleTimer();
}
bindPlotlyEvents();

// Idle reset on any user input
['mousemove','wheel','touchstart','keydown'].forEach((ev) =>
  window.addEventListener(ev, resetIdleTimer, { passive: true }));

// Keyboard: Escape closes panel
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') {
    panel.classList.remove('open'); sel.value = '';
  }
});

window.addEventListener('resize', () => {
  const el = getPlot();
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
    paradigm_plain = {
        "ARR": "priced on revenue",
        "GMV-take-rate": "priced on revenue",
        "mixed": "priced on narrative + revenue",
        "NPV/option": "priced on milestones",
        "rNPV": "priced on milestones",
        "token-FDV": "priced on narrative (tokens)",
    }
    sector_data = {
        r.sector: dict(
            round=r.round_M, post=r.post_M, arr=r.ARR_blend,
            low=r.ARR_low, high=r.ARR_high,
            a=r.ARR_A, b=r.ARR_B, c=r.ARR_C,
            paradigm=r.paradigm,
            paradigm_plain=paradigm_plain.get(r.paradigm, r.paradigm),
            grade=r.grade, klass=r.klass,
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
