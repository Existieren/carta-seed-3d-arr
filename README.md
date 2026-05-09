# Seed Stage Sector Map: 3D with estimated ARR

3D version of Carta's "Seed Stage Sector Map" (1,346 US seed rounds, April 2025 to March 2026), with the Z axis showing **estimated ARR at the time the seed round was raised**.

**[Open the interactive chart](https://existieren.github.io/carta-seed-3d-arr/)** (works on mobile and desktop).

X = median round size ($M), Y = median post-money valuation ($M), Z = blended ARR estimate ($M), bubble area = number of rounds, color = sector class. Open-diamond markers indicate sectors priced on NPV/option/token paradigms rather than revenue multiples (hardware, biotech, deeptech, web3 protocols).

## Methodology in one paragraph

ARR at seed is rarely disclosed publicly, so each sector's ARR is estimated three independent ways and blended via weighted geometric mean (revenue at seed is log-normal):

1. **Estimator A: stage-conditional revenue-multiple inversion** — `post_money / M_R(class)` with sector-specific seed multiples sourced from Finro, Aventis, Bessemer, High Alpha, Carta.
2. **Estimator B: burn-implied revenue** — 18-month seed runway divided by sector Net Burn Multiple (Bessemer, ICONIQ, High Alpha 2025).
3. **Estimator C: empirical Bayesian prior** — published median seed ARR per sector (Carta/Walker, Sacra, Metal.so, Pitchwise).

Sector-conditional weights down-weight the multiple-inversion estimator for option-priced and token-FDV sectors where post-money / ARR is meaningless. ±50% sensitivity bands are propagated through the blend and shown as Z-direction error bars.

Full methodology with sourced multiples, NBMs, priors, weights, and the per-sector estimator table: [methodology.md](methodology.md).

## Top-line findings

- Analytics ($0.96M) and AI Applications ($0.77M) lead by blended ARR
- AI Infra is the highest-valued sector ($65M post-money) but only sixth by blended ARR ($0.54M); the cohort is dominated by frontier model labs raising at $0 revenue (Thinking Machines, etc.)
- Round-weighted mean ARR across all 19 sectors: $0.53M

## Reproduce

```bash
pip install plotly scipy numpy pandas tabulate
python build_3d_chart.py
```

Outputs `seed_3d_chart.html` in the working directory.

## Files

- `index.html` — the interactive 3D chart (primary deliverable, served at the live URL above)
- `methodology.md` — full methodology with sources
- `build_3d_chart.py` — reproducible script
