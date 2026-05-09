# Methodology: 3D Carta Seed Sector Map with Estimated ARR

Author: Felix Förster
Date: 2026-05-09
Companion artifact: `seed_3d_chart.html`, `build_3d_chart.py`

## 1. Source chart and scope

The base chart is Carta's "Seed Stage Sector Map", covering 1,346 US seed rounds from April 2025 through March 2026. Each sector is a labeled bubble: X = median round size, Y = median post-money valuation, bubble area = number of rounds. The headline anchors stamped on the chart are median seed round $3.5M and median post-money valuation $19.8M for the period.

Carta does not publish the per-sector medians as a table. The bubble map exists as a standalone graphic (Peter Walker / Carta Insights) and the underlying CSV is gated behind State of Private Markets reports. Per-sector centroids in this analysis were read off the chart and are subject to plus or minus roughly $1M on each axis. They are clearly labeled as chart-eyeballed.

Independent corroboration of macro values: Carta reports FY2025 median seed round $4.0M and median post-money $20.0M (slightly higher than the trailing-12-month chart cut), AI = 41.7% of seed capital deployed, AI median pre-money $19M vs non-AI $13M (per Peter Walker, 2025-03-23, x.com/PeterJ_Walker/status/1903950230460244453).

## 2. Why ARR at seed is hard to observe

ARR at the moment a seed round closes is not publicly disclosed for almost any company. It is recoverable only through:

1. Self-reported survey data with response bias (OpenView, High Alpha, First Round, Lenny)
2. Cap-table platforms with privileged access (Carta, AngelList) that occasionally publish anonymized distributions
3. Inverse inference from valuation, round size, and known burn structure

The distribution is highly non-Gaussian. At seed:
- A large mass sits at exactly $0 (deep tech, biotech, frontier AI, hardware where pricing is option-based or rNPV, not revenue-multiple)
- A long upper tail driven by AI vibe revenue and SaaS companies that closed paid pilots before raising
- Within revenue-bearing sectors, the conditional distribution is approximately log-normal (consistent with multiplicative growth dynamics)

These features force three methodological choices:

1. We use the geometric mean for blending (preserves log-normality)
2. We separate revenue-bearing from option-priced sectors via a sector-conditional weight matrix
3. We attach a plus or minus 50% sensitivity band to every point estimate, since the true uncertainty at the cohort median is closer to plus or minus 50% than the plus or minus 5% confidence intervals one might naively report

## 3. Three-estimator construction

### 3.1 Estimator A: stage-conditional revenue-multiple inversion

ARR_A = post_money_valuation / M_R(class)

The seed-stage post-money / ARR multiple M_R differs sharply by sector. The values below are the central estimates, with low and high sensitivity points; the script applies plus or minus 50% perturbations from the central value, but the bracketed bands shown below describe the realistic spread of comparable observations:

| Class | Low | Central M_R | High | Anchor sources |
|---|---:|---:|---:|---|
| ai_infra | 50 | 150 | 1000+ | Thinking Machines $12B / $0 rev seed (TechCrunch, 2025-07-15); Finro AI Q1 2025 LLM-vendor 44.1x; carta.com/data/state-of-pre-seed-q1-2025/ |
| ai_app | 25 | 60 | 200 | Carta AI seed pre-money $19M / Sacra Cursor / Glean ARR; aventis-advisors.com/ai-valuation-multiples/ |
| saas | 15 | 40 | 100 | High Alpha 2025 SaaS Benchmarks; metal.so US SaaS Seed 2025; Aventis SaaS multiples 2015-2026 |
| fintech | 15 | 35 | 90 | Windsor Drake Fintech Multiples 2025; Carta fintech seed pre-money cap |
| marketplace | 8 | 25 | 70 | aventis-advisors.com/marketplace-valuation-multiples/; Whatnot 16.5x net rev anchor |
| hardware | 50 | 200 | 2000 | Figure $39B / 0 rev (Humanoids Daily); Anvil $5.5M / 0 rev (Crunchbase News); Carta Q1 2025 hardware top-three SAFE caps |
| deeptech_silicon | 75 | 250 | 2000 | Pre-tape-out $20M-$40M post-money / 0 rev; option-pricing on tape-out + LOIs |
| deeptech_climate | 25 | 100 | 1000 | finerva.com/report/green-energy-renewables-2025-valuation-multiples/; Sightline Climate $40.5B 2025 |
| web3 | 30 | 150 | 2000 | Carta Q1 2025 highest crypto SAFE caps; RootData 2025 avg seed $5.5M; FDV-priced |
| healthtech | 15 | 40 | 100 | Carta healthtech median $35M cap on rounds >=$2.5M (2025) |
| biotech | 100 | 500 | 5000 | finrofca.com/news/biotech-revenue-multiples-2025; >80% of biotech seeds milestone-structured (rNPV) |
| consumer | 5 | 15 | 50 | learn.icanpitch.com/blog/seed-valuation-benchmarks-2025/; consumer category $2.1M median val |

For sectors where the high column trends toward unbounded (hardware, deeptech, biotech), Estimator A approaches `valuation / 0` and is intellectually weak. The blend handles this by down-weighting w_A near zero for those classes (see Section 5).

### 3.2 Estimator B: burn-implied revenue

The seed runway target is 18 months at industry-median efficiency. From there, a sector-specific Net Burn Multiple (NBM = monthly net burn / monthly net new ARR) implies the rate of ARR being added against the burn:

monthly_burn = round_M / 18
ARR_B = monthly_burn x 12 / NBM = round_M x (12/18) / NBM

NBM bands at the seed stage in 2025, sourced from a follow-up live research sweep against High Alpha 2025 SaaS Benchmarks, ICONIQ State of Software 2025, Bessemer State of AI 2025, ICanPitch burn benchmarks, and CFO Advisors 2025:

| Class | Low | Central | High | Anchor sources |
|---|---:|---:|---:|---|
| ai_infra | 1.5 | 2.5 | 4.0 | Bessemer State of AI 2025; AI infra burns relatively heavily on GPU/compute; cohort split between frontier (high NBM) and tooling (low NBM); INFERRED at high end |
| ai_app | 0.8 | 1.2 | 2.0 | Bessemer / ICONIQ / High Alpha 2025 confirm AI-native stays consistently sub-1.5x; cfoadvisors.com/blog/2025-burn-multiple-benchmarks |
| saas | 1.5 | 2.5 | 3.5 | learn.icanpitch.com/blog/burn-rate-benchmarks-by-industry-stage/; High Alpha 2025 |
| fintech | 2.0 | 3.5 | 5.0 | ICanPitch + regulatory drag, INFERRED +42% vs SaaS |
| marketplace | 2.0 | 3.0 | 5.0 | gardinercolin.com/p/marketplace-startup-fundraising-2025; supply-side burn pre-liquidity |
| hardware | n/a | n/a | n/a | Pre-revenue at seed; NBM not the right metric, ARR_B set to 0 |
| deeptech_silicon | n/a | n/a | n/a | Pre-revenue, $5-10M seed for tech-readiness; albion.vc deeptech 2025 |
| deeptech_climate | n/a | n/a | n/a | Capex-driven; ctvc.co climate-tech 2024 |
| web3 | n/a | n/a | n/a | Token-fundraising replaces ARR; outlierventures.io/article/how-to-raise-a-pre-seed-round-in-web3 |
| healthtech | 2.0 | 3.0 | 4.5 | onhealthcare.tech 2025 playbook; pilots often free or grant-funded |
| biotech | n/a | n/a | n/a | rNPV / phase-gated milestones; qubit.capital biotech seed modeling |
| consumer | 2.5 | 4.0 | 6.0 | Carta Consumer Industry Spotlight Q1 2025 (carta.com/data/industry-spotlight-consumer-q1-2025/); paid-CAC heavy |

Cross-check anchors: Bessemer 2025 cap on "acceptable" institutional NBM = 1.5x (Series B); High Alpha 2025 reports pre-seed/seed average 2.5-3.4x for non-AI; ICONIQ early-stage ($10-20M ARR) median 1.5-2.0x; AI-native compresses to 0.8-1.2x at every stage. NBM data for seed by sector is sparser than valuation-multiple data because OpenView wound down (2024 SaaS Benchmarks is the last canonical snapshot) and Bessemer's State of the Cloud 2025 reports efficiency at growth stage rather than seed. Inferred values are flagged in the script comments and the source table.

### 3.3 Estimator C: empirical Bayesian prior

The empirical median seed ARR by sector for 2025-2026, anchored by a focused live research sweep against:

- Peter Walker / Carta posts and Carta Insights ARR-at-seed distributions (carta.com/data)
- Sacra company breakdowns: Cursor / Anysphere ($500M+ ARR by 2025), Mercor ($3.6M seed 2023), Glean ($4.5M seed 2019)
- Metal.so US SaaS Seed-Round Benchmarks 2025 (metal.so/collections/us-saas-seed-round-benchmarks-2025-...)
- Pitchwise median seed by industry 2026 (pitchwise.se/blog/median-seed-round-size-by-industry-in-2026-data)
- Carta Consumer Industry Spotlight Q1 2025 (consumer round size dropped to $700K Q1 2025)

| Class | Low ARR $M | Central | High | Anchor |
|---|---:|---:|---:|---|
| ai_infra | 0.0 | 0.2 | 1.0 | Cohort dominated by frontier pre-revenue (Thinking Machines $0 / $12B); tooling outliers (Together, Modal) at the high end |
| ai_app | 0.3 | 0.6 | 1.5 | Top decile $5M+ (Cursor pre-seed comp); Metal.so 2025 median |
| saas | 0.3 | 0.5 | 1.0 | Pitchwise + High Alpha 2025 stage medians |
| fintech | 0.2 | 0.4 | 0.8 | Pitchwise + INFERRED |
| marketplace | 0.3 | 0.6 | 1.5 | Net revenue (GMV / 5-15x take-rate); Gardiner Colin marketplace 2025 |
| hardware | 0.0 | 0.05 | 0.3 | Median near zero, small pilot deployments at the high end |
| deeptech_silicon | 0.0 | 0.0 | 0.0 | Genuinely pre-revenue at seed; Albion deeptech 2025 |
| deeptech_climate | 0.0 | 0.1 | 0.5 | Mostly grant / pilot revenue; CTVC climate-tech 2024 |
| web3 | 0.0 | 0.05 | 0.5 | Token-economy projects ARR ~ 0; outlierventures.io |
| healthtech | 0.1 | 0.3 | 0.8 | Pilots often free at seed; onhealthcare.tech 2025 |
| biotech | 0.0 | 0.0 | 0.0 | rNPV / phase-gated; qubit.capital biotech seed modeling |
| consumer | 0.1 | 0.3 | 0.7 | Carta Consumer Spotlight Q1 2025: median seed dropped to $700K |

Note: hardware, deeptech_climate, and healthtech show small but non-zero medians because in practice some seeds in these categories close paid pilots, PPAs, or deployed-unit revenue before the round. Biotech, deeptech_silicon, and most of Web3 medians sit at zero because the cohort is genuinely pre-revenue at seed. AI Infra median was revised from $1.8M to $0.2M after the live research sweep, because the Carta AI Infra label is dominated by frontier model labs (Thinking Machines, etc.) that raise huge rounds at $0 ARR; the $1M+ tooling players are the long upper tail, not the median.

## 4. Pricing-paradigm typology

Not every seed round is priced on revenue. The chart distinguishes four paradigms:

| Paradigm | How valuation is set | Sectors |
|---|---|---|
| ARR-multiple | post_money = M_R x ARR with sector-specific M_R | ai_app, saas, fintech, healthtech, consumer |
| GMV-take-rate | normalized via take rate to ARR equivalent | marketplace |
| NPV/option | discounted-cash-flow / option pricing on milestones | hardware, deeptech_silicon, deeptech_climate |
| rNPV (risk-adjusted NPV) | clinical / regulatory milestone tree | biotech, med devices |
| token-FDV | fully-diluted token valuation, often replacing ARR | web3 protocols (web3 apps revert to ARR-multiple) |
| Mixed | bimodal: frontier players option-priced, tooling players ARR-priced | ai_infra |

The weight matrix (Section 5) encodes this: option-priced and rNPV classes have w_A near zero so the multiple-inversion estimator does not contaminate the blend with garbage divisions.

## 5. Blend rationale

Revenue at seed is approximately log-normal within revenue-bearing classes, with a point mass at zero in pre-revenue classes. The geometric mean preserves that log-normal property:

log(ARR) = w_A * log(ARR_A) + w_B * log(ARR_B) + w_C * log(ARR_C)

with an epsilon floor of 0.001 to keep logs finite when an estimator returns zero.

Sector-conditional weights:

| Class | w_A | w_B | w_C | Reasoning |
|---|---:|---:|---:|---|
| ai_infra | 0.35 | 0.25 | 0.40 | Mixed paradigm; balanced |
| ai_app | 0.40 | 0.25 | 0.35 | All three estimators meaningful |
| saas | 0.40 | 0.30 | 0.30 | Most data; all three meaningful |
| fintech | 0.35 | 0.30 | 0.35 | All three meaningful |
| marketplace | 0.30 | 0.30 | 0.40 | Take-rate normalization adds prior weight |
| hardware | 0.05 | 0.00 | 0.95 | Multiple meaningless, NBM nonsense, prior dominates |
| deeptech_silicon | 0.05 | 0.00 | 0.95 | Same |
| deeptech_climate | 0.10 | 0.00 | 0.90 | Some revenue-bearing edge cases |
| web3 | 0.20 | 0.00 | 0.80 | Token-fundraising replaces ARR for most cohort; NBM not meaningful |
| healthtech | 0.25 | 0.25 | 0.50 | Long sales cycles weaken multiple/burn signals |
| biotech | 0.02 | 0.00 | 0.98 | rNPV; prior alone is meaningful |
| consumer | 0.30 | 0.25 | 0.45 | Light premium, prior leads |

## 6. Sensitivity analysis

Each estimator input (M_R, NBM, prior central) is perturbed plus or minus 50% to produce ARR_low and ARR_high per sector. The script propagates these through the geometric blend so the bands are mathematically consistent with the central estimate.

Visualized as Z-direction error bars on each 3D bubble. The width of the bar communicates real uncertainty; for biotech and deeptech_silicon the bar collapses to near zero because the prior is itself near zero.

## 7. Per-sector estimator table (output of `build_3d_chart.py`)

| sector          | klass            | paradigm      | round_M | post_M |   n | ARR_A | ARR_B | ARR_C | ARR_blend | ARR_low | ARR_high | grade |
|:----------------|:-----------------|:--------------|--------:|-------:|----:|------:|------:|------:|----------:|--------:|---------:|:------|
| Analytics       | ai_app           | ARR           |    5.50 |  36.00 | 220 |  0.60 |  3.06 |  0.60 |      0.90 |    0.38 |     1.95 | B     |
| AI Applications | ai_app           | ARR           |    4.00 |  30.00 | 350 |  0.50 |  2.22 |  0.60 |      0.77 |    0.33 |     1.68 | B     |
| Marketplace     | marketplace      | GMV-take-rate |    5.00 |  18.00 |  60 |  0.72 |  1.11 |  0.60 |      0.76 |    0.36 |     1.75 | A     |
| Cybersecurity   | saas             | ARR           |    5.00 |  21.00 |  90 |  0.53 |  1.33 |  0.50 |      0.68 |    0.37 |     1.45 | B     |
| Fintech         | fintech          | ARR           |    4.00 |  26.00 | 130 |  0.74 |  0.76 |  0.40 |      0.60 |    0.31 |     1.22 | A     |
| AI Infra        | ai_infra         | mixed         |   13.50 |  65.00 |  70 |  0.43 |  3.60 |  0.20 |      0.54 |    0.03 |     1.72 | B     |
| Proptech        | saas             | ARR           |    2.70 |  18.00 |  45 |  0.45 |  0.72 |  0.50 |      0.53 |    0.29 |     1.14 | A     |
| Media           | consumer         | ARR           |    3.00 |  20.00 |  40 |  1.33 |  0.50 |  0.30 |      0.53 |    0.20 |     1.22 | B     |
| HR              | saas             | ARR           |    3.50 |  13.00 |  45 |  0.33 |  0.93 |  0.50 |      0.51 |    0.27 |     1.08 | B     |
| Personal        | consumer         | ARR           |    2.00 |  16.00 |  35 |  1.07 |  0.33 |  0.30 |      0.45 |    0.17 |     1.03 | B     |
| Logistics       | saas             | ARR           |    2.50 |  12.00 |  35 |  0.30 |  0.67 |  0.50 |      0.44 |    0.24 |     0.94 | B     |
| Healthtech      | healthtech       | ARR           |    4.50 |  17.00 |  65 |  0.42 |  1.00 |  0.30 |      0.44 |    0.18 |     1.02 | B     |
| Renewables      | deeptech_climate | NPV/option    |    5.00 |  19.00 |  55 |  0.19 |  0.00 |  0.10 |      0.11 |    0.00 |     0.52 | C     |
| Web3            | web3             | token-FDV     |    5.20 |  28.00 |  75 |  0.19 |  0.00 |  0.05 |      0.07 |    0.00 |     0.57 | C     |
| Hardware        | hardware         | NPV/option    |    5.30 |  23.00 |  70 |  0.12 |  0.00 |  0.05 |      0.05 |    0.00 |     0.31 | C     |
| Transport       | hardware         | NPV/option    |    3.50 |  22.00 |  50 |  0.11 |  0.00 |  0.05 |      0.05 |    0.00 |     0.31 | C     |
| Semiconductors  | deeptech_silicon | NPV/option    |    7.50 |  24.00 |  60 |  0.10 |  0.00 |  0.00 |      0.00 |    0.00 |     0.00 | C     |
| Biotech         | biotech          | rNPV          |    4.00 |  13.00 |  50 |  0.03 |  0.00 |  0.00 |      0.00 |    0.00 |     0.00 | C     |
| Med Devices     | biotech          | rNPV          |    2.50 |  13.00 |  30 |  0.03 |  0.00 |  0.00 |      0.00 |    0.00 |     0.00 | C     |

Confidence grades:
- A: revenue-bearing, all three estimators agree within 2x
- B: revenue-bearing, estimators disagree by more than 2x (typical when burn-implied Estimator B pulls high against the empirical prior, since B is forward-looking implied run-rate not snapshot ARR)
- C: prior-dominated (pre-revenue), only Estimator C carries meaningful weight

Macro sanity check at the bottom: round-weighted mean ARR across all sectors is $0.53M, consistent with Carta's framing that AI App seeds at the high end carry $1M-$2M ARR while non-AI sectors cluster below $0.5M and pre-revenue sectors sit at zero.

The headline reordering: Analytics ($0.96M) and AI Applications ($0.77M) now lead, with AI Infra surprisingly LOWER at $0.54M despite the highest valuation in the dataset. This is the methodologically honest read: the AI Infra sector's $65M post-money is dominated by frontier model labs (Thinking Machines, Anthropic-style raises) priced on team and option value, not revenue. Tooling players (Modal, Together, Pinecone-tier) are the upper tail of that distribution but not the median.

## 8. Limitations

1. **Cohort medians, not per-company predictions.** A single company's ARR can deviate by an order of magnitude from its sector median. Top-decile AI seeds carry $5M-$10M ARR; bottom-decile $0.
2. **AI premium decay risk.** The 30-50% AI premium quantified by Carta in early 2025 may compress as the AI hype curve normalizes. M_R for ai_app and ai_infra are calibrated to early-2026 reality and may overstate ARR if multiples collapse.
3. **Seed timing skew.** A "seed" round in 2025-2026 increasingly behaves like a pre-Series-A: post-money of $20M is well above the $14M median Carta reported for Q4 2023. Companies are raising larger seeds with more revenue, biasing the dataset toward more mature seeds than the historical norm.
4. **Chart-eyeballed centroids.** Per-sector X and Y read off the published bubble map are accurate to about plus or minus $1M. The script flags this rather than pretending to higher precision.
5. **Sector boundaries blur.** Analytics sits inside AI Applications functionally; a company might be tagged ai_app, saas, or fintech depending on Carta's coding. Cross-sector double-counting is possible.
6. **Currency and geography.** Numbers are USD only, US startups only.

## 9. Source bibliography

Primary Carta and Peter Walker sources:
- carta.com/data (gated reports including State of Private Markets Q4 2025)
- carta.com/learn/resources/state-of-seed-2025/
- carta.com/data/state-of-pre-seed-q1-2025/
- carta.com/data/state-of-pre-seed-q2-2025/
- carta.com/data/state-of-pre-seed-q3-2025/
- carta.com/data/saas-industry-spotlight-Q3-2025/
- x.com/PeterJ_Walker/status/1903950230460244453 (AI vs non-AI seed pre-money)
- pmf.show/peter-walker-carta-insights-series-a-2025/

Multiples and benchmarks (valuation):
- highalpha.com/saas-benchmarks (2024 + 2025 SaaS Benchmarks)
- bvp.com/atlas/the-state-of-ai-2025
- bvp.com/atlas/the-cloud-100-benchmarks-report
- iconiq.com/growth/reports/2025-state-of-software (ICONIQ State of Software 2025)
- finrofca.com/news/ai-startup-valuations-q1-2025-edition
- finrofca.com/news/cybersecurity-valuation-mid-2025
- finrofca.com/news/biotech-revenue-multiples-2025
- aventis-advisors.com/saas-valuation-multiples/
- aventis-advisors.com/ai-valuation-multiples/
- aventis-advisors.com/marketplace-valuation-multiples/
- finerva.com/report/green-energy-renewables-2025-valuation-multiples/
- windsordrake.com/fintech-valuation-multiples/
- qubit.capital/blog/proptech-valuation-benchmarks
- learn.icanpitch.com/blog/seed-valuation-benchmarks-2025/
- metal.so/collections/us-saas-seed-round-benchmarks-2025-average-round-size-valuations-dilution
- pitchwise.se/blog/median-seed-round-size-by-industry-in-2026-data

Burn multiples and capital efficiency:
- learn.icanpitch.com/blog/burn-rate-benchmarks-by-industry-stage/
- cfoadvisors.com/blog/2025-burn-multiple-benchmarks_-how-series-a-saas-startups-can-prove-capital-efficiency
- saas-capital.com/blog-posts/spending-benchmarks-for-private-b2b-saas-companies/
- tomtunguz.com/burn-multiple-2023/
- saastr.com/the-great-rotation-how-ai-ml-crushed-traditional-saas-in-seed-investing-during-1h-2025-per-angellist/

Sector-specific:
- carta.com/data/industry-spotlight-consumer-q1-2025/ (Carta Consumer Industry Spotlight Q1 2025)
- gardinercolin.com/p/marketplace-startup-fundraising-2025
- outlierventures.io/article/how-to-raise-a-pre-seed-round-in-web3/
- qubit.capital/blog/financial-modelling-biotech-seed-round
- onhealthcare.tech/p/the-2025-playbook-for-early-stage
- albion.vc/app/uploads/2025/05/AlbionVC-x-Beauhurst-deeptech-future-of-compute-report_2025-FINAL.pdf
- ctvc.co/a-weak-11-3bn-start-to-2024-climate-tech/

Anchor deals (revenue and valuation triangulation):
- techcrunch.com/2025/07/15/mira-muratis-thinking-machines-lab-is-worth-12b-in-seed-round/
- techcrunch.com/2025/06/05/cursors-anysphere-nabs-9-9b-valuation-soars-past-500m-arr/
- techcrunch.com/2025/10/27/mercor-quintuples-valuation-to-10b-with-350m-series-c/
- humanoidsdaily.com/news/the-great-valuation-chasm-a-2025-guide-to-the-humanoid-robotics-capital-race (Figure $39B)
- news.crunchbase.com/robotics/physical-ai-custom-robot-builder-seed-funding-anvil/
- sacra.com/c/cursor/
- sacra.com/c/glean/
- sightlineclimate.com/research/40-5bn-and-8-uptick-as-power-demand-drives-25-investment
- chaincatcher.com/en/article/2234516 (RootData 2025 Web3 Annual Report)
- saastr.com/the-state-of-seed-today-10-key-learnings-from-cartas-latest-data/
- speedrun.substack.com/p/8-takeaways-from-cartas-state-of
