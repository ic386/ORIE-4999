# Research question

**Where in New Orleans do the rates at which calls for service become confirmed incidents, and confirmed incidents become filed police reports, diverge from the citywide pattern, and do neighborhood conditions logged through 311 (streetlight outages, blight, illegal dumping) help explain those differences?**

This builds on Ryan's point that the data doesn't capture everything. We can't see calls that never happen, but we can measure how the funnel behaves in each area. Unusually low call-to-report ratios, or serious incidents with few associated calls, are indirect signs of under-reporting that each district can act on.

## Supporting questions

- **Temporal:** How do call volume and funnel ratios shift by hour, day of week, and holidays or major events (Mardi Gras, Jazz Fest, Sugar Bowl, July 4, NYE, Halloween)?
- **Dedup (for Comms):** Can a spatiotemporal rule built on H3 cells reliably flag likely duplicate calls?

## Data

| Dataset | Socrata IDs | Notes |
|---|---|---|
| Calls for Service 2019–2025 | `config/datasets.yaml` | Incident-level, block-masked; `nopd_item` like `A0000124` |
| Electronic Police Reports 2019–2025 | `config/datasets.yaml` | One row per charge/offender/victim; `item_number` like `A-00001-24`; no coordinates |
| 311 OPCD Calls (2012–present) | `2jgv-pqrq` | ~1.04M rows; ~37K "Duplicate Request"; Date Created unreliable before 2019; ~41K at 0,0 |

## Caveats

- Absence of calls is inferred, not observed.
- CFS is incident-level: several 911 callers about one event are usually folded into one item number. True call-level dedup needs OPCD's internal call logs; public data can prototype the method but not fully solve Comms' problem.
- Location is masked to the block, which limits honest spatial resolution.
- Disposition and call-type codes can change across years; check before comparing long periods.
- NOPD staffing and policy shifts affect response and reporting independently of neighborhood conditions.

## Methods (8 steps, H3 as the core spatial unit)

1. **Data prep** – Socrata pulls, CFS–EPR join on item number, State Plane (LA South, ft) → lat/long, holiday/event calendar with Mardi Gras computed per year and hurricane confounders.
2. **H3 indexing** – res 8 (~0.7 km²) for public maps, res 9 (~0.1 km²) for analysis; hexes crosswalked to NOPD districts/zones/subzones.
3. **Descriptive** – hour × weekday heatmaps; holiday lift vs same-weekday baseline; per-capita normalization (ACS block groups → hexes).
4. **Funnel metrics per hex** – call→report conversion by call type; GOA/unfounded share; self-initiated share; empirical-Bayes smoothing.
5. **Under-reporting test** – confirmed shootings vs nearby shots-fired calls (ShotSpotter as Ryan's benchmark); repeat for burglary.
6. **Explaining gaps with 311** – per-hex streetlight/blight/dumping density; negative binomial regression; Moran's I and Getis-Ord Gi* on H3 rings; optional lag analysis.
7. **Dedup prototype** – same reason, same/adjacent H3 cell, time window; precision/recall on labeled 311 duplicates.
8. **Outputs** – interactive H3 map, per-district scorecards, tabular breakdowns.
