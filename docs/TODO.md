# TODO tracker

Roles from the project plan: **A** data engineer · **B** geospatial/H3 · **C** statistics · **D** visualization.
Each item lists the command that runs it and where the code lives.

## Data review & feasibility
- [ ] Review the data dictionaries — `make dictionary` → `outputs/tables/data_dictionary.csv`, `schema_drift_*.csv`; `notebooks/01_data_dictionary_review.py` (A)
- [ ] Run the CFS–EPR match rate and disposition-code checks — `make join qa` → `match_*.csv`, `match_disposition_vs_epr.csv`; `notebooks/02_feasibility_checks.py` (A, C)

## Cleaning & join
- [ ] Clean 311 and Calls for Service — `make clean-data` · `src/nola/clean.py` (A)
- [ ] Build the CFS–EPR join — `make join` · `src/nola/join.py` → `data/processed/cfs_epr.parquet` (A)

## H3 & population
- [ ] Assign H3 cells at resolutions 8 and 9 to all records — `make h3` · `src/nola/h3utils.py` (B)
- [ ] Start the ACS population interpolation — `make population` · `src/nola/population.py` (block groups → hexes, areal weighting; maup/NASA grid later) (B)

## Definitions
- [ ] Check code stability by year — `make qa` → `qa_cfs_*_by_year.csv`, `qa_cfs_*_jaccard.csv`, `qa_epr_*_by_year.csv` (C)
- [ ] Finalize the funnel-stage definitions and call-type crosswalk — edit `config/funnel.yaml` + `config/call_type_crosswalk.csv` after reviewing `qa_call_type_crosswalk_review.csv`; confirm NAT/REF with Ryan (C)

## Calendar & map prototype
- [ ] Build the holiday and event calendar — `make calendar` · `config/events.yaml`, `src/nola/events.py`; add WWOZ second lines to `data/external/second_lines.csv` (C)
- [ ] Prototype the map with sample hex data — `make sample-map` → `outputs/maps/map_v1_sample.html` (D)

## Analysis tables & reproducibility
- [ ] Build analysis tables (hex × call type × funnel stage) — `make tables` → `data/processed/funnel_hex_r{8,9}_*.parquet`, `outputs/tables/funnel_by_{police_district,zone,subzone}*.csv` (A, C)
- [ ] Make the pipeline reproducible — `make setup && make fetch && make all`; tests in `tests/` (`make test`) (A)

## Per-capita & resolution
- [ ] Compute per-capita hex counts — `make population` → `outputs/tables/hex_r{8,9}_per_capita.csv` (B)
- [ ] Test sensitivity to resolution — `make h3` → `outputs/tables/h3_resolution_sensitivity.csv` (B)

## Temporal
- [ ] Hour × weekday heatmaps — `make temporal` → `outputs/figures/heatmap_*.png` (C)
- [ ] Holiday-lift regression — `holiday_lift_ratio.csv`, `holiday_lift_regression_irr.csv` (C)
- [ ] Seasonality decomposition — `seasonality_components.csv`, `seasonality_decomposition.png` (C)

## Map v1
- [ ] Map v1 with a call-density layer and filters — `make map` → `outputs/maps/map_v1.html` (year, category, time of day, district, count vs per-1,000) (D)

## Later (in code, not yet scheduled)
- Dedup prototype — `make dedup` → `dedup_311_rule_grid.csv`
- Under-reporting test (shots fired vs confirmed shootings; ShotSpotter benchmark)
- 311-conditions regression, Moran's I, Gi* (`esda`/`libpysal` already in requirements)
- District scorecards
