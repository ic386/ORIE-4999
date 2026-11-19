# NOLA Reporting Funnel

ORIE 4999 · New Orleans calls for service → confirmed incidents → filed police reports, mapped on H3 hexagons and explained with 311 neighborhood conditions. Research question and methods: [`docs/research_question.md`](docs/research_question.md). Task tracker: [`docs/TODO.md`](docs/TODO.md).

## Quick start (VS Code)

```bash
cd ~/Desktop/nola-reporting-funnel
code .                 # open in VS Code; accept the recommended extensions
make setup             # .venv + dependencies + editable install; creates .env
# add your SOCRATA_APP_TOKEN to .env (free, raises rate limits)
make sample            # 20k rows per table - fast dev pull
make all               # dictionary → clean → join → qa → h3 → calendar → tables → population → temporal → map → dedup
make test
```

Full data: `make fetch` (~15 tables, a few million rows; takes a while), then `make all`.
VS Code: **Run and Debug → "Pipeline: pick a step"** runs any single step; `notebooks/*.py` are `# %%` cell files you run with "Run Cell".

Without Make: `python -m nola.pipeline <step>` (steps listed in `python -m nola.pipeline -h`).

## Data (never committed)

All data is pulled from the city's Socrata API into `data/` (git-ignored, along with every `.csv`/`.parquet`). Dataset IDs live in [`config/datasets.yaml`](config/datasets.yaml):

| Source | Tables | Join / key facts |
|---|---|---|
| Calls for Service (OPCD/NOPD) | one per year, 2019–2025 | `nopd_item` = `A0000124`; location point (some 0,0) + State Plane `mapx/mapy` (EPSG:3452) |
| Electronic Police Reports | one per year, 2019–2025 | `item_number` = `A-00001-24` → normalized to `A0000124`; one row per charge/person, collapsed to one per item |
| 311 OPCD Calls | `2jgv-pqrq` | ~37K labeled duplicates; created dates reliable from 2019-01-03; 0,0 rows dropped |

## Layout

```
config/            datasets.yaml · params.yaml · events.yaml · funnel.yaml · call_type_crosswalk.csv
src/nola/
  socrata.py       paginated API pulls → data/raw/*.parquet (+ metadata JSON = data dictionaries)
  dictionary.py    data-dictionary review, schema drift, code stability by year
  clean.py         CFS / EPR / 311 cleaning, item-number normalization, coordinates, beats → zones
  join.py          CFS–EPR join + match-rate and disposition checks
  h3utils.py       H3 res 8/9 indexing, hex ↔ NOPD district/zone lookup, resolution sensitivity
  events.py        holiday/event calendar (Mardi Gras = Easter − 47, parade offsets, confounders)
  funnel.py        stage definitions, call-type crosswalk, empirical-Bayes rates, analysis tables
  population.py    ACS block groups → hexes (areal weighting), per-capita rates
  temporal.py      hour×weekday heatmaps, holiday lift (ratio + NB regression), MSTL seasonality
  maps.py          Map v1 (deck.gl H3 layer + filters) → outputs/maps/*.html
  dedup.py         311 duplicate rule (same reason, k-ring, time window) with precision/recall
  pipeline.py      CLI that chains the steps
notebooks/         01 dictionary review · 02 feasibility checks · 03 temporal EDA
tests/             synthetic-data unit tests + end-to-end smoke test
data/  outputs/    created on first run; git-ignored
```

## Decisions still open (confirm with Ryan)

- Funnel stage for `NAT` (necessary action taken) and `REF` (referred) — currently NAT = confirmed, REF = other (`config/funnel.yaml`).
- Call-type crosswalk categories — review `outputs/tables/qa_call_type_crosswalk_review.csv`.
- Beat → zone/subzone parsing assumes `7L03`-style beats; check `beat_parsed` in `qa_cfs_completeness_by_year.csv`.
- Event dates from rules (Jazz Fest, Essence, Decadence) are approximate — verify each year in `config/events.yaml`.

## Git

```bash
git init && git add . && git commit -m "Project scaffold"
```
`.gitignore` already excludes `data/`, `outputs/`, `.env`, and all CSV/parquet files except `config/*.csv`.
