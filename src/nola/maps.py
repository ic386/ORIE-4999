"""Map v1: interactive H3 call-density map with filters (standalone HTML, deck.gl + MapLibre).

TODOs covered
- Prototype the map with sample hex data -> sample_points() + build_payload() + render()
- Map v1 with a call-density layer and filters (year, call category, time of day, district,
  count vs per-1,000-residents).

The HTML embeds pre-aggregated hex counts (never raw records), so it's safe to share.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .h3utils import assign_h3, modal_lookup

TEMPLATE = Path(__file__).parent / "templates" / "map_v1.html"


def tod_bucket(hour: pd.Series) -> pd.Series:
    b = config.params()["temporal"]["tod_buckets"]
    out = pd.Series(pd.NA, index=hour.index, dtype="string")
    for name, (lo, hi) in b.items():
        out[(hour >= lo) & (hour < hi)] = name
    return out


def sample_points(n: int = 60_000, seed: int = 7) -> pd.DataFrame:
    """Synthetic call records around real New Orleans neighborhoods - for building the UI only."""
    rng = np.random.default_rng(seed)
    centers = np.array([  # (lat, lon, weight)
        (29.958, -90.065, 4), (29.965, -90.040, 2), (29.930, -90.090, 2), (29.985, -90.030, 2),
        (29.990, -90.090, 1.5), (30.020, -89.970, 2), (30.040, -89.940, 1.5), (29.940, -90.030, 1),
        (29.920, -90.120, 1), (30.000, -90.060, 1.5), (29.960, -90.110, 1), (30.010, -90.010, 1)])
    w = centers[:, 2] / centers[:, 2].sum()
    idx = rng.choice(len(centers), size=n, p=w)
    lat = centers[idx, 0] + rng.normal(0, 0.012, n)
    lon = centers[idx, 1] + rng.normal(0, 0.015, n)
    cats = ["disturbance", "property", "traffic", "proactive", "violent", "alarm", "domestic",
            "medical_mental", "shots_fired", "admin", "other"]
    cp = np.array([22, 14, 13, 18, 6, 6, 5, 4, 2, 6, 4], float)
    df = pd.DataFrame({
        "lat": lat, "lon": lon,
        "year": rng.choice(config.years(), size=n),
        "category": rng.choice(cats, size=n, p=cp / cp.sum()),
        "hour": rng.choice(24, size=n, p=np.r_[np.full(6, 2), np.full(6, 3.5), np.full(6, 5), np.full(6, 6)] / 99),
    })
    df["police_district"] = pd.cut(df["lon"], [-91, -90.10, -90.075, -90.06, -90.045, -90.02, -89.99, -89.95, -89],
                                   labels=[6, 2, 1, 8, 5, 3, 4, 7]).astype(int)
    return df


def build_payload(df: pd.DataFrame, res: int = 8, pop: pd.DataFrame | None = None,
                  title: str = "New Orleans calls for service") -> dict:
    """Aggregate records -> compact arrays: hex x year x category x time-of-day counts."""
    cell = "h3_r{}".format(res)
    if cell not in df:
        df = assign_h3(df, resolutions=(res,))
    if "hour" not in df and "time_create" in df:
        df = df.assign(hour=pd.to_datetime(df["time_create"]).dt.hour)
    df = df.dropna(subset=[cell]).assign(tod=tod_bucket(df["hour"]))
    df["category"] = df["category"].fillna("other")
    g = df.groupby([cell, "year", "category", "tod"], observed=True).size().rename("n").reset_index()

    hexes = sorted(g[cell].unique())
    years = sorted(int(y) for y in g["year"].dropna().unique())
    cats = g.groupby("category")["n"].sum().sort_values(ascending=False).index.tolist()
    tods = list(config.params()["temporal"]["tod_buckets"])
    hi = {h: i for i, h in enumerate(hexes)}
    rows = np.column_stack([
        g[cell].map(hi), g["year"].map({y: i for i, y in enumerate(years)}),
        g["category"].map({c: i for i, c in enumerate(cats)}), g["tod"].map({t: i for i, t in enumerate(tods)}),
        g["n"],
    ]).astype(int).tolist()

    district = [None] * len(hexes)
    if "police_district" in df:
        lk = modal_lookup(df, cell, "police_district").set_index(cell)["police_district"]
        district = [None if pd.isna(lk.get(h)) else int(lk.get(h)) for h in hexes]
    popv = [None] * len(hexes)
    if pop is not None:
        pm = pop.set_index(cell)["pop"]
        popv = [None if pd.isna(pm.get(h)) else round(float(pm.get(h)), 1) for h in hexes]
    return {"title": title, "res": res, "hexes": hexes, "district": district, "pop": popv,
            "years": years, "cats": cats, "tods": tods, "rows": rows,
            "min_pop": config.params()["population"]["min_pop_per_hex"]}


def render(payload: dict, out_path: Path) -> Path:
    html = TEMPLATE.read_text().replace("/*__DATA__*/null", json.dumps(payload, separators=(",", ":")))
    html = html.replace("__TITLE__", payload["title"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html)
    return out_path
