"""Population denominators: ACS block groups -> H3 hexes by areal weighting.

TODOs covered
- Start the ACS population interpolation -> fetch_acs_blockgroups(), interpolate_to_hex()
- Compute per-capita hex counts           -> per_capita()

Block groups (finer than tracts) per Ryan. Areal weighting assumes population is spread
evenly within a block group; swap in maup (MGGG) or a gridded raster (NASA SEDAC GPW,
~1 km) later for a better prior - the interface (hex -> pop) stays the same.
"""

from __future__ import annotations

import os

import h3
import numpy as np
import pandas as pd
import requests

from . import config
from .h3utils import cell_polygon

CENSUS = "https://api.census.gov/data/{year}/acs/acs5"
TIGER = "https://www2.census.gov/geo/tiger/TIGER{year}/BG/tl_{year}_{state}_bg.zip"


def fetch_acs_blockgroups(year=None, variable=None, state=None, county=None) -> pd.DataFrame:
    p = config.params()["population"]
    year, variable = year or p["acs_year"], variable or p["acs_variable"]
    state, county = state or p["state_fips"], county or p["county_fips"]
    q = {"get": "NAME,{}".format(variable), "for": "block group:*", "in": "state:{} county:{} tract:*".format(state, county)}
    if os.getenv("CENSUS_API_KEY"):
        q["key"] = os.getenv("CENSUS_API_KEY")
    r = requests.get(CENSUS.format(year=year), params=q, timeout=120)
    r.raise_for_status()
    rows = r.json()
    df = pd.DataFrame(rows[1:], columns=rows[0])
    df["GEOID"] = df["state"] + df["county"] + df["tract"] + df["block group"]
    df["pop"] = pd.to_numeric(df[variable], errors="coerce")
    return df[["GEOID", "pop"]]


def load_blockgroup_geoms(year=None, state=None, county=None):
    import geopandas as gpd

    p = config.params()["population"]
    year, state, county = year or p["acs_year"], state or p["state_fips"], county or p["county_fips"]
    cache = config.EXTERNAL / "tl_{}_{}_bg.zip".format(year, state)
    if not cache.exists():
        r = requests.get(TIGER.format(year=year, state=state), timeout=300)
        r.raise_for_status()
        cache.write_bytes(r.content)
    g = gpd.read_file(cache)
    return g[g["COUNTYFP"] == county][["GEOID", "geometry"]]


def interpolate_to_hex(bg, res: int, pop_col: str = "pop"):
    """Areal-weighted hex population. `bg` is a GeoDataFrame with pop + geometry (any CRS)."""
    import geopandas as gpd

    area_crs = config.params()["geo"]["area_crs"]
    bg = bg.to_crs("EPSG:4326")
    cells = set()
    for geom in bg.geometry:
        polys = getattr(geom, "geoms", [geom])
        for poly in polys:
            cells |= set(h3.geo_to_cells(poly.buffer(0.003).__geo_interface__, res))
    hexes = gpd.GeoDataFrame({"h3": sorted(cells)}, geometry=[cell_polygon(c) for c in sorted(cells)],
                             crs="EPSG:4326").to_crs(area_crs)
    bgp = bg.to_crs(area_crs)
    bgp["bg_area"] = bgp.geometry.area
    inter = gpd.overlay(hexes, bgp[["GEOID", pop_col, "bg_area", "geometry"]], how="intersection")
    inter["w"] = inter.geometry.area / inter["bg_area"]
    inter["pop_part"] = inter[pop_col] * inter["w"]
    out = inter.groupby("h3", as_index=False)["pop_part"].sum().rename(columns={"pop_part": "pop"})
    return out.rename(columns={"h3": "h3_r{}".format(res)})


def build_hex_population(resolutions=(8, 9)) -> dict[int, pd.DataFrame]:
    acs = fetch_acs_blockgroups()
    geoms = load_blockgroup_geoms()
    bg = geoms.merge(acs, on="GEOID", how="left")
    return {r: interpolate_to_hex(bg, r) for r in resolutions}


def per_capita(counts: pd.DataFrame, pop: pd.DataFrame, cell_col: str, count_cols: list[str],
               per: int = 1000, min_pop: int | None = None) -> pd.DataFrame:
    """Rates per `per` residents. Hexes below min_pop get NaN (tourist/industrial areas)."""
    min_pop = config.params()["population"]["min_pop_per_hex"] if min_pop is None else min_pop
    out = counts.merge(pop, on=cell_col, how="left")
    denom = out["pop"].where(out["pop"] >= min_pop)
    for c in count_cols:
        out["{}_per_{}".format(c, per)] = out[c] / denom * per
    out["low_pop_hex"] = denom.isna()
    return out.replace([np.inf, -np.inf], np.nan)
