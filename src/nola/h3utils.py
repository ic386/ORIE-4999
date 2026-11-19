"""H3 indexing helpers (h3-py v4 API).

TODOs covered
- Assign H3 cells at resolutions 8 and 9 to all records -> assign_h3()
- Test sensitivity to resolution                          -> resolution_sensitivity()
"""

from __future__ import annotations

from collections import Counter

import h3
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def _cell(lat, lon, res):
    if lat is None or lon is None or np.isnan(lat) or np.isnan(lon):
        return None
    return h3.latlng_to_cell(float(lat), float(lon), res)


def assign_h3(df: pd.DataFrame, resolutions=(8, 9), lat="lat", lon="lon") -> pd.DataFrame:
    """Add h3_r{res} columns. Rows without coordinates get None."""
    la = pd.to_numeric(df[lat], errors="coerce").to_numpy(float)
    lo = pd.to_numeric(df[lon], errors="coerce").to_numpy(float)
    finest = max(resolutions)
    fine = [_cell(a, b, finest) for a, b in zip(la, lo)]
    df = df.copy()
    for r in resolutions:
        df["h3_r{}".format(r)] = fine if r == finest else [
            h3.cell_to_parent(c, r) if c else None for c in fine
        ]
    return df


def neighbors(cell: str, k: int = 1) -> set[str]:
    return set(h3.grid_disk(cell, k))


def hex_area_km2(res: int) -> float:
    return h3.average_hexagon_area(res, unit="km^2")


def cell_polygon(cell: str):
    from shapely.geometry import Polygon

    return Polygon([(lng, lat) for lat, lng in h3.cell_to_boundary(cell)])


def modal_lookup(df: pd.DataFrame, cell_col: str, value_col: str) -> pd.DataFrame:
    """Most common value (e.g. police district / zone) per hex -> hex<->NOPD crosswalk."""
    g = df.dropna(subset=[cell_col, value_col]).groupby(cell_col)[value_col]
    out = g.agg(lambda s: Counter(s).most_common(1)[0][0]).rename(value_col).to_frame()
    out["{}_purity".format(value_col)] = g.agg(lambda s: Counter(s).most_common(1)[0][1] / len(s))
    return out.reset_index()


def resolution_sensitivity(df: pd.DataFrame, lat="lat", lon="lon", resolutions=(7, 8, 9, 10),
                           min_count: int = 10) -> pd.DataFrame:
    """How do counts per hex behave as resolution changes?

    Reports cell counts, sparsity (share of cells with < min_count events), dispersion,
    and child_parent_spearman: rank correlation between each cell's count and its parent's
    count. Low values mean the coarser resolution hides real within-hex variation; very sparse
    cells mean the finer resolution is mostly noise. Pick the balance point.
    """
    d = assign_h3(df[[lat, lon]].dropna(), resolutions=tuple(resolutions), lat=lat, lon=lon)
    rows = []
    for r in sorted(resolutions):
        c = d["h3_r{}".format(r)].value_counts()
        rows.append({
            "res": r,
            "avg_hex_km2": round(hex_area_km2(r), 4),
            "n_cells": len(c),
            "median_per_cell": float(c.median()),
            "share_cells_lt_min": float((c < min_count).mean()),
            "var_to_mean": float(c.var() / c.mean()) if len(c) > 1 else np.nan,
        })
    out = pd.DataFrame(rows)
    rs = sorted(resolutions)
    rho = [np.nan]
    for parent, child in zip(rs, rs[1:]):
        child_counts = d["h3_r{}".format(child)].value_counts()
        parent_counts = d["h3_r{}".format(parent)].value_counts()
        par_of_child = parent_counts.reindex([h3.cell_to_parent(c, parent) for c in child_counts.index])
        ok = len(child_counts) > 2 and child_counts.nunique() > 1 and par_of_child.nunique() > 1
        rho.append(spearmanr(child_counts.to_numpy(), par_of_child.to_numpy()).statistic if ok else np.nan)
    out["child_parent_spearman"] = rho
    return out
