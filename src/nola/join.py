"""CFS <-> EPR join on normalized item number, plus the match-rate / disposition checks.

TODOs covered
- Run the CFS-EPR match rate and disposition-code checks -> match_rate_report()
- Build the CFS-EPR join                                 -> join_cfs_epr()
"""

from __future__ import annotations

import pandas as pd


def join_cfs_epr(cfs: pd.DataFrame, epr_items: pd.DataFrame) -> pd.DataFrame:
    out = cfs.merge(epr_items, on="item_id", how="left", indicator="_m")
    out["has_epr"] = out["_m"].eq("both")
    out["has_incident_report"] = out["has_incident_report"].fillna(False).astype(bool)
    return out.drop(columns="_m")


def _rate(df: pd.DataFrame, by: str | list[str], flag: str = "has_epr") -> pd.DataFrame:
    g = df.groupby(by, dropna=False)[flag].agg(n="size", matched="sum")
    g["match_rate"] = (g["matched"] / g["n"]).round(4)
    return g.sort_values("n", ascending=False)


def match_rate_report(joined: pd.DataFrame, epr_items: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Feasibility tables. Read these before trusting any funnel number."""
    resident = joined[~joined["self_initiated"]]
    rep = {
        "cfs_overall": pd.DataFrame(
            {
                "n_cfs": [len(joined)],
                "share_with_epr": [joined["has_epr"].mean()],
                "share_with_incident_report": [joined["has_incident_report"].mean()],
                "resident_share_with_epr": [resident["has_epr"].mean()],
            }
        ),
        "cfs_by_year": _rate(joined, "year"),
        "cfs_by_disposition": _rate(joined, ["disposition", "disposition_text"]),
        "cfs_by_type": _rate(joined, "type_text"),
        "cfs_by_district": _rate(joined, "police_district"),
        "cfs_by_self_initiated": _rate(joined, "self_initiated"),
    }
    # Disposition consistency: RTF ("report to follow") should almost always have an EPR;
    # GOA/UNF should rarely have one.
    rep["disposition_vs_epr"] = pd.crosstab(
        joined["disposition"].fillna("<NA>"), joined["has_epr"], normalize="index"
    ).round(4)
    # Reverse direction: how many reports have no call record?
    cfs_ids = set(joined["item_id"])
    e = epr_items.assign(in_cfs=epr_items["item_id"].isin(cfs_ids))
    e["epr_year"] = e["item_id"].str[-2:].radd("20")
    rep["epr_in_cfs_by_year"] = _rate(e, "epr_year", flag="in_cfs")
    return rep
