"""Data-dictionary review and code-stability checks.

TODOs covered
- Review the data dictionaries            -> data_dictionary(), schema_drift()
- Check code stability by year            -> code_stability()
"""

from __future__ import annotations

import json

import pandas as pd

from . import config


def _year_lookup() -> dict[str, tuple[str, str]]:
    """dataset_id -> (source, year)."""
    out = {}
    for source in ("calls_for_service", "epr", "service_311"):
        for key, did in config.datasets()[source].items():
            out[did] = (source, str(key))
    return out


def data_dictionary() -> pd.DataFrame:
    """One row per (dataset, column) from the saved Socrata view JSON."""
    rows = []
    lookup = _year_lookup()
    for path in sorted(config.METADATA.glob("*.json")):
        meta = json.loads(path.read_text())
        source, year = lookup.get(meta.get("id"), ("unknown", ""))
        for col in meta.get("columns", []):
            cached = col.get("cachedContents", {}) or {}
            top = cached.get("top", []) or []
            rows.append(
                {
                    "source": source,
                    "year": year,
                    "dataset_id": meta.get("id"),
                    "dataset_name": meta.get("name"),
                    "field_name": col.get("fieldName"),
                    "display_name": col.get("name"),
                    "data_type": col.get("dataTypeName"),
                    "description": (col.get("description") or "").strip(),
                    "non_null": cached.get("non_null"),
                    "null": cached.get("null"),
                    "cardinality": cached.get("cardinality"),
                    "top_values": "; ".join("{} ({})".format(t.get('item'), t.get('count')) for t in top[:10]),
                }
            )
    return pd.DataFrame(rows)


def schema_drift(dd: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Per source: field x year matrix of data types ('' = column missing that year)."""
    out = {}
    for source, g in dd.groupby("source"):
        out[source] = (
            g.pivot_table(index="field_name", columns="year", values="data_type", aggfunc="first")
            .fillna("")
            .sort_index()
        )
    return out


def code_stability(df: pd.DataFrame, code_col: str, year_col: str = "year") -> pd.DataFrame:
    """Share of rows per code per year + flags for codes that appear/disappear.

    Use on CFS disposition / type_text / priority and EPR disposition / signal_type
    before comparing long periods.
    """
    ct = pd.crosstab(df[code_col].fillna("<NA>"), df[year_col])
    share = ct.div(ct.sum(axis=0), axis=1).round(4)
    present = ct > 0
    share["years_present"] = present.sum(axis=1)
    share["first_year"] = present.idxmax(axis=1)
    share["last_year"] = present.iloc[:, ::-1].idxmax(axis=1)
    share["unstable"] = share["years_present"] < ct.shape[1]
    share["total"] = ct.sum(axis=1)
    return share.sort_values("total", ascending=False)


def year_over_year_jaccard(df: pd.DataFrame, code_col: str, year_col: str = "year") -> pd.DataFrame:
    """Jaccard overlap of the code set between consecutive years (1.0 = identical vocab)."""
    sets = df.groupby(year_col)[code_col].apply(lambda s: set(s.dropna().unique()))
    yrs = sorted(sets.index)
    rows = []
    for a, b in zip(yrs, yrs[1:]):
        inter, union = sets[a] & sets[b], sets[a] | sets[b]
        rows.append(
            {
                "from": a,
                "to": b,
                "jaccard": len(inter) / len(union) if union else 1.0,
                "added": sorted(sets[b] - sets[a])[:25],
                "dropped": sorted(sets[a] - sets[b])[:25],
            }
        )
    return pd.DataFrame(rows)


def write_dictionary_outputs() -> None:
    dd = data_dictionary()
    if dd.empty:
        raise FileNotFoundError("No metadata found; run `python -m nola.pipeline fetch` first")
    dd.to_csv(config.TABLES / "data_dictionary.csv", index=False)
    for source, m in schema_drift(dd).items():
        m.to_csv(config.TABLES / "schema_drift_{}.csv".format(source))
    print("[dictionary] {} columns across {} tables -> outputs/tables/".format(len(dd), dd.dataset_id.nunique()))
