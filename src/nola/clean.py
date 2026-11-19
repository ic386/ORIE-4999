"""Cleaning for Calls for Service (CFS), Electronic Police Reports (EPR) and 311.

Key facts (from the data dictionaries, Nov 2026):
- CFS fields are stable 2019-2025: nopd_item, type_, typetext, priority, initialtype*, mapx/mapy
  (Louisiana South State Plane, ftUS), timecreate/dispatch/arrive/closed, disposition(text),
  selfinitiated (Y/N), beat, block_address, zip, policedistrict, location (point; some 0,0).
- CFS item numbers look like 'A0000124'; EPR item numbers like 'A-00001-24'. normalize_item_id()
  maps both to 'A0000124' so the join works.
- EPR has one row per charge/offender/victim -> collapse to one row per item for the funnel.
  EPR has no coordinates (block address text only), so its location comes from the CFS side.
- 311 Date Created is unreliable before 2019-01-03; ~41K rows have 0,0 coordinates.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from pyproj import Transformer

from . import config

ITEM_RE = re.compile(r"^[A-Z]\d{7}$")


# --------------------------------------------------------------------------- helpers
def normalize_item_id(s: pd.Series) -> pd.Series:
    """'A-00001-24' / 'a0000124 ' -> 'A0000124'."""
    return s.astype("string").str.upper().str.replace(r"[^A-Z0-9]", "", regex=True)


def in_bbox(lat: pd.Series, lon: pd.Series) -> pd.Series:
    b = config.params()["geo"]["bbox"]
    return lat.between(b["min_lat"], b["max_lat"]) & lon.between(b["min_lon"], b["max_lon"])


def state_plane_to_lonlat(x: pd.Series, y: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    crs = config.params()["geo"]["state_plane_crs"]
    t = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    lon, lat = t.transform(pd.to_numeric(x, errors="coerce").to_numpy(float),
                           pd.to_numeric(y, errors="coerce").to_numpy(float))
    return np.asarray(lon), np.asarray(lat)


def _ts(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce", format="mixed")


def _col(df: pd.DataFrame, *names: str) -> pd.Series:
    """First column that exists (schemas drift slightly across years)."""
    for n in names:
        if n in df.columns:
            return df[n]
    return pd.Series(pd.NA, index=df.index)


# --------------------------------------------------------------------------- CFS
def clean_cfs(raw: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame(index=raw.index)
    df["item_id"] = normalize_item_id(_col(raw, "nopd_item"))
    df["item_id_valid"] = df["item_id"].str.match(ITEM_RE.pattern).fillna(False)
    df["type_code"] = _col(raw, "type_", "type")
    df["type_text"] = _col(raw, "typetext").str.strip().str.upper()
    df["priority"] = _col(raw, "priority")
    df["initial_type_code"] = _col(raw, "initialtype")
    df["initial_type_text"] = _col(raw, "initialtypetext").str.strip().str.upper()
    df["initial_priority"] = _col(raw, "initialpriority")
    for out, src in [("time_create", "timecreate"), ("time_dispatch", "timedispatch"),
                     ("time_arrive", "timearrive"), ("time_closed", "timeclosed")]:
        df[out] = _ts(_col(raw, src))
    df["disposition"] = _col(raw, "disposition").str.strip().str.upper()
    df["disposition_text"] = _col(raw, "dispositiontext").str.strip().str.upper()
    df["self_initiated"] = _col(raw, "selfinitiated").astype("string").str.upper().eq("Y")
    df["beat"] = _col(raw, "beat").astype("string").str.strip().str.upper()
    df["block_address"] = _col(raw, "block_address")
    df["zip"] = _col(raw, "zip")
    df["police_district"] = pd.to_numeric(_col(raw, "policedistrict"), errors="coerce").astype("Int64")
    for c in [c for c in raw.columns if c.startswith("region_")]:
        df[c] = raw[c]
    df["source_year"] = _col(raw, "source_year")

    # District / zone / subzone from beat (NOPD thinks in these units - Ryan)
    parts = df["beat"].str.extract(config.params()["cfs"]["beat_regex"])
    df["beat_district"] = pd.to_numeric(parts["district"], errors="coerce").astype("Int64")
    df["zone"] = parts["district"] + parts["zone"]
    df["subzone"] = df["beat"].where(parts["subzone"].notna())

    # Coordinates: Socrata point first, fall back to State Plane mapx/mapy
    lon = pd.to_numeric(_col(raw, "location_lon"), errors="coerce")
    lat = pd.to_numeric(_col(raw, "location_lat"), errors="coerce")
    ok = in_bbox(lat, lon)
    sp_lon, sp_lat = state_plane_to_lonlat(_col(raw, "mapx"), _col(raw, "mapy"))
    sp_lon, sp_lat = pd.Series(sp_lon, index=raw.index), pd.Series(sp_lat, index=raw.index)
    sp_ok = in_bbox(sp_lat, sp_lon)
    df["lon"] = lon.where(ok, sp_lon.where(sp_ok))
    df["lat"] = lat.where(ok, sp_lat.where(sp_ok))
    df["coord_source"] = np.select([ok, sp_ok], ["point", "state_plane"], default="none")

    # Intervals (minutes)
    m = lambda a, b: (df[b] - df[a]).dt.total_seconds() / 60  # noqa: E731
    df["dispatch_delay_min"] = m("time_create", "time_dispatch")
    df["travel_min"] = m("time_dispatch", "time_arrive")
    df["response_min"] = m("time_create", "time_arrive")
    for c in ("dispatch_delay_min", "travel_min", "response_min"):
        df.loc[df[c] < 0, c] = np.nan

    df["year"] = df["time_create"].dt.year.astype("Int64")
    df["hour"] = df["time_create"].dt.hour.astype("Int64")
    df["dow"] = df["time_create"].dt.dayofweek.astype("Int64")
    df["date"] = df["time_create"].dt.normalize()

    before = len(df)
    df = df[df["item_id"].notna()].drop_duplicates("item_id", keep="last")
    print("[clean] CFS: {:,} -> {:,} rows; "
          "geocoded {:.1%}".format(before, len(df), df['coord_source'].ne('none').mean()))
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------- EPR
def clean_epr_rows(raw: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame(index=raw.index)
    df["item_id"] = normalize_item_id(_col(raw, "item_number"))
    df["district"] = _col(raw, "district")
    df["block_address"] = _col(raw, "location")
    df["disposition"] = _col(raw, "disposition").str.strip().str.upper()
    df["signal_type"] = _col(raw, "signal_type").astype("string").str.strip().str.upper()
    df["signal_description"] = _col(raw, "signal_description").str.strip().str.upper()
    df["occurred_dt"] = _ts(_col(raw, "occurred_date_time"))
    df["charge_code"] = _col(raw, "charge_code")
    df["charge_description"] = _col(raw, "charge_description")
    df["report_type"] = _col(raw, "report_type").str.strip()
    df["victim_fatal_status"] = _col(raw, "victim_fatal_status")
    df["source_year"] = _col(raw, "source_year")
    return df[df["item_id"].notna()].reset_index(drop=True)


def collapse_epr(rows: pd.DataFrame) -> pd.DataFrame:
    """One row per item number. Incident-report fields win over supplementals."""
    rows = rows.assign(_is_inc=rows["report_type"].eq("Incident Report"))
    rows = rows.sort_values(["item_id", "_is_inc"], ascending=[True, False])
    first = rows.groupby("item_id", as_index=False).first()
    agg = rows.groupby("item_id").agg(
        epr_rows=("item_id", "size"),
        has_incident_report=("_is_inc", "any"),
        has_supplemental=("report_type", lambda s: s.eq("Supplemental Report").any()),
        n_charges=("charge_code", "nunique"),
        any_fatal=("victim_fatal_status", lambda s: s.astype("string").str.contains("Fatal", case=False, na=False).any()),
        epr_years=("source_year", lambda s: ",".join(sorted({str(v) for v in s.dropna()}))),
    ).reset_index()
    out = first[["item_id", "district", "block_address", "disposition", "signal_type",
                 "signal_description", "occurred_dt"]].merge(agg, on="item_id")
    return out.rename(columns={"disposition": "epr_disposition", "district": "epr_district",
                               "block_address": "epr_block_address"})


# --------------------------------------------------------------------------- 311
def clean_311(raw: pd.DataFrame) -> pd.DataFrame:
    p = config.params()["service_311"]
    df = pd.DataFrame(index=raw.index)
    df["service_request"] = _col(raw, "service_request")
    df["request_type"] = _col(raw, "request_type").str.strip()
    df["request_reason"] = _col(raw, "request_reason").str.strip()
    df["date_created"] = _ts(_col(raw, "date_created"))
    df["date_closed"] = _ts(_col(raw, "case_close_date"))
    df["request_status"] = _col(raw, "request_status")
    df["status"] = _col(raw, "status").str.strip()
    df["responsible_agency"] = _col(raw, "responsible_agency")
    df["council_district"] = _col(raw, "address_councildis")
    df["lat"] = pd.to_numeric(_col(raw, "latitude", "geocoded_column_lat"), errors="coerce")
    df["lon"] = pd.to_numeric(_col(raw, "longitude", "geocoded_column_lon"), errors="coerce")

    n0 = len(df)
    df = df[in_bbox(df["lat"], df["lon"])]  # drops 0,0 and stray geocodes
    print("[clean] 311: dropped {:,} rows with 0,0 / out-of-parish coordinates".format(n0 - len(df)))

    df["created_reliable"] = df["date_created"] >= pd.Timestamp(p["reliable_created_from"])
    df["is_duplicate"] = df["status"].isin(p["duplicate_status"])
    df["is_invalid"] = df["status"].isin(p["invalid_status"])
    text = (df["request_type"].fillna("") + " | " + df["request_reason"].fillna(""))
    for name, pattern in p["conditions"].items():
        df["cond_{}".format(name)] = text.str.contains(pattern, case=False, regex=True)
    df["resolution_days"] = (df["date_closed"] - df["date_created"]).dt.total_seconds() / 86400
    df.loc[~df["created_reliable"] | (df["resolution_days"] < 0), "resolution_days"] = np.nan
    df["year"] = df["date_created"].dt.year.astype("Int64")
    return df.reset_index(drop=True)
