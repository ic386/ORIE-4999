"""Pull data.nola.gov tables through the Socrata SODA API into parquet.

Nothing here is committed to git: raw pulls land in data/raw/ (git-ignored).
Set SOCRATA_APP_TOKEN in .env to avoid throttling.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

from . import config

load_dotenv(config.ROOT / ".env")

PAGE_SIZE = 50_000


def _session() -> requests.Session:
    s = requests.Session()
    token = os.getenv("SOCRATA_APP_TOKEN")
    if token:
        s.headers["X-App-Token"] = token
    return s


def _get(session: requests.Session, url: str, params: dict | None = None, retries: int = 5):
    for attempt in range(retries):
        r = session.get(url, params=params, timeout=180)
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(2**attempt)
            continue
        r.raise_for_status()
        return r.json()
    r.raise_for_status()


def fetch_metadata(dataset_id: str, domain: str | None = None, out_dir: Path | None = None) -> dict:
    """Save the view JSON (columns, descriptions, cached top values) - the data dictionary."""
    domain = domain or config.datasets()["domain"]
    out_dir = out_dir or config.METADATA
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = _get(_session(), "https://{}/api/views/{}.json".format(domain, dataset_id))
    (out_dir / "{}.json".format(dataset_id)).write_text(json.dumps(meta, indent=2))
    return meta


def _flatten_point(value) -> tuple[float | None, float | None]:
    """Socrata 'point' -> {'type':'Point','coordinates':[lon,lat]}; 'location' -> {'latitude','longitude',...}."""
    if not isinstance(value, dict):
        return None, None
    if "coordinates" in value:
        lon, lat = value["coordinates"][:2]
        return float(lon), float(lat)
    if "latitude" in value and "longitude" in value:
        try:
            return float(value["longitude"]), float(value["latitude"])
        except (TypeError, ValueError):
            return None, None
    return None, None


def _tidy(df: pd.DataFrame) -> pd.DataFrame:
    regions = config.datasets().get("computed_regions", {})
    df = df.rename(columns=regions)
    for col in [c for c in ("location", "geocoded_column") if c in df.columns]:
        pts = df[col].map(_flatten_point)
        df["{}_lon".format(col)] = pts.map(lambda p: p[0])
        df["{}_lat".format(col)] = pts.map(lambda p: p[1])
        df = df.drop(columns=col)
    # Anything still nested -> JSON string so parquet is happy
    for col in df.columns:
        if df[col].map(lambda v: isinstance(v, (dict, list))).any():
            df[col] = df[col].map(lambda v: json.dumps(v) if isinstance(v, (dict, list)) else v)
    return df


def fetch_dataset(
    dataset_id: str,
    out_path: Path,
    select: list[str] | None = None,
    where: str | None = None,
    max_rows: int | None = None,
    page_size: int = PAGE_SIZE,
) -> pd.DataFrame:
    """Page through /resource/<id>.json and write one parquet file."""
    domain = config.datasets()["domain"]
    url = "https://{}/resource/{}.json".format(domain, dataset_id)
    s = _session()
    parts, offset = [], 0
    while True:
        limit = page_size if max_rows is None else min(page_size, max_rows - offset)
        if limit <= 0:
            break
        q = {"$limit": limit, "$offset": offset, "$order": ":id"}
        if select:
            q["$select"] = ",".join(select)
        if where:
            q["$where"] = where
        batch = _get(s, url, q)
        if not batch:
            break
        parts.append(pd.DataFrame.from_records(batch))
        offset += len(batch)
        print("  {}: {:,} rows".format(dataset_id, offset), end="\r")
        if len(batch) < limit:
            break
    print()
    df = _tidy(pd.concat(parts, ignore_index=True)) if parts else pd.DataFrame()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    return df


def fetch_all(sources=("calls_for_service", "epr", "service_311"), years=None, sample=None):
    """Pull every configured table (+ its metadata). `sample` caps rows per table for dev."""
    config.ensure_dirs()
    ds = config.datasets()
    years = years or config.years()
    for source in sources:
        for key, dataset_id in ds[source].items():
            if key != "all" and int(key) not in years:
                continue
            out = config.RAW / "{}_{}.parquet".format(source, key)
            print("[fetch] {} {} ({}) -> {}".format(source, key, dataset_id, out.name))
            fetch_metadata(dataset_id)
            fetch_dataset(dataset_id, out, select=ds["select"].get(source), max_rows=sample)


def read_raw(source: str) -> pd.DataFrame:
    """Concatenate data/raw/<source>_*.parquet, adding a `source_year` column."""
    files = sorted(config.RAW.glob("{}_*.parquet".format(source)))
    if not files:
        raise FileNotFoundError("No raw files for {}; run `python -m nola.pipeline fetch`".format(source))
    frames = []
    for f in files:
        df = pd.read_parquet(f)
        tag = f.stem.rsplit("_", 1)[-1]
        df["source_year"] = int(tag) if tag.isdigit() else pd.NA
        frames.append(df)
    return pd.concat(frames, ignore_index=True, sort=False)
