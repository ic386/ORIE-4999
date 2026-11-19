"""Dedup prototype on labeled 311 duplicates (status = 'Duplicate Request').

Rule: a request is flagged as a likely duplicate if an earlier request with the same
reason exists in the same or a k-ring-adjacent H3 cell within `window` hours.
Scored with precision / recall against the city's own duplicate label.

Caveat for Comms: CFS is incident-level, so true call-level dedup needs OPCD call logs.
"""

from __future__ import annotations

import h3
import pandas as pd


def flag_prior_match(df: pd.DataFrame, window_hours: float, k: int, key: str = "request_reason",
                     cell: str = "h3_r9", ts: str = "date_created", id_col: str = "service_request") -> pd.Series:
    base = df[[id_col, key, cell, ts]].dropna().copy()
    base[id_col] = base[id_col].astype(str)
    left = base.assign(ring=base[cell].map(lambda c: list(h3.grid_disk(c, k)))).explode("ring")
    right = base.rename(columns={cell: "ring", ts: "prior_ts", id_col: "prior_id"})
    m = pd.merge_asof(
        left.sort_values(ts), right.sort_values("prior_ts"),
        left_on=ts, right_on="prior_ts", by=["ring", key], direction="backward",
        tolerance=pd.Timedelta(hours=window_hours), allow_exact_matches=False,
    )
    return m.groupby(id_col)["prior_id"].agg(lambda s: s.notna().any()).rename("flagged")


def evaluate(df: pd.DataFrame, windows, ks, key: str = "request_reason") -> pd.DataFrame:
    d = df[df["created_reliable"] & ~df["is_invalid"]].copy()
    d["service_request"] = d["service_request"].astype(str)
    truth = d.set_index("service_request")["is_duplicate"]
    rows = []
    for k in ks:
        for w in windows:
            pred = flag_prior_match(d, w, k, key=key).reindex(truth.index, fill_value=False)
            tp = int((pred & truth).sum())
            fp = int((pred & ~truth).sum())
            fn = int((~pred & truth).sum())
            prec = tp / (tp + fp) if tp + fp else float("nan")
            rec = tp / (tp + fn) if tp + fn else float("nan")
            f1 = 2 * prec * rec / (prec + rec) if prec and rec else float("nan")
            rows.append({"ring_k": k, "window_hours": w, "tp": tp, "fp": fp, "fn": fn,
                         "precision": prec, "recall": rec, "f1": f1, "flag_rate": float(pred.mean())})
    return pd.DataFrame(rows).sort_values("f1", ascending=False)
