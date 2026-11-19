"""Synthetic raw tables shaped exactly like the Socrata pulls (field names from the data dictionaries)."""

import numpy as np
import pandas as pd
import pytest

LETTERS = list("ABCDEFGHIJKL")


def fake_cfs(n=3000, year=2024, seed=0):
    rng = np.random.default_rng(seed)
    item = ["{}{:05d}{}".format(LETTERS[i % 12], i, str(year)[-2:]) for i in range(n)]
    t0 = pd.Timestamp("{}-01-01".format(year)) + pd.to_timedelta(rng.integers(0, 365 * 24 * 60, n), unit="min")
    lat = 29.96 + rng.normal(0, 0.02, n)
    lon = -90.06 + rng.normal(0, 0.03, n)
    loc = [{"type": "Point", "coordinates": [float(a), float(b)]} for a, b in zip(lon, lat)]
    loc[0] = {"type": "Point", "coordinates": [0, 0]}          # 0,0 -> state-plane fallback
    types = rng.choice(["THEFT", "DISCHARGING FIREARM", "AREA CHECK", "DISTURBANCE (OTHER)",
                        "AGGRAVATED BATTERY BY SHOOTING", "AUTO ACCIDENT", "BURGLAR ALARM, SILENT"], n)
    disp = rng.choice(["NAT", "RTF", "GOA", "UNF", "VOI", "DUP", "CBC"], n, p=[.5, .2, .15, .03, .04, .04, .04])
    return pd.DataFrame({
        "nopd_item": item, "type_": "X", "typetext": types, "priority": "1J",
        "initialtype": "X", "initialtypetext": types, "initialpriority": "1J",
        "mapx": "3680000", "mapy": "530000",
        "timecreate": t0.strftime("%Y-%m-%dT%H:%M:%S.000"),
        "timedispatch": (t0 + pd.Timedelta(minutes=5)).strftime("%Y-%m-%dT%H:%M:%S.000"),
        "timearrive": (t0 + pd.Timedelta(minutes=15)).strftime("%Y-%m-%dT%H:%M:%S.000"),
        "timeclosed": (t0 + pd.Timedelta(minutes=60)).strftime("%Y-%m-%dT%H:%M:%S.000"),
        "disposition": disp, "dispositiontext": disp,
        "selfinitiated": rng.choice(["Y", "N"], n, p=[.35, .65]),
        "beat": rng.choice(["7L03", "2U04", "1A01", "8B02"], n),
        "block_address": "001XX Canal St", "zip": "70112",
        "policedistrict": rng.choice([1, 2, 7, 8], n).astype(str),
        "location": loc,
    })


def fake_epr(cfs, frac=0.4, seed=1):
    rng = np.random.default_rng(seed)
    pick = cfs.sample(frac=frac, random_state=seed)["nopd_item"]
    rows = []
    for it in pick:
        epr_id = "{}-{}-{}".format(it[0], it[1:6], it[6:])
        for k in range(rng.integers(1, 4)):
            rows.append({"item_number": epr_id, "district": "7", "location": "001XX Canal St",
                         "disposition": "OPEN", "signal_type": "21", "signal_description": "THEFT",
                         "occurred_date_time": "2024-03-01 10:00:00", "charge_code": "14 67 {}".format(k),
                         "charge_description": "THEFT", "victim_fatal_status": "Non-fatal",
                         "report_type": "Incident Report" if k == 0 else "Supplemental Report"})
    return pd.DataFrame(rows)


def fake_311(n=4000, seed=2):
    rng = np.random.default_rng(seed)
    t = pd.Timestamp("2023-01-01") + pd.to_timedelta(rng.integers(0, 2 * 365 * 24 * 60, n), unit="min")
    reasons = rng.choice(["Street Light Outage/Malfunction", "Illegal Dumping", "Property Maintenance",
                          "Pothole"], n)
    lat = 29.96 + rng.normal(0, 0.02, n)
    lon = -90.06 + rng.normal(0, 0.03, n)
    lat[:20], lon[:20] = 0, 0
    status = rng.choice(["Resolved", "Duplicate Request", "Pending", "Invalid Request"], n, p=[.8, .1, .08, .02])
    return pd.DataFrame({
        "service_request": ["SR{:07d}".format(i) for i in range(n)], "request_type": reasons,
        "request_reason": reasons, "date_created": t.strftime("%Y-%m-%dT%H:%M:%S.000"),
        "case_close_date": (t + pd.Timedelta(days=3)).strftime("%Y-%m-%dT%H:%M:%S.000"),
        "request_status": "Closed", "status": status, "responsible_agency": "DPW",
        "address_councildis": "B", "latitude": lat.astype(str), "longitude": lon.astype(str),
    })


@pytest.fixture
def raw_cfs():
    return fake_cfs()
