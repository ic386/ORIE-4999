import h3
import numpy as np
import pandas as pd

from nola.dedup import evaluate, flag_prior_match
from nola.funnel import apply_crosswalk, assign_stages, eb_beta_binomial
from nola.h3utils import assign_h3, resolution_sensitivity


def test_crosswalk_order():
    cw = apply_crosswalk(pd.Series(["AGGRAVATED BATTERY BY SHOOTING", "DISCHARGING FIREARM",
                                    "TRAFFIC STOP", "AUTO ACCIDENT", "SIMPLE BURGLARY", "SOMETHING NEW"]))
    assert cw["category"].tolist() == ["shooting", "shots_fired", "proactive", "traffic", "property", "other"]
    assert cw["report_expected"].tolist()[:2] == [True, False]


def test_stages():
    df = pd.DataFrame({"disposition": ["RTF", "NAT", "GOA", "VOI", "REF"],
                       "has_incident_report": [True, False, False, False, False],
                       "self_initiated": False})
    s = assign_stages(df)
    assert s["stage_call"].tolist() == [True, True, True, False, True]
    assert s["stage_confirmed"].tolist() == [True, True, False, False, False]
    assert s["stage_reported"].tolist() == [True, False, False, False, False]


def test_eb_shrinks_small_hexes_more():
    rng = np.random.default_rng(0)
    n = pd.Series(np.r_[rng.integers(200, 400, 50), [2]])
    k = pd.Series(np.r_[rng.binomial(n[:50], 0.3), [2]])     # tiny hex at 100%
    eb = eb_beta_binomial(k, n)
    assert eb.iloc[-1] < 0.9                    # pulled toward ~0.3
    assert abs(eb.iloc[0] - k[0] / n[0]) < 0.05  # big hex barely moves


def test_assign_h3_parent_consistency():
    df = assign_h3(pd.DataFrame({"lat": [29.95, np.nan], "lon": [-90.07, -90.0]}), resolutions=(8, 9))
    assert h3.cell_to_parent(df["h3_r9"][0], 8) == df["h3_r8"][0]
    assert pd.isna(df["h3_r9"][1])


def test_resolution_sensitivity_runs():
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"lat": 29.96 + rng.normal(0, .02, 2000), "lon": -90.06 + rng.normal(0, .03, 2000)})
    out = resolution_sensitivity(df, resolutions=(7, 8, 9))
    assert list(out["res"]) == [7, 8, 9]
    assert out["n_cells"].is_monotonic_increasing


def test_dedup_rule():
    c = h3.latlng_to_cell(29.95, -90.07, 9)
    nb = [x for x in h3.grid_disk(c, 1) if x != c][0]
    t = pd.Timestamp("2024-05-01 10:00")
    df = pd.DataFrame({
        "service_request": ["a", "b", "c", "d"],
        "request_reason": ["Illegal Dumping"] * 3 + ["Pothole"],
        "h3_r9": [c, c, nb, c],
        "date_created": [t, t + pd.Timedelta(hours=2), t + pd.Timedelta(hours=30), t + pd.Timedelta(hours=1)],
        "is_duplicate": [False, True, True, False], "is_invalid": False, "created_reliable": True,
    })
    f0 = flag_prior_match(df, 6, 0)
    assert f0.to_dict() == {"a": False, "b": True, "c": False, "d": False}
    f1 = flag_prior_match(df, 48, 1)
    assert f1["c"]
    grid = evaluate(df, [6, 48], [0, 1])
    best = grid.iloc[0]
    assert best["recall"] == 1.0 and best["precision"] == 1.0
