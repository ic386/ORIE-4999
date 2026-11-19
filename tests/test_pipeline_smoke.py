"""End-to-end run on synthetic raw files (no network): clean -> join -> qa -> h3 -> calendar
-> tables -> temporal -> map -> dedup. Population is skipped (needs Census downloads)."""

import json

import pandas as pd

from conftest import fake_311, fake_cfs, fake_epr
from nola import config, pipeline, socrata


def test_pipeline_end_to_end(tmp_path, monkeypatch):
    dirs = {name: tmp_path / name for name in
            ("RAW", "METADATA", "INTERIM", "PROCESSED", "EXTERNAL", "FIGURES", "TABLES", "MAPS")}
    for name, d in dirs.items():
        monkeypatch.setattr(config, name, d)
    monkeypatch.setattr(config, "ALL_DIRS", list(dirs.values()))
    monkeypatch.setattr(pipeline, "P", dirs["PROCESSED"])
    monkeypatch.setattr(pipeline, "I", dirs["INTERIM"])
    monkeypatch.setattr(pipeline, "T", dirs["TABLES"])
    monkeypatch.setattr(config, "years", lambda: [2023, 2024])
    config.ensure_dirs()

    frames = []
    for i, y in enumerate((2023, 2024)):
        cfs = fake_cfs(2500, year=y, seed=i)
        frames.append(cfs)
        socrata._tidy(cfs).to_parquet(dirs["RAW"] / "calls_for_service_{}.parquet".format(y))
        fake_epr(cfs, seed=i).to_parquet(dirs["RAW"] / "epr_{}.parquet".format(y))
    fake_311(5000).to_parquet(dirs["RAW"] / "service_311_all.parquet")
    (dirs["METADATA"] / "qf6q-pp4b.json").write_text(json.dumps(
        {"id": "qf6q-pp4b", "name": "Calls for Service 2019",
         "columns": [{"fieldName": "nopd_item", "name": "NOPD_Item", "dataTypeName": "text"}]}))

    for step in ["dictionary", "clean", "join", "qa", "h3", "calendar", "tables", "temporal", "dedup"]:
        pipeline.main([step])
    pipeline.main(["map", "--sample"])
    pipeline.main(["map"])

    T = dirs["TABLES"]
    assert (T / "match_cfs_overall.csv").exists()
    assert (T / "qa_call_type_crosswalk_review.csv").exists()
    assert (T / "holiday_lift_regression_irr.csv").exists()
    assert (T / "dedup_311_rule_grid.csv").exists()
    assert (dirs["FIGURES"] / "heatmap_all.png").exists()
    long = pd.read_parquet(dirs["PROCESSED"] / "funnel_hex_r9_category_stage_long.parquet")
    assert set(long["stage"]) == {"1_call", "2_confirmed_incident", "3_filed_report"}
    html = (dirs["MAPS"] / "map_v1.html").read_text()
    assert "/*__DATA__*/null" not in html and '"hexes"' in html
