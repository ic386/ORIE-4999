import pandas as pd

from conftest import fake_311, fake_cfs, fake_epr
from nola import clean
from nola.join import join_cfs_epr, match_rate_report
from nola.socrata import _tidy


def test_normalize_item_id():
    s = pd.Series(["A-00001-24", "a0000124 ", "K-12777-24", None])
    out = clean.normalize_item_id(s)
    assert out.iloc[0] == out.iloc[1] == "A0000124"
    assert out.iloc[2] == "K1277724"
    assert pd.isna(out.iloc[3])


def test_state_plane_lands_in_new_orleans():
    lon, lat = clean.state_plane_to_lonlat(pd.Series(["3680000"]), pd.Series(["530000"]))
    assert -90.2 < lon[0] < -89.9 and 29.9 < lat[0] < 30.0


def test_clean_cfs_coords_and_beats():
    df = clean.clean_cfs(_tidy(fake_cfs(500)))
    assert df["coord_source"].iloc[0] == "state_plane"      # 0,0 point fell back to mapx/mapy
    assert df["coord_source"].ne("none").all()
    assert set(df["zone"].dropna()) <= {"7L", "2U", "1A", "8B"}
    assert (df["response_min"].dropna() == 15).all()
    assert df["item_id_valid"].all()


def test_join_matches_hyphenated_epr_ids():
    raw = fake_cfs(800)
    cfs = clean.clean_cfs(_tidy(raw))
    items = clean.collapse_epr(clean.clean_epr_rows(fake_epr(raw, frac=0.5)))
    assert items["item_id"].is_unique
    j = join_cfs_epr(cfs, items)
    assert abs(j["has_epr"].mean() - 0.5) < 0.01
    rep = match_rate_report(j, items)
    assert rep["epr_in_cfs_by_year"]["match_rate"].iloc[0] == 1.0


def test_clean_311_drops_zero_coords_and_flags():
    s = clean.clean_311(fake_311(1000))
    assert len(s) <= 980 and (s["lat"] != 0).all()
    assert s["cond_streetlight"].any() and s["cond_illegal_dumping"].any() and s["cond_blight"].any()
    assert not s.loc[s["request_reason"] == "Pothole", ["cond_streetlight", "cond_blight"]].any().any()
