"""Reproducible pipeline. Each step reads the previous step's parquet and writes its own.

    python -m nola.pipeline fetch --sample 20000   # dev pull
    python -m nola.pipeline all                    # everything after fetch

Steps: fetch -> dictionary -> clean -> join -> qa -> h3 -> calendar -> tables
       -> population -> temporal -> map -> dedup
"""

from __future__ import annotations

import argparse

import pandas as pd

from . import config

P = config.PROCESSED
I = config.INTERIM  # noqa: E741
T = config.TABLES


def _save_tables(tables: dict[str, pd.DataFrame], prefix: str) -> None:
    for name, df in tables.items():
        df.to_csv(T / "{}_{}.csv".format(prefix, name))


# --------------------------------------------------------------------------- steps
def step_fetch(args):
    from .socrata import fetch_all

    srcs = args.sources or ("calls_for_service", "epr", "service_311")
    fetch_all(sources=srcs, years=args.years, sample=args.sample)


def step_dictionary(args):
    from .dictionary import write_dictionary_outputs

    write_dictionary_outputs()


def step_clean(args):
    from . import clean
    from .socrata import read_raw

    cfs = clean.clean_cfs(read_raw("calls_for_service"))
    cfs.to_parquet(I / "cfs.parquet", index=False)
    rows = clean.clean_epr_rows(read_raw("epr"))
    rows.to_parquet(I / "epr_rows.parquet", index=False)
    items = clean.collapse_epr(rows)
    items.to_parquet(I / "epr_items.parquet", index=False)
    s311 = clean.clean_311(read_raw("service_311"))
    s311.to_parquet(I / "s311.parquet", index=False)
    print("[clean] cfs={:,} epr_items={:,} (rows {:,}) 311={:,}".format(len(cfs), len(items), len(rows), len(s311)))


def step_join(args):
    from .funnel import apply_crosswalk, assign_stages
    from .join import join_cfs_epr, match_rate_report

    cfs = pd.read_parquet(I / "cfs.parquet")
    items = pd.read_parquet(I / "epr_items.parquet")
    j = join_cfs_epr(cfs, items)
    j = j.join(apply_crosswalk(j["type_text"]))
    j = assign_stages(j)
    j.to_parquet(P / "cfs_epr.parquet", index=False)
    rep = match_rate_report(j, items)
    _save_tables(rep, "match")
    o = rep["cfs_overall"].iloc[0]
    print("[join] {:,} calls; {:.1%} have an EPR "
          "({:.1%} of resident-initiated) -> outputs/tables/match_*.csv".format(int(o.n_cfs), o.share_with_epr, o.resident_share_with_epr))


def step_qa(args):
    from .dictionary import code_stability, year_over_year_jaccard
    from .funnel import crosswalk_review

    j = pd.read_parquet(P / "cfs_epr.parquet")
    for col in ("disposition", "type_text", "initial_type_text", "priority"):
        code_stability(j, col).to_csv(T / "qa_cfs_{}_by_year.csv".format(col))
        year_over_year_jaccard(j, col).to_csv(T / "qa_cfs_{}_jaccard.csv".format(col), index=False)
    rows = pd.read_parquet(I / "epr_rows.parquet")
    rows["year"] = rows["source_year"]
    for col in ("disposition", "signal_type", "report_type"):
        code_stability(rows, col).to_csv(T / "qa_epr_{}_by_year.csv".format(col))
    crosswalk_review(j).to_csv(T / "qa_call_type_crosswalk_review.csv")

    by_year = j.groupby("year").agg(
        n=("item_id", "size"),
        geocoded=("coord_source", lambda s: s.ne("none").mean()),
        from_state_plane=("coord_source", lambda s: s.eq("state_plane").mean()),
        beat_parsed=("zone", lambda s: s.notna().mean()),
        missing_arrival=("time_arrive", lambda s: s.isna().mean()),
        self_initiated=("self_initiated", "mean"),
        unmapped_category=("category", lambda s: s.isin(["other", "unmapped"]).mean()),
    )
    by_year.to_csv(T / "qa_cfs_completeness_by_year.csv")

    s311 = pd.read_parquet(I / "s311.parquet")
    stamps = s311["date_created"].value_counts().head(25).rename("rows_sharing_timestamp")
    stamps.to_csv(T / "qa_311_shared_created_timestamps.csv")
    s311.groupby("year")[["is_duplicate", "is_invalid", "cond_streetlight", "cond_illegal_dumping",
                         "cond_blight"]].mean().to_csv(T / "qa_311_flags_by_year.csv")
    print("[qa] wrote outputs/tables/qa_*.csv - review crosswalk + dispositions, then edit config/funnel.yaml")


def step_h3(args):
    from .h3utils import assign_h3, modal_lookup, resolution_sensitivity

    p = config.params()["h3"]
    res = (p["public_res"], p["analysis_res"])
    j = assign_h3(pd.read_parquet(P / "cfs_epr.parquet"), resolutions=res)
    j.to_parquet(P / "cfs_epr_h3.parquet", index=False)
    s = assign_h3(pd.read_parquet(I / "s311.parquet"), resolutions=res)
    s.to_parquet(P / "s311_h3.parquet", index=False)
    for r in res:
        cell = "h3_r{}".format(r)
        lk = modal_lookup(j, cell, "police_district")
        for extra in ("zone", "subzone"):
            lk = lk.merge(modal_lookup(j, cell, extra), on=cell, how="outer")
        lk.to_csv(T / "hex_r{}_to_nopd_district_zone.csv".format(r), index=False)
    resolution_sensitivity(j, resolutions=p["sensitivity_res"]).to_csv(
        T / "h3_resolution_sensitivity.csv", index=False)
    print("[h3] indexed {:.1%} of calls, "
          "{:.1%} of 311 requests".format(j['h3_r{}'.format(res[1])].notna().mean(), s['h3_r{}'.format(res[1])].notna().mean()))


def step_calendar(args):
    from .events import build_calendar

    ys = config.years()
    cal = build_calendar("{}-01-01".format(min(ys)), "{}-12-31".format(max(ys)))
    cal.to_parquet(P / "calendar.parquet", index=False)
    cal.to_csv(T / "event_calendar.csv", index=False)
    print("[calendar] {:,} days, {} event days, "
          "{} confounder days".format(len(cal), cal['any_event'].sum(), cal['any_confounder'].sum()))


def step_tables(args):
    from .funnel import analysis_table, funnel_counts, long_stage_table

    j = pd.read_parquet(P / "cfs_epr_h3.parquet")
    p = config.params()["h3"]
    for r in (p["public_res"], p["analysis_res"]):
        wide = analysis_table(j, r)
        wide.to_parquet(P / "funnel_hex_r{}_category.parquet".format(r), index=False)
        long_stage_table(wide, ["h3_r{}".format(r), "category"]).to_parquet(
            P / "funnel_hex_r{}_category_stage_long.parquet".format(r), index=False)
        analysis_table(j, r, extra_by=["year"]).to_parquet(P / "funnel_hex_r{}_category_year.parquet".format(r), index=False)
    # NOPD units - tabular breakdowns for districts (Ryan)
    for unit in ("police_district", "zone", "subzone"):
        funnel_counts(j, [unit, "category"]).to_csv(T / "funnel_by_{}_category.csv".format(unit), index=False)
        funnel_counts(j, [unit]).to_csv(T / "funnel_by_{}.csv".format(unit), index=False)
    # 311 neighborhood conditions per hex (reliable dates, valid, non-duplicate)
    s = pd.read_parquet(P / "s311_h3.parquet")
    s = s[~s["is_duplicate"] & ~s["is_invalid"]]
    cell = "h3_r{}".format(p['analysis_res'])
    cond = s.groupby(cell)[["cond_streetlight", "cond_illegal_dumping", "cond_blight"]].sum()
    cond.to_parquet(P / "hex_r{}_311_conditions.parquet".format(p['analysis_res']))
    print("[tables] funnel tables -> data/processed/, NOPD-unit breakdowns -> outputs/tables/")


def step_population(args):
    from .population import build_hex_population, per_capita

    pops = build_hex_population(resolutions=(config.params()["h3"]["public_res"],
                                             config.params()["h3"]["analysis_res"]))
    for r, pop in pops.items():
        pop.to_parquet(P / "hex_r{}_population.parquet".format(r), index=False)
        f = P / "funnel_hex_r{}_category.parquet".format(r)
        if f.exists():
            wide = pd.read_parquet(f)
            tot = wide.groupby("h3_r{}".format(r), as_index=False)[["n_calls", "n_confirmed", "n_reported"]].sum()
            per_capita(tot, pop, "h3_r{}".format(r), ["n_calls", "n_confirmed", "n_reported"]).to_csv(
                T / "hex_r{}_per_capita.csv".format(r), index=False)
    print("[population] block-group -> hex population + per-capita tables written")


def step_temporal(args):
    from . import temporal as tm

    j = pd.read_parquet(P / "cfs_epr_h3.parquet")
    j = j[j["stage_call"]]
    F = config.FIGURES
    tm.plot_heatmap(tm.hour_weekday(j), "All calls for service: hour x weekday", F / "heatmap_all.png")
    res = j[~j["self_initiated"]]
    tm.plot_heatmap(tm.hour_weekday(res), "Resident-initiated calls: hour x weekday", F / "heatmap_resident.png")
    for cat in j["category"].value_counts().head(8).index:
        tm.plot_heatmap(tm.hour_weekday(j[j["category"] == cat]), "{}: hour x weekday".format(cat.replace('_', ' ').title()),
                        F / "heatmap_cat_{}.png".format(cat))
    for d in sorted(j["police_district"].dropna().unique()):
        tm.plot_heatmap(tm.hour_weekday(j[j["police_district"] == d]), "District {}: hour x weekday".format(d),
                        F / "heatmap_district_{}.png".format(d))

    cal = pd.read_parquet(P / "calendar.parquet")
    ev = [c for c in cal.columns if c in {"mardi_gras_day", "lundi_gras", "parade_day", "jazz_fest",
                                          "essence_fest", "southern_decadence", "bayou_classic", "sugar_bowl",
                                          "super_bowl_lix", "new_years_day", "july_4", "halloween",
                                          "new_years_eve", "second_lines", "special_events"}]
    cf = [c for c in cal.columns if c.startswith("hurricane_") or c in {"covid_stay_home", "bourbon_st_attack"}]
    daily = tm.daily_counts(j, start=cal["date"].min(), end=cal["date"].max())
    daily.to_frame().to_csv(T / "daily_call_counts.csv")
    tm.holiday_lift_ratio(daily, cal, ev).to_csv(T / "holiday_lift_ratio.csv", index=False)
    reg = tm.holiday_lift_regression(daily, cal, ev, cf)
    reg.to_csv(T / "holiday_lift_regression_irr.csv", index=False)
    comp = tm.seasonality(daily)
    comp.to_csv(T / "seasonality_components.csv")
    tm.plot_seasonality(comp, F / "seasonality_decomposition.png")
    print("[temporal] heatmaps -> outputs/figures/, lift tables -> outputs/tables/ (NB alpha={:.3f})".format(reg.attrs['alpha']))


def step_map(args):
    from .maps import build_payload, render, sample_points

    res = config.params()["h3"]["public_res"]
    if args.sample:
        payload = build_payload(sample_points(), res=res, title="Map v1 prototype - SYNTHETIC sample data")
        out = render(payload, config.MAPS / "map_v1_sample.html")
    else:
        j = pd.read_parquet(P / "cfs_epr_h3.parquet")
        j = j[j["stage_call"]]
        popf = P / "hex_r{}_population.parquet".format(res)
        pop = pd.read_parquet(popf) if popf.exists() else None
        payload = build_payload(j, res=res, pop=pop, title="New Orleans calls for service, 2019-2025")
        out = render(payload, config.MAPS / "map_v1.html")
    print("[map] outputs/maps/{} ({} hexes) - open in a browser".format(out.name, len(payload['hexes'])))


def step_dedup(args):
    from .dedup import evaluate

    s = pd.read_parquet(P / "s311_h3.parquet")
    p = config.params()["dedup"]
    grid = evaluate(s, p["window_hours"], p["ring_k"], key=p["match_on"])
    grid.to_csv(T / "dedup_311_rule_grid.csv", index=False)
    b = grid.iloc[0]
    print("[dedup] best F1={:.3f} (k={}, {}h, "
          "precision={:.2f}, recall={:.2f})".format(b.f1, int(b.ring_k), b.window_hours, b.precision, b.recall))


STEPS = {
    "fetch": step_fetch, "dictionary": step_dictionary, "clean": step_clean, "join": step_join,
    "qa": step_qa, "h3": step_h3, "calendar": step_calendar, "tables": step_tables,
    "population": step_population, "temporal": step_temporal, "map": step_map, "dedup": step_dedup,
}
ALL_AFTER_FETCH = ["dictionary", "clean", "join", "qa", "h3", "calendar", "tables",
                   "population", "temporal", "map", "dedup"]


def main(argv=None):
    ap = argparse.ArgumentParser(prog="nola.pipeline", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=[*STEPS, "all"])
    ap.add_argument("--sample", nargs="?", const=20000, type=int, default=None,
                    help="fetch: cap rows per table; map: use synthetic data")
    ap.add_argument("--years", nargs="*", type=int, default=None)
    ap.add_argument("--sources", nargs="*", default=None,
                    choices=["calls_for_service", "epr", "service_311"])
    args = ap.parse_args(argv)
    config.ensure_dirs()
    for s in (ALL_AFTER_FETCH if args.step == "all" else [args.step]):
        STEPS[s](args)


if __name__ == "__main__":
    main()
