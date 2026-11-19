"""Holiday & event calendar (one row per day) with Mardi Gras computed per year.

TODO covered: Build the holiday and event calendar.

Mardi Gras = Easter - 47 days. Parade nights are placed by their offset from Fat Tuesday
(Ryan: the parade pattern is fixed relative to Mardi Gras), so the same config works for
every year past and future. Hurricanes, COVID and the Jan 1 2025 attack are flagged as
confounders, not events.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

from . import config


def easter(year: int) -> date:
    """Anonymous Gregorian algorithm."""
    a, b, c = year % 19, year // 100, year % 100
    d, e = b // 4, b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    ell = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * ell) // 451
    month = (h + ell - 7 * m + 114) // 31
    day = (h + ell - 7 * m + 114) % 31 + 1
    return date(year, month, day)


def mardi_gras(year: int) -> date:
    return easter(year) - timedelta(days=47)


def nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """n-th (1-based) weekday (Mon=0) of a month; n=-1 for the last."""
    if n > 0:
        d = date(year, month, 1)
        d += timedelta(days=(weekday - d.weekday()) % 7)
        return d + timedelta(weeks=n - 1)
    nxt = date(year + (month == 12), month % 12 + 1, 1)
    d = nxt - timedelta(days=1)
    return d - timedelta(days=(d.weekday() - weekday) % 7)


def _span(start: date, n_days: int) -> list[date]:
    return [start + timedelta(days=i) for i in range(n_days)]


def _rule_dates(name: str, y: int) -> list[date]:
    if name == "jazz_fest":
        fri = nth_weekday(y, 4, 4, -1)
        return _span(fri, 3) + _span(fri + timedelta(days=6), 4)
    if name == "essence_fest":
        jul5 = date(y, 7, 5)
        fri = jul5 - timedelta(days=(jul5.weekday() - 4) % 7)
        return _span(fri, 3)
    if name == "southern_decadence":
        labor = nth_weekday(y, 9, 0, 1)
        return _span(labor - timedelta(days=4), 5)
    if name == "bayou_classic":
        return [nth_weekday(y, 11, 3, 4) + timedelta(days=2)]
    raise KeyError(name)


def build_calendar(start: str | date, end: str | date, cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or config.load("events")
    days = pd.date_range(start, end, freq="D")
    cal = pd.DataFrame({"date": days})
    cal["year"] = cal["date"].dt.year
    cal["dow"] = cal["date"].dt.dayofweek
    cal["is_weekend"] = cal["dow"] >= 5
    cal["month"] = cal["date"].dt.month
    cal["federal_holiday"] = cal["date"].isin(USFederalHolidayCalendar().holidays(days.min(), days.max()))

    events: dict[str, set] = {}
    kinds: dict[str, str] = {}

    def add(name, ds, kind="event"):
        events.setdefault(name, set()).update(pd.Timestamp(d) for d in ds)
        kinds[name] = kind

    years = range(days.min().year, days.max().year + 1)
    carn = cfg["carnival"]
    for y in years:
        mg = mardi_gras(y)
        add("mardi_gras_day", [mg])
        add("lundi_gras", [mg - timedelta(days=1)])
        s = date(y, carn["season_start"]["month"], carn["season_start"]["day"])
        add("carnival_season", _span(s, (mg - s).days + 1))
        cancelled = y in carn.get("parades_cancelled_years", [])
        add("parade_day", [] if cancelled else [mg + timedelta(days=o) for o in carn["parade_offsets_days"]])
        for f in cfg.get("fixed", []):
            add(f["name"], [date(y, f["month"], f["day"])])
        for name, rule in cfg.get("rules", {}).items():
            cancelled = y in (rule or {}).get("cancelled_years", [])
            add(name, [] if cancelled else _rule_dates(name, y))

    for ev in cfg.get("explicit", []):
        s = pd.Timestamp(ev["start"]).date()
        e = pd.Timestamp(ev.get("end", ev["start"])).date()
        add(ev["name"], _span(s, (e - s).days + 1), ev.get("kind", "event"))

    for key, rel in (cfg.get("external") or {}).items():
        p = config.ROOT / rel
        if Path(p).exists():
            ext = pd.read_csv(p, parse_dates=["date"])
            add(key, ext["date"].dt.date.tolist())

    for name, ds in events.items():
        cal[name] = cal["date"].isin(ds)
    cal["days_to_mardi_gras"] = [
        (pd.Timestamp(mardi_gras(d.year)) - d).days for d in cal["date"]
    ]
    ev_cols = [n for n, k in kinds.items() if k == "event" and n != "carnival_season"]
    cf_cols = [n for n, k in kinds.items() if k == "confounder"]
    cal["any_event"] = cal[ev_cols].any(axis=1)
    cal["any_confounder"] = cal[cf_cols].any(axis=1) if cf_cols else False
    cal["event_label"] = cal[ev_cols].apply(lambda r: ",".join(c for c in ev_cols if r[c]), axis=1)
    cal.attrs["event_columns"] = ev_cols
    cal.attrs["confounder_columns"] = cf_cols
    return cal
