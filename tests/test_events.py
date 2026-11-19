from datetime import date

import pytest

from nola.events import build_calendar, easter, mardi_gras, nth_weekday


@pytest.mark.parametrize("year,expected", [
    (2019, date(2019, 3, 5)), (2020, date(2020, 2, 25)), (2021, date(2021, 2, 16)),
    (2022, date(2022, 3, 1)), (2023, date(2023, 2, 21)), (2024, date(2024, 2, 13)),
    (2025, date(2025, 3, 4)), (2026, date(2026, 2, 17)),
])
def test_mardi_gras(year, expected):
    assert mardi_gras(year) == expected


def test_easter():
    assert easter(2024) == date(2024, 3, 31)


def test_nth_weekday():
    assert nth_weekday(2024, 11, 3, 4) == date(2024, 11, 28)   # Thanksgiving
    assert nth_weekday(2024, 4, 4, -1) == date(2024, 4, 26)    # last Friday of April


def test_calendar_flags():
    cal = build_calendar("2024-01-01", "2024-12-31").set_index("date")
    assert cal.loc["2024-02-13", "mardi_gras_day"]
    assert cal.loc["2024-02-12", "lundi_gras"]
    assert cal.loc["2024-02-13", "days_to_mardi_gras"] == 0
    assert cal.loc["2024-07-04", "july_4"] and cal.loc["2024-07-04", "federal_holiday"]
    assert cal.loc["2024-11-30", "bayou_classic"]
    assert cal.loc["2024-09-12", "hurricane_francine"] and cal.loc["2024-09-12", "any_confounder"]
    assert not cal.loc["2024-03-12", "any_event"]


def test_cancelled_years():
    cal = build_calendar("2021-01-01", "2021-12-31")
    assert not cal["parade_day"].any()
    assert not cal["jazz_fest"].any()
