"""Temporal analysis.

TODOs covered
- Hour x weekday heatmaps            -> hour_weekday(), plot_heatmap()
- Holiday-lift regression            -> holiday_lift_ratio(), holiday_lift_regression()
- Seasonality decomposition          -> seasonality()
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import statsmodels.api as sm  # noqa: E402
import statsmodels.formula.api as smf  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

# Single-hue sequential ramp (light -> dark), validated default palette
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SEQ_CMAP = LinearSegmentedColormap.from_list("seq_blue", SEQ)
INK, MUTED = "#1f1f1d", "#6b6a66"
DOW = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def hour_weekday(df: pd.DataFrame, ts: str = "time_create") -> pd.DataFrame:
    t = pd.to_datetime(df[ts]).dropna()
    pv = pd.crosstab(t.dt.dayofweek, t.dt.hour).reindex(index=range(7), columns=range(24), fill_value=0)
    pv.index = DOW
    return pv


def plot_heatmap(pv: pd.DataFrame, title: str, path, normalize: bool = True):
    data = pv / pv.values.sum() * 100 if normalize else pv
    fig, ax = plt.subplots(figsize=(10, 3.4), dpi=150)
    im = ax.imshow(data.values, aspect="auto", cmap=SEQ_CMAP)
    ax.set_xticks(range(0, 24, 2), ["{:02d}".format(h) for h in range(0, 24, 2)], color=MUTED, fontsize=8)
    ax.set_yticks(range(7), data.index, color=MUTED, fontsize=8)
    ax.set_xlabel("Hour of day", color=MUTED, fontsize=8)
    ax.set_title(title, loc="left", color=INK, fontsize=11)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-0.5, 24, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 7, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)   # 2px-style surface gap between cells
    ax.tick_params(which="both", length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label("% of calls" if normalize else "calls", color=MUTED, fontsize=8)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=7, colors=MUTED, length=0)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def daily_counts(df: pd.DataFrame, ts: str = "time_create", start=None, end=None) -> pd.Series:
    d = pd.to_datetime(df[ts]).dropna().dt.normalize()
    s = d.value_counts().sort_index()
    idx = pd.date_range(start or s.index.min(), end or s.index.max(), freq="D")
    return s.reindex(idx, fill_value=0).rename("count")


def holiday_lift_ratio(daily: pd.Series, cal: pd.DataFrame, event_cols: list[str],
                       window_weeks: int = 4) -> pd.DataFrame:
    """Event-day volume / mean of the same weekday +/- window_weeks (excluding events & confounders)."""
    c = cal.set_index("date").reindex(daily.index)
    clean = ~(c["any_event"].fillna(False) | c["any_confounder"].fillna(False) | c["carnival_season"].fillna(False))
    rows = []
    for col in event_cols:
        for day in c.index[c[col].fillna(False).astype(bool)]:
            ref = [day + pd.Timedelta(weeks=w) for w in range(-window_weeks, window_weeks + 1) if w]
            ref = [r for r in ref if r in daily.index and clean.get(r, False)]
            if len(ref) >= 2:
                base = daily.loc[ref].mean()
                rows.append({"event": col, "date": day, "count": daily[day], "baseline": base,
                             "lift": daily[day] / base if base else np.nan})
    per_day = pd.DataFrame(rows)
    if per_day.empty:
        return per_day
    summary = per_day.groupby("event").agg(days=("lift", "size"), mean_lift=("lift", "mean"),
                                            median_lift=("lift", "median")).reset_index()
    return summary.sort_values("mean_lift", ascending=False)


def holiday_lift_regression(daily: pd.Series, cal: pd.DataFrame, event_cols: list[str],
                            confounder_cols: list[str] | None = None) -> pd.DataFrame:
    """Negative-binomial GLM: count ~ weekday + month + year + events + confounders.

    exp(coef) of an event = multiplicative lift over a comparable ordinary day.
    alpha (overdispersion) estimated by the Cameron-Trivedi auxiliary regression.
    """
    confounder_cols = confounder_cols or []
    d = cal.set_index("date").reindex(daily.index).assign(count=daily.values)
    cols = [c for c in event_cols + confounder_cols if d[c].fillna(False).any()]
    for c in cols:
        d[c] = d[c].fillna(False).astype(int)
    rhs = " + ".join(["C(dow)", "C(month)", "C(year)"] + cols)
    pois = smf.glm("count ~ {}".format(rhs), data=d, family=sm.families.Poisson()).fit()
    mu = pois.mu
    aux = ((d["count"] - mu) ** 2 - d["count"]) / mu
    alpha = max(sm.OLS(aux, mu).fit().params.iloc[0], 1e-6)
    nb = smf.glm("count ~ {}".format(rhs), data=d, family=sm.families.NegativeBinomial(alpha=alpha)).fit()
    ci = nb.conf_int()
    out = pd.DataFrame({"term": nb.params.index, "irr": np.exp(nb.params.values),
                        "ci_low": np.exp(ci[0].values), "ci_high": np.exp(ci[1].values),
                        "p_value": nb.pvalues.values})
    out = out[out["term"].isin(cols)].sort_values("irr", ascending=False)
    out.attrs["alpha"] = alpha
    return out


def seasonality(daily: pd.Series) -> pd.DataFrame:
    """MSTL with weekly + yearly periods when >= 2 years of data, else weekly STL."""
    from statsmodels.tsa.seasonal import MSTL, STL

    y = daily.astype(float)
    if len(y) >= 2 * 365:
        res = MSTL(y, periods=(7, 365)).fit()
        comp = pd.DataFrame({"observed": res.observed, "trend": res.trend,
                             "weekly": res.seasonal["seasonal_7"], "yearly": res.seasonal["seasonal_365"],
                             "resid": res.resid})
    else:
        res = STL(y, period=7).fit()
        comp = pd.DataFrame({"observed": res.observed, "trend": res.trend,
                             "weekly": res.seasonal, "resid": res.resid})
    return comp


def plot_seasonality(comp: pd.DataFrame, path, title: str = "Daily calls: decomposition"):
    cols = [c for c in ("observed", "trend", "weekly", "yearly", "resid") if c in comp]
    fig, axes = plt.subplots(len(cols), 1, figsize=(10, 1.6 * len(cols)), dpi=150, sharex=True)
    for ax, c in zip(axes, cols):
        ax.plot(comp.index, comp[c], color="#256abf", linewidth=1 if c in ("observed", "resid") else 2)
        ax.set_ylabel(c, color=MUTED, fontsize=8)
        ax.tick_params(colors=MUTED, labelsize=7, length=0)
        ax.grid(axis="y", color="#e6e5e1", linewidth=0.6)
        for s in ax.spines.values():
            s.set_visible(False)
    axes[0].set_title(title, loc="left", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
