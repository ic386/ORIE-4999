"""Funnel stages, call-type crosswalk, empirical-Bayes smoothing, analysis tables.

TODOs covered
- Finalize the funnel-stage definitions and call-type crosswalk -> assign_stages(), apply_crosswalk(),
                                                                   crosswalk_review()
- Build analysis tables (hex x call type x funnel stage)         -> analysis_table()
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from . import config


# --------------------------------------------------------------------------- crosswalk
def apply_crosswalk(type_text: pd.Series, cfg: dict | None = None) -> pd.DataFrame:
    """type_text -> (category, report_expected). CSV overrides first, then regex rules in order."""
    cfg = cfg or config.load("funnel")
    overrides = pd.read_csv(config.CONFIG_DIR / "call_type_crosswalk.csv")
    ov = dict(zip(overrides["type_text"].str.upper(), zip(overrides["category"], overrides["report_expected"])))
    rules = [(c["name"], bool(c["report_expected"]), re.compile(c["pattern"], re.I)) for c in cfg["categories"]]
    expected = {c["name"]: bool(c["report_expected"]) for c in cfg["categories"]}

    def one(t):
        if t is None or (isinstance(t, float) and np.isnan(t)) or t is pd.NA:
            return ("unmapped", False)
        t = str(t).upper()
        if t in ov:
            cat, rep = ov[t]
            return (cat, str(rep).lower() == "true" if not isinstance(rep, bool) else rep)
        for name, rep, rx in rules:
            if rx.search(t):
                return (name, rep)
        return ("other", expected.get("other", False))

    uniq = pd.Series(type_text.dropna().unique())
    lut = {t: one(t) for t in uniq}
    mapped = type_text.map(lambda t: lut.get(t, ("unmapped", False)))
    return pd.DataFrame({"category": mapped.map(lambda x: x[0]),
                         "report_expected": mapped.map(lambda x: x[1])}, index=type_text.index)


def crosswalk_review(df: pd.DataFrame) -> pd.DataFrame:
    """Every observed type_text with its assigned category and counts by year - for the team to finalize."""
    ct = pd.crosstab(df["type_text"].fillna("<NA>"), df["year"])
    ct["total"] = ct.sum(axis=1)
    cw = apply_crosswalk(pd.Series(ct.index, index=ct.index))
    return ct.join(cw).sort_values("total", ascending=False)


# --------------------------------------------------------------------------- stages
def assign_stages(df: pd.DataFrame, cfg: dict | None = None) -> pd.DataFrame:
    """Adds disposition_group, stage_call, stage_confirmed, stage_reported(, stage_rtf)."""
    cfg = cfg or config.load("funnel")
    d = cfg["dispositions"]
    group = pd.Series("unknown", index=df.index)
    for g in ("confirmed", "unconfirmed", "excluded", "other"):
        group[df["disposition"].isin(d[g])] = g
    out = df.copy()
    out["disposition_group"] = group
    out["stage_call"] = group.ne("excluded")
    out["stage_confirmed"] = out["stage_call"] & group.eq("confirmed")
    rep_types = set(cfg["report"]["epr_report_types"])
    has_rep = out.get("has_incident_report", pd.Series(False, index=df.index)).fillna(False).astype(bool)
    if rep_types != {"Incident Report"}:
        has_rep = out.get("has_epr", pd.Series(False, index=df.index)).fillna(False).astype(bool)
    out["stage_reported"] = out["stage_call"] & has_rep
    out["stage_rtf"] = out["stage_call"] & out["disposition"].eq("RTF")  # proxy when EPR join is thin
    out["is_goa"] = out["disposition"].eq("GOA")
    out["is_unfounded"] = out["disposition"].eq("UNF")
    return out


# --------------------------------------------------------------------------- smoothing
def eb_beta_binomial(successes: pd.Series, trials: pd.Series, min_trials: int = 1) -> pd.Series:
    """Empirical-Bayes posterior mean for a rate, Beta prior fit by method of moments.

    Low-count hexes are pulled toward the citywide rate so they don't produce misleading
    extremes; high-count hexes keep their observed rate.
    """
    k = successes.astype(float)
    n = trials.astype(float)
    use = n >= min_trials
    if use.sum() < 2:
        return pd.Series(np.where(n > 0, k / n.replace(0, np.nan), np.nan), index=k.index)
    p = k[use] / n[use]
    mu = k[use].sum() / n[use].sum()
    # Variance of observed rates minus expected binomial noise -> between-hex variance
    var_obs = np.average((p - mu) ** 2, weights=n[use])
    var_binom = mu * (1 - mu) * np.mean(1 / n[use])
    tau2 = max(var_obs - var_binom, 1e-8)
    s = max(mu * (1 - mu) / tau2 - 1, 1e-6)   # prior strength alpha + beta
    alpha, beta = mu * s, (1 - mu) * s
    return ((k + alpha) / (n + alpha + beta)).where(n > 0)


# --------------------------------------------------------------------------- tables
def funnel_counts(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    calls = df[df["stage_call"]]
    g = calls.groupby(by, dropna=False).agg(
        n_calls=("stage_call", "size"),
        n_confirmed=("stage_confirmed", "sum"),
        n_reported=("stage_reported", "sum"),
        n_rtf=("stage_rtf", "sum"),
        n_goa=("is_goa", "sum"),
        n_unfounded=("is_unfounded", "sum"),
        n_self_initiated=("self_initiated", "sum"),
    ).reset_index()
    g["n_resident"] = g["n_calls"] - g["n_self_initiated"]
    g["rate_confirmed"] = g["n_confirmed"] / g["n_calls"]
    g["rate_reported_given_confirmed"] = g["n_reported"] / g["n_confirmed"].replace(0, np.nan)
    g["rate_call_to_report"] = g["n_reported"] / g["n_calls"]
    g["share_goa_unf"] = (g["n_goa"] + g["n_unfounded"]) / g["n_calls"]
    g["share_self_initiated"] = g["n_self_initiated"] / g["n_calls"]
    # EB-smoothed rates, shrunk toward the citywide rate *within* each call category
    for num, den, name in [("n_confirmed", "n_calls", "eb_rate_confirmed"),
                           ("n_reported", "n_calls", "eb_rate_call_to_report"),
                           ("n_reported", "n_confirmed", "eb_rate_reported_given_confirmed")]:
        if "category" in by:
            parts = [eb_beta_binomial(sub[num], sub[den]) for _, sub in g.groupby("category")]
            g[name] = pd.concat(parts).reindex(g.index)
        else:
            g[name] = eb_beta_binomial(g[num], g[den])
    return g


def analysis_table(df: pd.DataFrame, res: int, extra_by: list[str] | None = None) -> pd.DataFrame:
    """Wide table: one row per hex x category (x extras) with counts at each funnel stage."""
    by = ["h3_r{}".format(res), "category"] + (extra_by or [])
    return funnel_counts(df.dropna(subset=["h3_r{}".format(res)]), by)


def long_stage_table(wide: pd.DataFrame, id_cols: list[str]) -> pd.DataFrame:
    """hex x call type x funnel stage (long format) for plotting / BI tools."""
    stages = {"n_calls": "1_call", "n_confirmed": "2_confirmed_incident", "n_reported": "3_filed_report"}
    long = wide.melt(id_vars=id_cols, value_vars=list(stages), var_name="stage", value_name="count")
    long["stage"] = long["stage"].map(stages)
    return long
