# %% [markdown]
# # 02 · Feasibility: CFS–EPR match rate, dispositions, code stability
# Run `make clean-data join qa` first. These numbers decide how hard we can lean on the funnel.

# %%
import pandas as pd

from nola import config

T = config.TABLES
pd.read_csv(T / "match_cfs_overall.csv", index_col=0)

# %% [markdown]
# ## Match rate by year and by disposition
# Expect RTF ("report to follow") near 100% matched, GOA/UNF near 0%. If not, the join
# key or the disposition meaning has shifted - flag to Ryan.

# %%
display(pd.read_csv(T / "match_cfs_by_year.csv", index_col=0))  # noqa: F821
display(pd.read_csv(T / "match_disposition_vs_epr.csv", index_col=0))  # noqa: F821

# %% [markdown]
# ## Reverse direction: reports with no call record (walk-ins, online reports, follow-ups)

# %%
pd.read_csv(T / "match_epr_in_cfs_by_year.csv", index_col=0)

# %% [markdown]
# ## Code stability by year (`unstable = True` -> code not present every year)

# %%
for col in ("disposition", "priority"):
    print(col)
    display(pd.read_csv(T / "qa_cfs_{}_by_year.csv".format(col), index_col=0).head(25))  # noqa: F821
pd.read_csv(T / "qa_cfs_type_text_jaccard.csv")

# %% [markdown]
# ## Completeness by year (geocoding, beats, missing arrivals, unmapped call types)

# %%
pd.read_csv(T / "qa_cfs_completeness_by_year.csv", index_col=0)

# %% [markdown]
# ## Call-type crosswalk review -> edit `config/funnel.yaml` / `config/call_type_crosswalk.csv`

# %%
cw = pd.read_csv(T / "qa_call_type_crosswalk_review.csv", index_col=0)
cw[cw["category"].isin(["other", "unmapped"])].head(40)
