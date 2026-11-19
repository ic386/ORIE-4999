# %% [markdown]
# # 01 · Data-dictionary review
# Run `make sample` (or `make fetch`) and `make dictionary` first.
# Click "Run Cell" above each `# %%` in VS Code (needs the Jupyter extension + ipykernel).

# %%
import pandas as pd

from nola import config

pd.set_option("display.max_rows", 200, "display.max_colwidth", 120)
dd = pd.read_csv(config.TABLES / "data_dictionary.csv")
dd.groupby("source")["dataset_id"].nunique()

# %% [markdown]
# ## Schema drift: which fields appear in which year?
# Blank = missing that year. Watch for type changes (calendar_date -> text in CFS 2025)
# and renamed EPR fields (person_type -> persontype in 2025).

# %%
for src in ("calls_for_service", "epr", "service_311"):
    print(src)
    display(pd.read_csv(config.TABLES / "schema_drift_{}.csv".format(src), index_col=0))  # noqa: F821

# %% [markdown]
# ## Key fields to confirm
# - CFS: `nopd_item`, `disposition`, `selfinitiated`, `timearrive` (null share), `mapx/mapy`, `location`
# - EPR: `item_number`, `report_type`, `signal_type`, `disposition`
# - 311: `status` (Duplicate Request), `date_created`, `latitude/longitude`

# %%
key = ["nopd_item", "disposition", "selfinitiated", "timearrive", "mapx", "location",
       "item_number", "report_type", "signal_type", "status", "date_created", "latitude"]
dd[dd["field_name"].isin(key)][["source", "year", "field_name", "data_type", "non_null", "null", "top_values"]]
