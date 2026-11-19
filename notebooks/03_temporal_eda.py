# %% [markdown]
# # 03 · Temporal patterns: heatmaps, holiday lift, seasonality
# Run `make calendar temporal` first. Figures land in outputs/figures/.

# %%
import pandas as pd
from IPython.display import Image, display

from nola import config

F, T = config.FIGURES, config.TABLES
display(Image(F / "heatmap_all.png"), Image(F / "heatmap_resident.png"))

# %% [markdown]
# ## Holiday lift
# Ratio: event-day volume ÷ same weekday ±4 weeks (clean days only).
# Regression: NB GLM with weekday, month, year fixed effects; IRR = multiplicative lift.

# %%
display(pd.read_csv(T / "holiday_lift_ratio.csv"))
pd.read_csv(T / "holiday_lift_regression_irr.csv")

# %% [markdown]
# ## Mardi Gras by day offset (parade pattern is fixed relative to Fat Tuesday)

# %%
daily = pd.read_csv(T / "daily_call_counts.csv", index_col=0, parse_dates=True)["count"]
cal = pd.read_parquet(config.PROCESSED / "calendar.parquet").set_index("date")
d = cal.join(daily.rename("count"))
d[d["days_to_mardi_gras"].between(0, 30)].groupby("days_to_mardi_gras")["count"].mean().sort_index(ascending=False)

# %%
display(Image(F / "seasonality_decomposition.png"))
