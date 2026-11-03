"""Revision R12: are the rising and fading narratives robust to autocorrelation and multiple testing?

Notebook 06 regresses each major community's yearly share (1990–2024, 35 years) on year with HC3 errors and calls
a community rising or fading when p < 0.05, without correcting for the 35 tests. Yearly shares are autocorrelated,
which HC3 ignores. This script
  1. reproduces the published trends (Notebook 06, cell 6),
  2. recomputes the p-values with Newey–West (HAC) errors, lag 3,
  3. applies the Benjamini–Hochberg correction across the 35 communities, to both sets of p-values.

Run from the project root:  python revisions/r12_trend_tests.py
Writes outputs/tables/revisions/r12_trend_tests.csv
"""
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "processed" / "features" / "modeling_table_success_star_power.csv", low_memory=False,
                 usecols=["community_id", "release_year"])
labels = pd.read_csv(ROOT / "outputs" / "tables" / "community_manual_labeling_template_labeled.csv")
MAJOR = labels["community_id"].tolist()
LAB = labels.set_index("community_id")["suggested_label_manual"]
m = df[df["community_id"].isin(MAJOR)]
yearly = pd.crosstab(m["community_id"], m["release_year"], normalize="columns")

rows = []
for c in MAJOR:
    s = yearly.loc[c]
    d = pd.DataFrame({"s": s.values * 100, "year": s.index.values})
    hc3 = smf.ols("s ~ year", d).fit(cov_type="HC3")
    hac = smf.ols("s ~ year", d).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
    resid = hc3.resid.values
    rows.append({"community_id": c, "label": LAB[c], "trend_pp_per_decade": hc3.params["year"] * 10,
                 "p_hc3": hc3.pvalues["year"], "p_hac": hac.pvalues["year"],
                 "lag1_autocorrelation": np.corrcoef(resid[:-1], resid[1:])[0, 1]})
t = pd.DataFrame(rows)
t["q_hc3"] = multipletests(t["p_hc3"], method="fdr_bh")[1]
t["q_hac"] = multipletests(t["p_hac"], method="fdr_bh")[1]

pub = pd.read_csv(ROOT / "outputs" / "tables" / "narrative_dynamics" / "community_trends.csv")
chk = t.merge(pub[["community_id", "trend_pp_per_decade", "p_value"]], on="community_id", suffixes=("", "_pub"))
assert np.allclose(chk["trend_pp_per_decade"], chk["trend_pp_per_decade_pub"], atol=1e-3)
assert np.allclose(chk["p_hc3"], chk["p_value"], atol=1e-3)
print("published trends and HC3 p-values reproduced ✓")

t = t.sort_values("trend_pp_per_decade")
t.round(5).to_csv(ROOT / "outputs" / "tables" / "revisions" / "r12_trend_tests.csv", index=False)
count = lambda p: f"{int(((p < .05) & (t['trend_pp_per_decade'] > 0)).sum())} rising, {int(((p < .05) & (t['trend_pp_per_decade'] < 0)).sum())} fading"
for name in ["p_hc3", "p_hac", "q_hc3", "q_hac"]:
    print(f"{name:6s} < 0.05: {count(t[name])}")
print(f"median lag-1 autocorrelation of residuals: {t['lag1_autocorrelation'].median():.2f}")
pd.set_option("display.width", 200)
print(t[(t["p_hc3"] < .05)][["community_id", "label", "trend_pp_per_decade", "p_hc3", "p_hac", "q_hc3", "q_hac",
                             "lag1_autocorrelation"]].round(4).to_string(index=False))
