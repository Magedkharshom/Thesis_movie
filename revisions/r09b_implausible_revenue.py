"""Revision R9b: do implausibly small reported revenues affect the revenue models?

169 network films (2.2% of those with revenue) report a worldwide revenue below US$10,000, the threshold used
for return on investment; 37 report less than US$100, which are almost certainly placeholders. The full model
M4 is re-estimated for the revenue outcomes without these films. Their revenue percentile is unchanged for the
other films, because it is ranked against the whole market.

Run from the project root:  python revisions/r09b_implausible_revenue.py
Writes outputs/tables/revisions/r09b_implausible_revenue.csv
"""
from pathlib import Path
import pandas as pd
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "processed" / "features" / "modeling_table_success_star_power.csv", low_memory=False)
NET = "z_novelty + z_novelty_sq + z_log_betweenness + z_participation + z_clustering"
F = ("C(release_year) + C(primary_genre_model) + z_runtime + is_english + " + NET +
     " + z_star_index + z_director_index + has_star + C(community_group, Treatment('minor'))")
rows = []
for y in ["revenue_percentile_year", "log_revenue"]:
    for label, d in [("all films with revenue (published)", df), ("revenue at least US$10,000", df[df["revenue_clean"] >= 10_000])]:
        d = d.dropna(subset=[y, "z_runtime", "z_star_index", "z_novelty", "z_participation", "z_clustering"])
        r = smf.ols(f"{y} ~ {F}", d).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d["community_group"])[0]})
        for t in ["z_novelty", "z_novelty_sq", "z_log_betweenness", "z_participation", "z_clustering"]:
            rows.append({"outcome": y, "sample": label, "n": int(r.nobs), "term": t, "coef": r.params[t], "p_value": r.pvalues[t]})
out = pd.DataFrame(rows)
out.round(5).to_csv(ROOT / "outputs" / "tables" / "revisions" / "r09b_implausible_revenue.csv", index=False)
print(out.pivot_table(index=["outcome", "term"], columns="sample", values=["coef", "p_value"]).round(4).to_string())
