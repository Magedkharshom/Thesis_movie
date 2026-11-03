"""Revision R7: are "similar fates" explained by language, star power or visibility rather than by plot?

Notebook 06 computes Moran's I on the plot network after adjusting each outcome for release year and primary
genre only. Here the outcome is adjusted step by step:
  (a) year + primary genre (published);
  (b) + runtime, English-language original, cast and director star power;
  (c) + original language (every language with at least 50 network films; the rest pooled);
  (d) + log number of votes (visibility; not used for the popularity outcome, which it largely measures);
  (e) + narrative community (the 35 major communities and the pooled minor group), which asks whether outcomes
      are similar between neighbours beyond the shared community.
Moran's I, the permutation test (499 reshufflings) and the edge set are exactly those of Notebook 06.

Run from the project root:  python revisions/r07_morans_i_adjustments.py
Writes outputs/tables/revisions/r07_morans_i_adjustments.csv
"""
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "processed" / "features" / "modeling_table_success_star_power.csv", low_memory=False)
edges = pd.read_csv(ROOT / "processed" / "network" / "semantic_network_edges_mutual_knn.csv",
                    usecols=["source_node_id", "target_node_id"])
mm = 10_000
ok = (df["revenue_clean"] >= mm) & (df["budget_clean"] >= mm)
df["log_roi"] = np.where(ok, np.log(df["revenue_clean"].where(ok) / df["budget_clean"].where(ok)), np.nan)
df["primary_genre_model"] = df["primary_genre_model"].fillna("Other")
lang = df["original_language"].fillna("xx")
df["language"] = lang.where(lang.map(lang.value_counts()) >= 50, "other")
df["log_votes"] = np.log1p(df["vote_count"])

pos = pd.Series(np.arange(len(df)), index=df["node_id"])
a = edges["source_node_id"].map(pos).values
b = edges["target_node_id"].map(pos).values
keep = ~(np.isnan(a) | np.isnan(b))
a, b = a[keep].astype(int), b[keep].astype(int)

STEPS = {
    "(a) year + genre (published)": "C(release_year) + C(primary_genre_model)",
    "(b) + runtime, English, star power": "C(release_year) + C(primary_genre_model) + z_runtime + is_english"
                                          " + z_star_index + z_director_index + has_star",
    "(c) + original language": "C(release_year) + C(primary_genre_model) + z_runtime + is_english"
                               " + z_star_index + z_director_index + has_star + C(language)",
    "(d) + log votes": "C(release_year) + C(primary_genre_model) + z_runtime + is_english"
                       " + z_star_index + z_director_index + has_star + C(language) + log_votes",
    "(e) + narrative community": "C(release_year) + C(primary_genre_model) + z_runtime + is_english"
                                 " + z_star_index + z_director_index + has_star + C(language) + log_votes"
                                 " + C(community_group)",
}
OUT = {"revenue_percentile_year": "Revenue percentile", "log_roi": "Log ROI", "vote_average": "Rating",
       "log_popularity": "Log popularity"}
rng = np.random.default_rng(42)
rows = []
for y, yname in OUT.items():
    for step, rhs in STEPS.items():
        if y == "log_popularity" and "log_votes" in rhs:
            rhs = rhs.replace(" + log_votes", "")
            step_label = step + " [without votes]"
        else:
            step_label = step
        d = df.dropna(subset=[y, "z_runtime", "z_star_index"])
        resid = pd.Series(np.nan, index=df.index)
        resid.loc[d.index] = smf.ols(f"{y} ~ {rhs}", d).fit().resid
        zz = ((resid - resid.mean()) / resid.std()).values
        m_ = ~(np.isnan(zz[a]) | np.isnan(zz[b]))
        ea, eb = a[m_], b[m_]
        present = ~np.isnan(zz)
        vv = np.where(present, zz, 0.0)
        n_ = present.sum()
        I = lambda v: (n_ / len(ea)) * np.sum(v[ea] * v[eb]) / np.sum(v[present] ** 2)
        obs = I(vv)
        idx = np.where(present)[0]
        null = np.empty(499)
        for i in range(499):
            p_ = vv.copy(); p_[idx] = vv[rng.permutation(idx)]
            null[i] = I(p_)
        rows.append({"outcome": yname, "adjustment": step_label, "films": int(n_), "morans_I": obs,
                     "null_p95": np.quantile(null, 0.95), "p_value": (np.sum(null >= obs) + 1) / 500})
    print(f"{yname} done")
res = pd.DataFrame(rows)
res.round(4).to_csv(ROOT / "outputs" / "tables" / "revisions" / "r07_morans_i_adjustments.csv", index=False)
pd.set_option("display.width", 200)
print(res.pivot(index="adjustment", columns="outcome", values="morans_I")[list(OUT.values())].round(3))
print("max null p95:", res["null_p95"].max().round(4), "| max p:", res["p_value"].max())
