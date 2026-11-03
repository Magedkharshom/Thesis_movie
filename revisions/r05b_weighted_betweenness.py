"""Revision R5b: does computing betweenness on similarity-weighted paths change the results?

Notebook 03 computes exact betweenness on the unweighted network (each edge is one step). Here it is recomputed
with edge lengths 1 - cosine similarity (at least 1e-6, for the few near-identical overviews), so that paths through more similar plots are shorter, and the full model
M4 is re-estimated with the weighted measure in place of the unweighted one.

Run from the project root:  python revisions/r05b_weighted_betweenness.py
Writes outputs/tables/revisions/r05b_weighted_betweenness.csv
"""
from pathlib import Path
import igraph as ig
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "processed" / "features" / "modeling_table_success_star_power.csv", low_memory=False)
edges = pd.read_csv(ROOT / "processed" / "network" / "semantic_network_edges_mutual_knn.csv",
                    usecols=["source_node_id", "target_node_id", "cosine_similarity"])
pos = pd.Series(np.arange(len(df)), index=df["node_id"])
e = np.c_[edges["source_node_id"].map(pos), edges["target_node_id"].map(pos)].astype(int)
g = ig.Graph(n=len(df), edges=e.tolist())
n = len(df)
bw = np.array(g.betweenness(directed=False, weights=(1 - edges["cosine_similarity"]).clip(lower=1e-6).tolist())) / ((n - 1) * (n - 2) / 2)
df["log_betw_w"] = np.log1p(bw * n)
df["z_log_betw_w"] = (df["log_betw_w"] - df["log_betw_w"].mean()) / df["log_betw_w"].std()
rho = df[["log_betweenness", "log_betw_w"]].corr(method="spearman").iloc[0, 1]
print(f"Spearman rho, unweighted vs weighted log betweenness: {rho:.3f}")

CTRL = "C(release_year) + C(primary_genre_model) + z_runtime + is_english"
STAR = "z_star_index + z_director_index + has_star"
COMM = "C(community_group, Treatment('minor'))"
rows = []
for y in ["log_revenue", "revenue_percentile_year", "vote_average", "log_popularity"]:
    for label, b in [("unweighted (published)", "z_log_betweenness"), ("similarity-weighted", "z_log_betw_w")]:
        f = f"{y} ~ {CTRL} + z_novelty + z_novelty_sq + {b} + z_participation + z_clustering + {STAR} + {COMM}"
        d = df.dropna(subset=[y, "z_runtime", "z_star_index", "z_novelty", "z_participation", "z_clustering"])
        r = smf.ols(f, d).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d["community_group"])[0]})
        for t in [b, "z_participation", "z_clustering"]:
            rows.append({"outcome": y, "betweenness": label, "term": t.replace(b, "betweenness"),
                         "coef": r.params[t], "p_value": r.pvalues[t], "n": int(r.nobs)})
out = pd.DataFrame(rows)
out.round(5).to_csv(ROOT / "outputs" / "tables" / "revisions" / "r05b_weighted_betweenness.csv", index=False)
print(out.round(4).to_string(index=False))
