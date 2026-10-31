"""Revision R5: specification details the examiner asked about.

  A. Louvain: modularity of single runs, agreement between single runs and with the consensus partition, and the
     modularity of the consensus partition, all computed in the same way (igraph multilevel, weighted, resolution 1,
     as in Notebook 03) on the analyzed network.
  B. Language of the overviews: detected with langdetect (fixed seed) instead of the Latin-letter share used in
     Notebook 02, which cannot see non-English text written in Latin script.
  C. Budget check: the full model on films that report a budget, with and without log budget, so that the change
     in the coefficients is not confounded with the change of sample.
  D. Outcome sets: the nested models M1–M4 for log return on investment, and the timing model T3 for log revenue,
     reported as supplementary results outside the 36-test family.

Run from the project root:  python revisions/r05_specification_details.py
Writes outputs/tables/revisions/r05_*.csv
"""
import random
from itertools import combinations
from pathlib import Path
import igraph as ig
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from langdetect import DetectorFactory, detect
from sklearn.metrics import adjusted_rand_score as ari

ROOT = Path(__file__).resolve().parents[1]
FEAT, NETD = ROOT / "processed" / "features", ROOT / "processed" / "network"
OUT_DIR = ROOT / "outputs" / "tables" / "revisions"
OUT_DIR.mkdir(parents=True, exist_ok=True)
pd.set_option("display.width", 220)
df = pd.read_csv(FEAT / "modeling_table_success_star_power.csv", low_memory=False)

# ── A. Louvain runs ───────────────────────────────────────────────────────────────────────────────
edges = pd.read_csv(NETD / "semantic_network_edges_mutual_knn.csv",
                    usecols=["source_node_id", "target_node_id", "cosine_similarity"])
pos = pd.Series(np.arange(len(df)), index=df["node_id"])
e = np.c_[edges["source_node_id"].map(pos), edges["target_node_id"].map(pos)].astype(int)
w = edges["cosine_similarity"].tolist()
g = ig.Graph(n=len(df), edges=e.tolist())
runs, mods = [], []
for seed in range(20):
    random.seed(seed)
    part = g.community_multilevel(weights=w, resolution=1.0)
    runs.append(np.array(part.membership)); mods.append(g.modularity(part.membership, weights=w))
cons = pd.factorize(df["community_id"])[0]
pair_ari = [ari(a, b) for a, b in combinations(runs, 2)]
vs_cons = [ari(r, cons) for r in runs]
A = pd.DataFrame([
    ("single runs: modularity, mean", np.mean(mods)), ("single runs: modularity, min", np.min(mods)),
    ("single runs: modularity, max", np.max(mods)),
    ("single runs: communities, mean", np.mean([r.max() + 1 for r in runs])),
    ("single runs vs each other: ARI, mean", np.mean(pair_ari)),
    ("single runs vs each other: ARI, min", np.min(pair_ari)), ("single runs vs each other: ARI, max", np.max(pair_ari)),
    ("single runs vs consensus: ARI, mean", np.mean(vs_cons)),
    ("consensus partition: modularity", g.modularity(cons.tolist(), weights=w)),
    ("consensus partition: communities", cons.max() + 1),
], columns=["measure", "value"])
A.round(4).to_csv(OUT_DIR / "r05_louvain_runs.csv", index=False)
print("A. Louvain (20 single runs, same settings as Notebook 03):\n", A.round(4).to_string(index=False))

# ── B. language of the overviews ─────────────────────────────────────────────────────────────────
DetectorFactory.seed = 0
ov = pd.read_csv(ROOT / "processed" / "core" / "df_network_corpus.csv", usecols=["tmdb_id", "overview"]).drop_duplicates("tmdb_id")
def lang(t):
    try:
        return detect(t)
    except Exception:
        return "unknown"
ov["detected"] = ov["overview"].fillna("").map(lang)
B = ov["detected"].value_counts().rename("films").to_frame()
B["share"] = B["films"] / len(ov)
B.round(5).to_csv(OUT_DIR / "r05_overview_language.csv")
non_en = ov[ov["detected"].ne("en")].merge(df[["tmdb_id", "community_id"]], on="tmdb_id", how="left")
non_en[["tmdb_id", "detected", "community_id", "overview"]].to_csv(OUT_DIR / "r05_non_english_overviews.csv", index=False)
print(f"\nB. Overviews detected as not English: {len(non_en)} of {len(ov):,} ({len(non_en) / len(ov):.2%})")
print(B.head(8).round(4).to_string())
print("network films among them:", non_en["community_id"].notna().sum())

# ── C. budget check on a fixed sample ────────────────────────────────────────────────────────────
NET = "z_novelty + z_novelty_sq + z_log_betweenness + z_participation + z_clustering"
CTRL = "C(release_year) + C(primary_genre_model) + z_runtime + is_english"
STAR = "z_star_index + z_director_index + has_star"
COMM = "C(community_group, Treatment('minor'))"
M4 = f"{CTRL} + {NET} + {STAR} + {COMM}"
NEED = ["release_year", "primary_genre_model", "z_runtime", "is_english", "z_star_index", "z_director_index",
        "has_star", "community_group", "z_novelty", "z_novelty_sq", "z_log_betweenness", "z_participation", "z_clustering"]
fit = lambda f, d: smf.ols(f, d).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d["community_group"])[0]})
C_rows = []
for y, yname in {"revenue_percentile_year": "Revenue percentile", "vote_average": "Rating"}.items():
    full = df.dropna(subset=[y] + NEED)
    bud = full.dropna(subset=["log_budget"])
    for spec, d, f in [("all films, no budget (published main)", full, M4),
                       ("budget sample, no budget control", bud, M4),
                       ("budget sample, + log budget (published check)", bud, M4 + " + log_budget")]:
        r = fit(f"{y} ~ {f}", d)
        for t in ["z_log_betweenness", "z_participation", "z_clustering"]:
            C_rows.append({"outcome": yname, "specification": spec, "n": int(r.nobs), "term": t,
                           "coef": r.params[t], "p_value": r.pvalues[t]})
budget_tab = pd.DataFrame(C_rows)
budget_tab.round(5).to_csv(OUT_DIR / "r05_budget_fixed_sample.csv", index=False)
print("\nC. Budget check on a fixed sample:\n", budget_tab.round(4).to_string(index=False))

# ── D. supplementary outcomes ────────────────────────────────────────────────────────────────────
mm = 10_000
ok = (df["revenue_clean"] >= mm) & (df["budget_clean"] >= mm)
df["log_roi"] = np.where(ok, np.log(df["revenue_clean"].where(ok) / df["budget_clean"].where(ok)), np.nan)
d = df.dropna(subset=["log_roi"] + NEED)
specs = {"M1": CTRL, "M2": f"{CTRL} + {NET}", "M3": f"{CTRL} + {NET} + {STAR}", "M4": M4}
res = {m: fit(f"log_roi ~ {f}", d) for m, f in specs.items()}
wt = res["M4"].wald_test(", ".join(f"{t} = 0" for t in NET.split(" + ")), scalar=True)
D1 = pd.DataFrame([{"model": m, "n": int(r.nobs), "r2": r.rsquared} for m, r in res.items()])
D1.loc[len(D1)] = {"model": "network block joint p (M4)", "n": int(res["M4"].nobs), "r2": float(wt.pvalue)}
D1c = pd.DataFrame([{"term": t, "coef": res["M4"].params[t], "p_value": res["M4"].pvalues[t]}
                    for t in NET.split(" + ") + STAR.split(" + ")])
tf = pd.read_csv(FEAT / "time_aware_plot_features.csv", usecols=["tmdb_id", "crowding_z", "foresight_z", "competition_z"])
dt = df.merge(tf, on="tmdb_id", how="left").dropna(subset=["log_revenue", "crowding_z", "foresight_z", "competition_z"] + NEED)
BASE_T = f"{CTRL} + {STAR} + C(community_group)"
rt = fit(f"log_revenue ~ {BASE_T} + crowding_z + I(crowding_z**2) + foresight_z + competition_z", dt)
D2 = pd.DataFrame([{"term": t, "coef": rt.params[t], "p_value": rt.pvalues[t], "n": int(rt.nobs)}
                   for t in ["crowding_z", "I(crowding_z ** 2)", "foresight_z", "competition_z"]])
D1.round(5).to_csv(OUT_DIR / "r05_nested_models_log_roi_fit.csv", index=False)
D1c.round(5).to_csv(OUT_DIR / "r05_nested_models_log_roi_coefficients.csv", index=False)
D2.round(5).to_csv(OUT_DIR / "r05_timing_log_revenue.csv", index=False)
print("\nD1. Nested models for log ROI:\n", D1.round(4).to_string(index=False), "\n", D1c.round(4).to_string(index=False))
print("\nD2. Timing model T3 for log revenue:\n", D2.round(4).to_string(index=False))
